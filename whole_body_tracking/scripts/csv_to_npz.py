"""This script replay a motion from a csv file and output it to a npz file"""

"""Launch Isaac Sim Simulator first."""

import argparse
import numpy as np
import os
import torch

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Replay motion from csv file and output to npz file.")
parser.add_argument("--input_file", type=str, required=True, help="The path to the input motion csv file.")
parser.add_argument("--input_fps", type=int, default=30, help="The fps of the input motion.")
parser.add_argument("--frame_range", nargs=2, type=int, metavar=("START", "END"), help="frame range: START END")
parser.add_argument("--output_name", type=str, required=True, help="The name of the motion npz file.")
parser.add_argument("--output_fps", type=int, default=50, help="The fps of the output motion.")

# === 修正参数 ===
parser.add_argument("--z_offset", type=float, default=0.0, help="Height offset (meters). Positive = lower the robot.")
parser.add_argument("--roll_offset", type=float, default=0.0, help="Roll offset (degrees). Fix side-to-side tilt.")
parser.add_argument("--pitch_offset", type=float, default=0.0, help="Pitch offset (degrees). Fix forward-backward tilt.")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import axis_angle_from_quat, quat_conjugate, quat_mul, quat_slerp
from whole_body_tracking.robots.g1 import G1_CYLINDER_CFG

@configclass
class ReplayMotionsSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
    sky_light = AssetBaseCfg(prim_path="/World/skyLight", spawn=sim_utils.DomeLightCfg(intensity=750.0, texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr"))
    robot: ArticulationCfg = G1_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

class MotionLoader:
    def __init__(self, motion_file, input_fps, output_fps, device, frame_range):
        self.motion_file = motion_file
        self.input_fps = input_fps
        self.output_fps = output_fps
        self.input_dt = 1.0 / input_fps
        self.output_dt = 1.0 / output_fps
        self.device = device
        self.frame_range = frame_range
        self.current_idx = 0
        self._load_motion()
        self._interpolate_motion()
        self._compute_velocities()

    def _load_motion(self):
        if self.frame_range is None:
            motion = torch.from_numpy(np.loadtxt(self.motion_file, delimiter=","))
        else:
            motion = torch.from_numpy(np.loadtxt(self.motion_file, delimiter=",", skiprows=self.frame_range[0]-1, max_rows=self.frame_range[1]-self.frame_range[0]+1))
        motion = motion.to(torch.float32).to(self.device)
        self.pos = motion[:, :3]
        self.rot = motion[:, 3:7][:, [3, 0, 1, 2]] # xyzw -> wxyz
        self.dof_pos = motion[:, 7:]
        self.input_frames = motion.shape[0]
        self.duration = (self.input_frames - 1) * self.input_dt
        print(f"Motion loaded: {self.input_frames} frames, {self.duration:.2f}s")

    def _interpolate_motion(self):
        times = torch.arange(0, self.duration, self.output_dt, device=self.device, dtype=torch.float32)
        self.output_frames = times.shape[0]
        phase = times / self.duration
        idx0 = (phase * (self.input_frames - 1)).floor().long()
        idx1 = torch.minimum(idx0 + 1, torch.tensor(self.input_frames - 1, device=self.device))
        blend = phase * (self.input_frames - 1) - idx0
        
        self.out_pos = self._lerp(self.pos[idx0], self.pos[idx1], blend.unsqueeze(1))
        self.out_rot = self._slerp(self.rot[idx0], self.rot[idx1], blend)
        self.out_dof_pos = self._lerp(self.dof_pos[idx0], self.dof_pos[idx1], blend.unsqueeze(1))

    def _lerp(self, a, b, t): return a * (1 - t) + b * t
    def _slerp(self, a, b, t):
        res = torch.zeros_like(a)
        for i in range(a.shape[0]): res[i] = quat_slerp(a[i], b[i], t[i])
        return res

    def _compute_velocities(self):
        self.out_lin_vel = torch.gradient(self.out_pos, spacing=self.output_dt, dim=0)[0]
        self.out_dof_vel = torch.gradient(self.out_dof_pos, spacing=self.output_dt, dim=0)[0]
        # Angular velocity
        q_prev, q_next = self.out_rot[:-2], self.out_rot[2:]
        q_rel = quat_mul(q_next, quat_conjugate(q_prev))
        omega = axis_angle_from_quat(q_rel) / (2.0 * self.output_dt)
        self.out_ang_vel = torch.cat([omega[:1], omega, omega[-1:]], dim=0)

    def get_state(self, idx):
        return (self.out_pos[idx:idx+1], self.out_rot[idx:idx+1], self.out_lin_vel[idx:idx+1], 
                self.out_ang_vel[idx:idx+1], self.out_dof_pos[idx:idx+1], self.out_dof_vel[idx:idx+1])

def get_correction_quat(roll, pitch, device):
    r, p, y = torch.tensor(np.radians(roll)), torch.tensor(np.radians(pitch)), torch.tensor(0.0)
    cy, sy = torch.cos(y*0.5), torch.sin(y*0.5)
    cp, sp = torch.cos(p*0.5), torch.sin(p*0.5)
    cr, sr = torch.cos(r*0.5), torch.sin(r*0.5)
    q = torch.tensor([cr*cp*cy + sr*sp*sy, sr*cp*cy - cr*sp*sy, cr*sp*cy + sr*cp*sy, cr*cp*sy - sr*sp*cy], device=device)
    return q.unsqueeze(0)

def run_simulator(sim, scene, joint_names):
    motion = MotionLoader(args_cli.input_file, args_cli.input_fps, args_cli.output_fps, sim.device, args_cli.frame_range)
    robot = scene["robot"]
    robot_joint_indexes = robot.find_joints(joint_names, preserve_order=True)[0]
    
    log = {"fps": [args_cli.output_fps], "joint_pos": [], "joint_vel": [], "body_pos_w": [], "body_quat_w": [], "body_lin_vel_w": [], "body_ang_vel_w": []}
    file_saved = False
    q_fix = get_correction_quat(args_cli.roll_offset, args_cli.pitch_offset, sim.device)
    
    current_idx = 0
    while simulation_app.is_running():
        state = motion.get_state(current_idx)
        pos, rot, lin_vel, ang_vel, dof_pos, dof_vel = state
        
        # === 修正 ===
        pos[:, 2] -= args_cli.z_offset # 高度修正
        rot = quat_mul(q_fix, rot)     # 旋转修正
        
        # 写回
        root_state = robot.data.default_root_state.clone()
        root_state[:, :3] = pos
        root_state[:, :2] += scene.env_origins[:, :2]
        root_state[:, 3:7] = rot
        root_state[:, 7:10] = lin_vel
        root_state[:, 10:] = ang_vel
        robot.write_root_state_to_sim(root_state)
        
        j_pos, j_vel = robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone()
        j_pos[:, robot_joint_indexes] = dof_pos
        j_vel[:, robot_joint_indexes] = dof_vel
        robot.write_joint_state_to_sim(j_pos, j_vel)
        
        sim.render()
        scene.update(sim.get_physics_dt())
        
        if not file_saved:
            log["joint_pos"].append(robot.data.joint_pos[0].cpu().numpy())
            log["joint_vel"].append(robot.data.joint_vel[0].cpu().numpy())
            log["body_pos_w"].append(robot.data.body_pos_w[0].cpu().numpy())
            log["body_quat_w"].append(robot.data.body_quat_w[0].cpu().numpy())
            log["body_lin_vel_w"].append(robot.data.body_lin_vel_w[0].cpu().numpy())
            log["body_ang_vel_w"].append(robot.data.body_ang_vel_w[0].cpu().numpy())
            
        current_idx += 1
        if current_idx >= motion.output_frames:
            if not file_saved:
                file_saved = True
                for k in log: 
                    if isinstance(log[k], list): log[k] = np.stack(log[k])
                save_path = f"/tmp/{args_cli.output_name}.npz"
                np.savez(save_path, **log)
                print(f"[INFO]: Local file saved to: {save_path}")
            current_idx = 0

def main():
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim_cfg.dt = 1.0 / args_cli.output_fps
    sim = SimulationContext(sim_cfg)
    scene = InteractiveScene(ReplayMotionsSceneCfg(num_envs=1, env_spacing=2.0))
    sim.reset()
    run_simulator(sim, scene, [
        "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
        "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
        "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
        "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
        "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint"
    ])

if __name__ == "__main__":
    main()
    simulation_app.close()
