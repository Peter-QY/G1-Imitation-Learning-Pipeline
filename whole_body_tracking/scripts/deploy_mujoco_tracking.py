import time
import argparse
import numpy as np
import mujoco
import mujoco.viewer
import onnxruntime as ort
from scipy.spatial.transform import Rotation as R
from collections import deque

# ==========================================
# 👇 用户配置区域 (请确保这 3 个路径正确)
# ==========================================
# 1. 机器人模型
XML_PATH = "source/whole_body_tracking/whole_body_tracking/assets/unitree_description/mjcf/g1.xml" 

# 2. 策略模型 (ONNX)
POLICY_PATH = "logs/rsl_rl/g1_flat/2026-01-26_13-55-47_motion_4060_continued/exported/policy.onnx"

# 3. 动作文件 (NPZ)
MOTION_PATH = "motions/motion.npz" 
# ==========================================

class G1Sim2Sim:
    def __init__(self, xml_path, policy_path, motion_path):
        # 1. 加载 MuJoCo 模型
        print(f"[Info] Loading MuJoCo model from: {xml_path}")
        self.m = mujoco.MjModel.from_xml_path(xml_path)
        self.d = mujoco.MjData(self.m)
        
        # 减去 7 个自由度 (Root: 3 pos + 4 quat)
        self.model_num_joints = self.m.nq - 7
        print(f"[Info] Robot Model expects {self.model_num_joints} joints.")
        
        # 检测是否有 Actuators
        self.has_actuators = self.m.nu > 0
        if not self.has_actuators:
            print("[Warning] XML defines 0 actuators. Switching to direct joint torque control (qfrc_applied).")

        # 2. 加载 ONNX 策略
        print(f"[Info] Loading ONNX policy from: {policy_path}")
        self.sess = ort.InferenceSession(policy_path)
        
        # 3. 加载动作数据
        print(f"[Info] Loading Motion from: {motion_path}")
        self.motion_data = np.load(motion_path)
        
        try:
            root_pos = self.motion_data['body_pos_w']   # (T, 30, 3)
            root_quat = self.motion_data['body_quat_w'] # (T, 30, 4)
            joint_pos = self.motion_data['joint_pos']   # (T, 29)

            if root_pos.ndim == 3: root_pos = root_pos[:, 0, :]
            if root_quat.ndim == 3: root_quat = root_quat[:, 0, :]
            
            # 截断多余关节 (比如 XML 是 23, NPZ 是 29)
            if joint_pos.shape[-1] > self.model_num_joints:
                joint_pos = joint_pos[:, :self.model_num_joints]
                
            if joint_pos.ndim > 2: joint_pos = joint_pos.reshape(joint_pos.shape[0], -1)

            print(f"[Debug] Final Shapes -> Root: {root_pos.shape}, Quat: {root_quat.shape}, Joints: {joint_pos.shape}")
            self.ref_qpos = np.concatenate([root_pos, root_quat, joint_pos], axis=-1)
            print(f"[Success] Loaded motion sequence with shape: {self.ref_qpos.shape}")
            
        except Exception as e:
            print(f"[Error] Motion Load Failed: {e}")
            raise

        # 仿真参数
        self.dt = 0.005  
        self.decimation = 4 
        self.sim_duration = 60 
        
        # 默认姿态
        if self.m.nkey > 0:
            self.default_dof_pos = self.m.key_qpos[0][7:]
        else:
            self.default_dof_pos = self.m.qpos0[7:]
            
        self.num_actions = self.model_num_joints 
        self.action_scale = 0.25 
        
        self.action = np.zeros(self.num_actions)

        # 维度补全 (Obs Padding)
        self.target_obs_dim = 160 

    def get_gravity_orientation(self, quaternion):
        r = R.from_quat([quaternion[1], quaternion[2], quaternion[3], quaternion[0]])
        return r.apply([0, 0, -1], inverse=True)

    def get_obs(self):
        qpos = self.d.qpos
        qvel = self.d.qvel
        
        base_ang_vel = qvel[3:6]
        base_quat = qpos[3:7]
        gravity = self.get_gravity_orientation(base_quat)
        
        dof_pos = qpos[7:] - self.default_dof_pos
        dof_vel = qvel[6:]
        
        commands = np.zeros(3) 
        
        # 1. 基础 Observation
        obs = np.concatenate([
            base_ang_vel * 0.25,
            gravity,
            commands, 
            dof_pos * 1.0,
            dof_vel * 0.05,
            self.action
        ]).astype(np.float32)
        
        # 2. 维度补全 (Padding to 160)
        current_dim = obs.shape[0]
        if current_dim < self.target_obs_dim:
            padding_size = self.target_obs_dim - current_dim
            padding = np.zeros(padding_size, dtype=np.float32)
            obs = np.concatenate([obs, padding])
        
        return obs

    def run(self):
        print("\n[Info] Starting Simulation... Press SPACE to pause.")
        
        self.d.qpos[:] = self.ref_qpos[0]
        self.d.qvel[:] = 0.0
        
        with mujoco.viewer.launch_passive(self.m, self.d) as viewer:
            start_time = time.time()
            step_count = 0
            
            while viewer.is_running() and time.time() - start_time < self.sim_duration:
                step_start = time.time()

                if step_count % self.decimation == 0:
                    obs = self.get_obs()
                    
                    input_feed = {}
                    for node in self.sess.get_inputs():
                        if node.name == 'obs':
                            input_feed[node.name] = obs[None, :]
                        elif node.name == 'time_step':
                            input_feed[node.name] = np.array([[float(step_count)]], dtype=np.float32)
                    
                    action_out = self.sess.run(None, input_feed)[0][0]
                    self.action = action_out
                    
                    target_pos = self.action * self.action_scale + self.default_dof_pos
                
                # PD Control Calculation
                kp = 20.0
                kd = 0.5
                current_pos = self.d.qpos[7:]
                current_vel = self.d.qvel[6:]
                torques = kp * (target_pos - current_pos) - kd * current_vel
                
                # ==================== 修复逻辑 (兼容无Actuator模型) ====================
                if self.has_actuators:
                    # 正常写入 ctrl
                    self.d.ctrl[:] = torques
                else:
                    # 如果没有 actuator，直接写入关节力矩 (跳过前6个自由度Root)
                    # qfrc_applied 的维度通常是 nv (35 for G1)，前6个是Base，后面29个是关节
                    self.d.qfrc_applied[6:] = torques
                # ===================================================================
                
                mujoco.mj_step(self.m, self.d)
                viewer.sync()
                step_count += 1
                
                time_until_next = self.m.opt.timestep - (time.time() - step_start)
                if time_until_next > 0:
                    time.sleep(time_until_next)

if __name__ == "__main__":
    sim = G1Sim2Sim(XML_PATH, POLICY_PATH, MOTION_PATH)
    sim.run()