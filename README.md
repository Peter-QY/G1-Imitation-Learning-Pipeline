# GVHMR+GMR+Beyondmimic全流程

## 一、GVHMR

安装及环境配置参照https://github.com/zju3dv/GVHMR/

**视频转换数据**

```shell
python tools/demo/demo.py --video=docs/example_video/tennis.mp4 -s
```

 其中，--video=/path/to/video，为输入（即需要处理）的视频路径

![img](https://i-blog.csdnimg.cn/direct/13276126ee264824b8caeeac2f94970f.png)

## 二、GMR

安装和环境配置参照https://github.com/YanjieZe/GMR

**将数据转换重定向，并转换成csv格式**

```shell
python scripts/gvhmr_to_robot.py --gvhmr_pred_file <path_to_hmr4d_results.pt> --robot unitree_g1 --record_video --save_path motions/G1/G1.pkl
```

其中，--gvhmr_pred_file就是上面GVHMR生成的pt文件的路径，--save_path为转化出来的pkl的保存路径。

转换成功之后，会进行数据重定向视频播放，注意观察重定向效果。

```shell
python3 scripts/batch_gmr_pkl_to_csv.py --folder /home/teacher/zzw/GMR/motions/G1/
```

其中，--folder为pkl保存的文件夹路径。

## 三、Beyondmimmic

安装和环境配置参照https://github.com/HybridRobotics/whole_body_tracking

**首先需要注册wandb**：https://wandb.ai/home，这里有点麻烦，多问ai。

**将数据转成npz格式**

```shell
python scripts/csv_to_npz.py --input_file {motion_name}.csv --input_fps 30 --output_name {motion_name} --headless
```

其中，--input_file为本地保存的csv文件路径，--output_name为wandb保存的run名称（注意只有名字不要后缀和路径）。

**如果保存至tmp下，移动到motion下**

```shell
mv /tmp/kunkun.npz ~/qy_ws/Beyondmimic/.../motion
```

如果脚本未自动上传，手动上传，脚本见manual_upload.py，运行python manual_upload.py即可

```python
# 配置信息
PROJECT_NAME = "my-g1-motions"  # 修改项目名，避开保留前缀
ARTIFACT_NAME = "kunkun" 
FILE_PATH = "motions/kunkun.npz"
```

**配置信息修改后可上传**

为了防止从wandb下载失败，我们修改scripts/replay_npz.py和scripts/rsl_rl/train.py

```python
# 原代码长这样：
api = wandb.Api()
artifact = api.artifact(args_cli.registry_name)
motion_file = str(pathlib.Path(artifact.download()) / "motion.npz")
```

修改为

replay.py（大约第 77-80 行）

```python
================= [修改开始] 自动回退逻辑 =================
import os  # 确保头部已经导入 os，如果没有请在这里导入

# 定义你的本地文件绝对路径 (硬编码作为兜底)
LOCAL_FALLBACK_PATH = "/home/yiqiu/qy_ws/Beyondmimic/whole_body_tracking/motions/kunkun.npz"

try:
    print(f"正在尝试从 WandB 下载 Artifact: {args_cli.registry_name} ...")
    # 尝试连接云端
    api = wandb.Api()
    artifact = api.artifact(args_cli.registry_name)
    # 尝试下载
    download_dir = artifact.download()
    motion_file = str(pathlib.Path(download_dir) / "motion.npz")
    print(f"WandB 下载成功，使用文件: {motion_file}")
    
except Exception as e:
    print(f"\n[警告] WandB 连接或下载失败: {e}")
    print(">>> 正在尝试使用本地文件作为替补...")
    
    if os.path.exists(LOCAL_FALLBACK_PATH):
        motion_file = LOCAL_FALLBACK_PATH
        print(f"!!! 成功切换至本地文件: {motion_file} !!!\n")
    else:
        # 如果本地也没文件，那就真的抛出错误
        raise FileNotFoundError(f"云端下载失败，且本地路径未找到文件: {LOCAL_FALLBACK_PATH}")
# ================= [修改结束] =================
```

------

 train.py（大约第 116-121 行）

```python
================= [修改开始] 自动回退逻辑 =================
# load the motion file from the wandb registry
registry_name = args_cli.registry_name
if ":" not in registry_name:
    registry_name += ":latest"

import pathlib
import wandb
import os 

# 定义本地兜底路径
LOCAL_FALLBACK_PATH = "/home/yiqiu/qy_ws/Beyondmimic/whole_body_tracking/motions/kunkun.npz"

try:
    print(f"[INFO] 尝试连接 WandB 下载: {registry_name}")
    api = wandb.Api()
    artifact = api.artifact(registry_name)
    download_path = artifact.download()
    env_cfg.commands.motion.motion_file = str(pathlib.Path(download_path) / "motion.npz")
    print(f"[INFO] WandB 下载成功，路径: {env_cfg.commands.motion.motion_file}")

except Exception as e:
    print(f"[WARN] WandB 下载失败 (可能是网络原因): {e}")
    print(f"[INFO] 尝试加载本地文件: {LOCAL_FALLBACK_PATH}")
    
    if os.path.exists(LOCAL_FALLBACK_PATH):
        env_cfg.commands.motion.motion_file = LOCAL_FALLBACK_PATH
        print(f"[INFO] !!! 已启用本地文件回退模式 !!!")
    else:
        raise FileNotFoundError(f"WandB 失败且本地文件不存在: {LOCAL_FALLBACK_PATH}")
# ================= [修改结束] =================
```

修改完成后，检查动作是否正确

```shell
python scripts/replay_npz.py --registry_name=yiqiupeter-nwpu/my-g1-motions/kunkun:latest
```

其中，--registry_name={your-organization}-org/wandb-registry-motions/{motion_name}

![image-20260130232109580](/home/yiqiu/.config/Typora/typora-user-images/image-20260130232109580.png)

确认无误后，开始训练

```shell
python scripts/rsl_rl/train.py --task=Tracking-Flat-G1-v0 \
--registry_name=yiqiupeter-nwpu/my-g1-motions/kunkun:latest \
--headless --logger wandb \
--log_project_name G1_Kunkun_Auto \
--run_name kunkun_v1
```

当参数

![截图 2026-01-30 23-30-43](/home/yiqiu/图片/截图/截图 2026-01-30 23-30-43.png)

训练完成后，查看效果

将play.py中的

```python
    if args_cli.motion_file is not None:
            print(f"[INFO]: Using motion file from CLI: {args_cli.motion_file}")
            env_cfg.commands.motion.motion_file = args_cli.motion_file
    art = next((a for a in wandb_run.used_artifacts() if a.type == "motions"), None)
    if art is None:
        print("[WARN] No model artifact found in the run.")
    else:
        env_cfg.commands.motion.motion_file = str(pathlib.Path(art.download()) / "motion.npz")

else:
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
```

修改为

```python
[WandB 模式]  
  # 1. 优先使用命令行参数
    if args_cli.motion_file is not None:
        print(f"[INFO]: Using motion file from CLI: {args_cli.motion_file}")
        env_cfg.commands.motion.motion_file = args_cli.motion_file
    else:
        # 2. 如果没传参数，才尝试从 WandB 下载
        art = next((a for a in wandb_run.used_artifacts() if a.type == "motions"), None)
        if art is None:
            print("[WARN] No model artifact found in the run.")
        else:
            try:
                env_cfg.commands.motion.motion_file = str(pathlib.Path(art.download()) / "motion.npz")
            except Exception as e:
                print(f"[WARN] WandB download failed: {e}")
                # 3. 如果下载失败，回退到本地硬编码路径
                fallback = "/home/yiqiu/qy_ws/Beyondmimic/whole_body_tracking/motions/kunkun.npz"
                if os.path.exists(fallback):
                    print(f"[INFO] Using local fallback: {fallback}")
                    env_cfg.commands.motion.motion_file = fallback

else:
        # [修改] 绕过自动搜索，直接指定你图片里的那个文件的绝对路径
        # 请确保下面的路径和你截图里的完全一致
        resume_path = "/home/yiqiu/qy_ws/Beyondmimic/whole_body_tracking/logs/rsl_rl/g1_flat/2026-01-26_13-55-47_motion_4060_continued/model_5000.pt"
        
        print(f"[INFO] !!! 强制加载本地模型文件 !!!")
        print(f"[INFO] 路径: {resume_path}")
        
        # 确保文件真的存在，不存在就报错
        if not os.path.exists(resume_path):
            raise FileNotFoundError(f"严重错误：找不到模型文件，请检查路径是否写错: {resume_path}")

        # [保持我们之前加的动作文件逻辑]
        if args_cli.motion_file is not None:
            print(f"[INFO]: Using motion file from CLI: {args_cli.motion_file}")
            env_cfg.commands.motion.motion_file = args_cli.motion_file
        else:
            # 本地兜底动作文件
            local_motion = "/home/yiqiu/qy_ws/Beyondmimic/whole_body_tracking/motions/tennis.npz" # 你这里想看 tennis 还是 kunkun？记得改
            if os.path.exists(local_motion):
                env_cfg.commands.motion.motion_file = local_motion
                print(f"[INFO] 使用本地动作文件: {local_motion}")
    
    # [修改] 新增：本地模式下的动作文件加载逻辑
    if args_cli.motion_file is not None:
        # 1. 优先使用命令行参数 --motion_file
        print(f"[INFO]: Using motion file from CLI: {args_cli.motion_file}")
        env_cfg.commands.motion.motion_file = args_cli.motion_file
    else:
        # 2. 使用本地硬编码路径作为兜底
        local_fallback = "/home/yiqiu/qy_ws/Beyondmimic/whole_body_tracking/motions/kunkun.npz"
        if os.path.exists(local_fallback):
            print(f"[INFO]: Using local fallback motion file: {local_fallback}")
            env_cfg.commands.motion.motion_file = local_fallback
        else:
            print(f"[WARN] No motion file specified and fallback not found at {local_fallback}")
```

运行

```shell
python scripts/rsl_rl/play.py \
--task=Tracking-Flat-G1-v0 \
--num_envs=1 \
--load_run {你的时间戳文件夹名} \
--checkpoint 5000 \#第多少次迭代文件
--motion_file /home/yiqiu/qy_ws/Beyondmimic/whole_body_tracking/motions/kunkun.npz#路径
```

展现出如下效果即可

![截图 2026-01-30 23-53-27](/home/yiqiu/图片/截图/截图 2026-01-30 23-53-27.png)

## 四、Sim2real
