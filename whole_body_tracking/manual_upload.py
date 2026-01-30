import wandb
import os

# 配置信息
PROJECT_NAME = "my-g1-motions"  # 修改项目名，避开保留前缀
ARTIFACT_NAME = "kunkun" 
FILE_PATH = "motions/kunkun.npz" 

def upload():
    # 初始化
    run = wandb.init(project=PROJECT_NAME, job_type="dataset-upload")
    
    # 创建 Artifact
    artifact = wandb.Artifact(name=ARTIFACT_NAME, type="dataset")
    
    # [关键] 必须映射为 "motion.npz" 以适配 train.py 的硬编码逻辑
    if os.path.exists(FILE_PATH):
        artifact.add_file(FILE_PATH, name="motion.npz")
        run.log_artifact(artifact)
        print(f"成功上传！Artifact 路径为: {run.entity}/{PROJECT_NAME}/{ARTIFACT_NAME}")
    else:
        print(f"错误：找不到文件 {FILE_PATH}")
    
    run.finish()

if __name__ == "__main__":
    upload()
