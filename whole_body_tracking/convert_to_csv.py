import numpy as np
import pandas as pd
import os

# 1. 你的输入文件路径
input_path = "data/raw/cxk_g1.npz"
output_path = "cxk_g1.csv" # 临时保存在当前目录

# 2. 加载数据
print(f"Loading {input_path}...")
data = np.load(input_path)
qpos = data['qpos'] # (N, dof)

print(f"Shape: {qpos.shape}")

# 3. 转换为 DataFrame 并保存为 CSV
# Unitree 的 csv 格式通常每一行是一个时间步的关节角度
df = pd.DataFrame(qpos)
df.to_csv(output_path, header=False, index=False)

print(f"✅ Converted to {output_path}")
