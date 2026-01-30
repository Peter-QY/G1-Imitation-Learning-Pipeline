import numpy as np
import pandas as pd
import os

input_path = "data/raw/cai_g1.npz"
output_path = "cai_g1.csv"

if os.path.exists(input_path):
    print(f"Loading {input_path}...")
    data = np.load(input_path)
    qpos = data['qpos']
    
    # 转 CSV (无表头，无索引，纯数据)
    df = pd.DataFrame(qpos)
    df.to_csv(output_path, header=False, index=False)
    print(f"✅ CSV Ready: {os.path.abspath(output_path)}")
else:
    print(f"❌ Input file not found: {input_path}")
