import os
import pandas as pd
import numpy as np
from Mahle_Model_parallel import plot, get_next_fuel_and_soc, from_soc_to_fuel
import matplotlib.pyplot as plt
from tqdm import tqdm

import glob
import pickle
from natsort import natsorted


######################加强高速dataset ########################################

# 原始文件路径模式
source_pattern = "./dataset/buffer/VEDdata_*.pkl"  

# 目标目录
target_root = "./dataset/buffer_out"

# 确保目标目录存在
os.makedirs(target_root, exist_ok=True)

# 遍历所有符合模式的 pkl 文件
for file_path in natsorted(glob.glob(source_pattern)):
    file_name = os.path.basename(file_path)  # 例如 VEDdata_1.pkl

    with open(f'./dataset/buffer/{file_name}', 'rb') as f:
        driving_cycle_list = pickle.load(f)
    new_segments = []
    for driving_cycle in driving_cycle_list:
        max_speed = max(driving_cycle)
        driving_cycle = driving_cycle / max_speed * 36
        
        new_segments.append(driving_cycle)

    with open(f"./dataset/buffer_out/{file_name}", "wb") as f:
        pickle.dump(new_segments, f)