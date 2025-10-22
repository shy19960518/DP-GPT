import os
import pandas as pd
import numpy as np
from Mahle_Model_parallel import plot, get_next_fuel_and_soc, from_soc_to_fuel
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
import random
import torch
from torch.utils.data import Dataset, DataLoader

from utils import EMSdataset, collate_fn

def random_split_list(lst, num_splits):

    random.shuffle(lst)

    total_len = len(lst)
    avg_len = total_len // num_splits

    splits = []
    start_idx = 0
    for i in range(num_splits - 1):
        splits.append(lst[start_idx:start_idx + avg_len])
        start_idx += avg_len

    splits.append(lst[start_idx:])
    
    return splits


with open('./dataset/zerostart_zeroend_driving_cycle_list.pkl', 'rb') as f:
    zerostart_zeroend_driving_cycle_list = pickle.load(f)


splits = random_split_list(zerostart_zeroend_driving_cycle_list, 30)

print(len(splits[0]))
# assert 1==2

for index, split in enumerate(splits):
    with open(f"./dataset/zerostart_zeroend_driving_cycle_list/VEDdata_{index}.pkl", "wb") as f:
        pickle.dump(split, f)




















# i = 0
# while True:
#     fig, axes = plt.subplots(3, 3, figsize=(12, 10))  # 创建 3x3 的子图网格

#     for idx, ax in enumerate(axes.flat):  # 遍历 3x3 的所有子图
#         segment_idx = i + idx  # 计算当前要绘制的索引
        
#         if segment_idx >= len(total_segments):  # 防止索引超出范围
#             break  

#         ax.plot(total_segments[segment_idx])  # 画出对应的数据
#         ax.set_title(f"Segment {segment_idx}")  # 设置标题
#         ax.set_xlabel("Time")
#         ax.set_ylabel("Value")

#     plt.tight_layout()  # 调整子图间距，使其美观
#     plt.show()

#     i += 8  # 让 i 递增，进入下一批数据

