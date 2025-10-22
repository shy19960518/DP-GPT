import os
import glob
import re

import pandas as pd
import numpy as np
from Mahle_Model_parallel import plot, get_next_fuel_and_soc, from_soc_to_fuel
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle

import torch
from torch.utils.data import Dataset, DataLoader

from utils import EMSdataset, collate_fn

def clip_cost_shape(J):

    SOC_max = 0.310
    SOC_min = 0.250
    delta_t = 1        

    dsoc = 0.0001  

    soc_values = np.arange(SOC_min, SOC_max, dsoc)

    clip_max = 0.2830
    clip_min = 0.2720

    soc_max_index = np.where(np.isclose(soc_values, clip_max))[0][0]
    soc_min_index = np.where(np.isclose(soc_values, clip_min))[0][0]

    J = J[soc_min_index:soc_max_index, :]

    soc_values = soc_values[soc_min_index:soc_max_index]

    assert J.shape[0] == soc_values.shape[0]
    return J



pkl_files = glob.glob('./dataset/zerostart_zeroend_driving_cycle_list/*.pkl')
pkl_names = [os.path.basename(f) for f in pkl_files]
pkl_names.sort(key=lambda x: int(re.findall(r'\d+', x)[0]))




data_list = []
target_list = []
label_list = []
for pkl_name in pkl_names:

    with open(f'./dataset/zerostart_zeroend_driving_cycle_list/{pkl_name}', 'rb') as f:
        total_segments = pickle.load(f)

    name_without_ext = os.path.splitext(pkl_name)[0]
    for i in range(len(total_segments)):
        J_path = f"./dataset/zerostart_zeroend_driving_cycle_list_data/{name_without_ext}/J{i}.npy"
        if not os.path.exists(J_path):
            continue
        J = np.load(J_path)
        if J.size == 0:

            continue
        J = clip_cost_shape(J)
        driving_cycle = total_segments[i]

        driving_cycle = (driving_cycle - 0) / 39

        acceleration = np.diff(driving_cycle)  
        acceleration = np.insert(acceleration, 0, 0)

        T = len(driving_cycle)

        assert T == J.shape[1]
        data = []
        targets = []
        label = []
        for t in range(T - 1, -1, -1):
            v = driving_cycle[t]
            a = acceleration[t]
            target = J[:,t] # (110,)


            min_value = np.min(target)
            min_indices = np.where(target == min_value)[0]
            target_index = min_indices[-1]

            data.append((v,a))
            targets.append(target)
            label.append(target_index)


        J1 = np.load(J_path)

        data = np.array(data)  
        targets = np.array(targets)  
        label = np.array(label)

        assert data.shape == (T,2)
        assert targets.shape == (T, 110)
        data_list.append(data)
        target_list.append(targets)
        label_list.append(label)

dataset = EMSdataset(data_list, label_list)
torch.save(dataset, './dataset/ems_dataset_based_on_dp.pt')

