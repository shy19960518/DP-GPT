import os
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
    print(len(soc_values))
    u_values = np.arange(0, 1 + 0.2, 0.2)

with open('./dataset/total_segments.pkl', 'rb') as f:
    total_segments = pickle.load(f)

i = 30
driving_cycle = total_segments[i]

J = np.load(f"./dataset/saved_data_total_segments/J{i}.npy")
policy = np.load(f"./dataset/saved_data_total_segments/policy{i}.npy")

np.savetxt("./J.csv", J, delimiter=",", fmt="%.4f")


SOC_max = 0.310
SOC_min = 0.250
delta_t = 1        

T = len(driving_cycle)
dsoc = 0.00005  

soc_values = np.arange(SOC_min, SOC_max, dsoc)
u_values = np.arange(0, 1 + 0.2, 0.2)
u_values = [1]



#####################################test###############################################
soc_list = []
control_list = []
fuel_cost_total = 0

current_soc = 0.28
acceleration = np.diff(driving_cycle) 
acceleration = np.insert(acceleration, 0, 0)
for t, (v, a) in enumerate(zip(driving_cycle, acceleration)):
    soc_list.append(current_soc)
    current_soc_index = np.argmin(np.abs(soc_values - current_soc))

    total_cost_list = []

    predicted_soc_cost_map = J[:, t]

    next_soc_list = []

    for u in u_values:
        fuel_cost, next_soc = get_next_fuel_and_soc(u, v, a, current_soc)
        next_soc_list.append(next_soc)
        next_soc_index = np.argmin(np.abs(soc_values - next_soc))
        total_cost = 1000 * fuel_cost + predicted_soc_cost_map[next_soc_index]
        total_cost_list.append(total_cost)

        min_cost = np.min(total_cost_list)
        min_indices = np.where(total_cost_list == min_cost)[0]
        ran_indice = np.random.choice(min_indices)
        max_indice = np.max(min_indices)
        min_indice = np.min(min_indices)
        u_opt = u_values[ran_indice]

        next_soc = next_soc_list[ran_indice]

    fuel_cost_total = fuel_cost_total + fuel_cost
    control_list.append(u_opt)

    current_soc = next_soc


print(soc_list[-1])
print(fuel_cost_total)

plt.figure(figsize=(10, 12))  

plt.subplot(3, 1, 1)
plt.plot(soc_list)
plt.title('soc_list')
plt.xlabel('Time (s)')
plt.ylabel('soc')

plt.subplot(3, 1, 2)
plt.plot(control_list)
plt.title('control_list')
plt.xlabel('Time (s)')
plt.ylabel('u')

plt.subplot(3, 1, 3)
plt.plot(driving_cycle)
plt.title('driving_cycle')
plt.xlabel('Time (s)')
plt.ylabel('Speed')

plt.tight_layout()  
plt.show()


