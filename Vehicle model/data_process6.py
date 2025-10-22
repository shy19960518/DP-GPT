import os
import pandas as pd
import numpy as np
from Mahle_Model_parallel import plot, get_next_fuel_and_soc, from_soc_to_fuel
import matplotlib.pyplot as plt
from tqdm import tqdm

import glob
import pickle
from natsort import natsorted

def dp_process(driving_cycle, driving_cycle_number):

    acceleration = np.diff(driving_cycle)
    acceleration = np.insert(acceleration, 0, 0)


    SOC_set = 0.28  


    SOC_max = 0.310
    SOC_min = 0.250
    delta_t = 1        

    T = len(driving_cycle)
    dsoc = 0.0001 


    soc_values = np.arange(SOC_min, SOC_max, dsoc)  
    n_steps = len(soc_values)

    idx = np.where(np.isclose(soc_values, 0.28))[0][0]

    soc_lower_part, soc_higher_part = soc_values[:idx], soc_values[idx:] 

    u_values = np.arange(0, 1 + 0.2, 0.2)

    ############################################################################
    J = np.inf * np.ones((n_steps, T))
    J[idx, -1] = 0

    policy = np.zeros((n_steps, T))

    for t in tqdm(range(T - 2, -1, -1), desc=f"{driving_cycle_number} on process", ncols=100, leave=True, dynamic_ncols=True):
    # for t in range(T - 2, -1, -1):
        for i, soc in enumerate(soc_higher_part):
            i = i + idx
            u_cost_list = []
            for u in u_values:
                v = driving_cycle[t]
                a = acceleration[t]
                fuel_cost, next_soc = get_next_fuel_and_soc(u, v, a, soc)
                next_soc_index = np.argmin(np.abs(soc_values - next_soc))
                total_cost = 1000 * fuel_cost + J[next_soc_index, t + 1]
                u_cost_list.append(total_cost)

            min_cost = np.min(u_cost_list)

            J[i, t] = min_cost

            min_indices = np.where(u_cost_list == min_cost)[0]
            min_indice = np.random.choice(min_indices)
            policy[i, t] = u_values[min_indice]
            if (min_cost == np.inf) and J[idx,t] != np.inf:
                break
        for i, soc in enumerate(reversed(soc_lower_part)):

            i = idx - 1 - i
            
            u_cost_list = []
            for u in u_values:
                v = driving_cycle[t]
                a = acceleration[t]
                fuel_cost, next_soc = get_next_fuel_and_soc(u, v, a, soc)
                next_soc_index = np.argmin(np.abs(soc_values - next_soc))
                total_cost = 1000 * fuel_cost + J[next_soc_index, t + 1]
                u_cost_list.append(total_cost)

            min_cost = np.min(u_cost_list)
            J[i, t] = min_cost

            min_indices = np.where(u_cost_list == min_cost)[0]
            min_indice = np.random.choice(min_indices)
            policy[i, t] = u_values[min_indice]
            
            if (min_cost == np.inf) and J[idx-1,t] != np.inf:
                break


        np.save(os.path.join(target_dir, f"J{driving_cycle_number}.npy"), J)




source_pattern = "./dataset/buffer/VEDdata_*.pkl"  

target_root = "./dataset/zerostart_zeroend_driving_cycle_list_data"


os.makedirs(target_root, exist_ok=True)

for file_path in natsorted(glob.glob(source_pattern)):
    file_name = os.path.basename(file_path)  

    
    target_dir = os.path.join(target_root, file_name.replace(".pkl", ""))
    os.makedirs(target_dir, exist_ok=True)  

    with open(f'./dataset/buffer/{file_name}', 'rb') as f:
        zerostart_zeroend_driving_cycle_list = pickle.load(f)

    for driving_cycle_number, driving_cycle in tqdm(enumerate(zerostart_zeroend_driving_cycle_list), total=len(zerostart_zeroend_driving_cycle_list), desc=f"{file_name}", ncols=100, leave=True, dynamic_ncols=True):

        dp_process(driving_cycle, driving_cycle_number)
