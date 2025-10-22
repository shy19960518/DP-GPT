import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
import numpy as np
from utils import EMSdataset, collate_fn, get_segments_from_driving_cycle
from Mahle_Model_parallel import plot, get_next_fuel_and_soc, from_soc_to_fuel


from typing import Optional
import matplotlib.pyplot as plt
from model import DecoderOnlyTransformer


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DecoderOnlyTransformer(in_channels=2, out_channels=110, d_model=256, n_heads=4, num_layers=8, d_ff=512, max_len=200, dropout=0.1).to(device)
model.load_state_dict(torch.load('../Deep model/example/model_epoch_200.pt'))
model.eval()  

df1 = pd.read_csv('./Driving_Cycles/WLTP_CLASS_3.csv')
driving_cycle1 = df1.iloc[:, 0].to_numpy()

df2 = pd.read_csv('./Driving_Cycles/FTP75.csv')
driving_cycle2 = df2.iloc[:, 0].to_numpy()


driving_cycle_total = np.concatenate((driving_cycle1, driving_cycle2))

segments = get_segments_from_driving_cycle(driving_cycle_total)

######################################segment <= 200##########################################
MAX_LEN = 200
final_segments = []
for seg in segments:
    seg = np.array(seg)  

    if len(seg) <= MAX_LEN:
        final_segments.append(seg)
    else:
        # 分段处理
        num_splits = (len(seg) + MAX_LEN - 1) // MAX_LEN  
        for i in range(num_splits):
            start = i * MAX_LEN
            end = min((i + 1) * MAX_LEN, len(seg))
            chunk = seg[start:end]
            final_segments.append(chunk)
##########################################################################################

SOC_max = 0.310
SOC_min = 0.250
delta_t = 1        
dsoc = 0.0001

soc_values = np.arange(SOC_min, SOC_max, dsoc)

clip_max = 0.2830
clip_min = 0.2720

soc_max_index = np.where(np.isclose(soc_values, clip_max))[0][0]
soc_min_index = np.where(np.isclose(soc_values, clip_min))[0][0]

soc_values = soc_values[soc_min_index:soc_max_index]
idx = np.where(np.isclose(soc_values, 0.28))[0][0]

u_values = np.arange(0, 1 + 0.1, 0.1)

###########################################################################################
current_soc = 0.28
soc_list = []
fuel_cost_total = 0
fuel_cost_total_list = []

control_list = []
travel_distance = 0
travel_distance_list = []

for driving_cycle in final_segments:
    
    acceleration = np.diff(driving_cycle)  
    acceleration = np.insert(acceleration, 0, 0)

    input_driving_cycle = driving_cycle / 39
    input_acceleration = acceleration / 39
    input_data = np.column_stack((input_driving_cycle, input_acceleration))
    input_data = input_data[::-1].copy() 
    input_data = torch.from_numpy(input_data).float().unsqueeze(0).to(device)

    with torch.no_grad(): 
        output = model(input_data)  
        _, predicted_labels = torch.max(output, dim=-1)  

    for t, (v, a) in enumerate(zip(driving_cycle, acceleration)):


        soc_list.append(current_soc)
        current_soc_index = np.argmin(np.abs(soc_values - current_soc))

        if len(input_driving_cycle) >= 2:
            best_next_soc_index = predicted_labels.squeeze()[-(1+t)].item()
        else:
            best_next_soc_index = 80
        
        next_soc_index_list = []
        next_soc_list = []
        fuel_cost_list = []
        for u in u_values:
            fuel_cost, next_soc = get_next_fuel_and_soc(u, v, a, current_soc)
            next_soc_list.append(next_soc)
            next_soc_index = np.argmin(np.abs(soc_values - next_soc))
            next_soc_index_list.append(next_soc_index)

            fuel_cost_list.append(fuel_cost)

 
        abs_diff = np.abs(np.array(next_soc_index_list) - best_next_soc_index)


        min_diff = np.min(abs_diff)


        min_indices = np.where(abs_diff == min_diff)[0]


        if current_soc < 0.28:
            opt_index = min_indices[-1]
        else:
            opt_index = min_indices[0]
        

        u_opt = u_values[opt_index]
        next_soc = next_soc_list[opt_index]


        fuel_cost_total = fuel_cost_total + fuel_cost_list[opt_index]

        fuel_cost_total_list.append(fuel_cost_total)
        control_list.append(u_opt)
        current_soc = next_soc



    ###############################Calculate the driving distance to convert fuel consumption into liters per 100 kilometers################################
    
        travel_distance += v
        travel_distance_list.append(travel_distance)


plt.figure(figsize=(10, 12))  


print(soc_list[-1])
print(fuel_cost_total_list[-1])
print(travel_distance)
baigongliyouhao = fuel_cost_total_list[-1] * 100 * 1000/ travel_distance
print(baigongliyouhao)
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
plt.plot(driving_cycle_total)
plt.title('driving_cycle')
plt.xlabel('Time (s)')
plt.ylabel('Speed')

plt.tight_layout()  
plt.savefig('results_plot.png', dpi=300)  
# plt.show()


