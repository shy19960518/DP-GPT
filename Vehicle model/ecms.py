import pandas as pd
import numpy as np
from tqdm import tqdm

from utils import get_segments_from_driving_cycle
from Mahle_Model_parallel import plot, get_next_fuel_and_soc, from_soc_to_fuel
import matplotlib.pyplot as plt




df1 = pd.read_csv('../2_vehicle model/dataset/testset/WLTP_CLASS_3.csv')
driving_cycle1 = df1.iloc[:, 0].to_numpy()

df2 = pd.read_csv('../2_vehicle model/dataset/testset/FTP75.csv')
driving_cycle2 = df2.iloc[:, 0].to_numpy()


driving_cycle_total = np.concatenate((driving_cycle1, driving_cycle2))
acceleration = np.diff(driving_cycle_total) 
acceleration = np.insert(acceleration, 0, 0)

current_soc = 0.28

SOC_max = 0.310
SOC_min = 0.250
delta_t = 1        

T = len(driving_cycle_total)

soc_list = []
fuel_cost_total = 0
fuel_cost_total_list = []

control_list = []
travel_distance = 0
travel_distance_list = []

lambda_eq = 3.5


for v,a in zip(driving_cycle_total, acceleration):

    soc_list.append(current_soc)

    u_list = np.arange(0, 1 + 0.1, 0.1)
    best_J = float('inf')
    best_u = 0.0

    for u in u_list:
        fuel_cost, next_soc = get_next_fuel_and_soc(u, v, a, current_soc)
        delta_soc = current_soc - next_soc  
        electric_cost = lambda_eq * delta_soc
        J = fuel_cost + electric_cost

        if J < best_J:
            best_J = J
            best_u = u
            best_next_soc = next_soc
            best_fuel_cost = fuel_cost


    u = best_u

    fuel_cost, next_soc = get_next_fuel_and_soc(u, v, a, current_soc)
    control_list.append(u)


    fuel_cost_total = fuel_cost_total + fuel_cost
    fuel_cost_total_list.append(fuel_cost_total)


    current_soc = next_soc

    travel_distance += v
    travel_distance_list.append(travel_distance)

# # 保存成 .npy 文件
# np.save('../3_Experiment/B/ECMS/soc.npy', np.array(soc_list))
# np.save('../3_Experiment/B/ECMS/control.npy', np.array(control_list))
# np.save('../3_Experiment/B/ECMS/fuel_cost_total.npy', np.array(fuel_cost_total_list))
# np.save('../3_Experiment/B/ECMS/travel_distance.npy', np.array(travel_distance_list))
# np.save('../3_Experiment/B/ECMS/driving_cycle_total.npy', driving_cycle_total)

# plt.figure(figsize=(10, 12))  # 调整 figsize 以适应三行图


# print(soc_list[-1])
# print(fuel_cost_total_list[-1])
# print(travel_distance)
# baigongliyouhao = fuel_cost_total_list[-1] * 100 * 1000/ travel_distance
# print(baigongliyouhao)
# plt.figure(figsize=(10, 12))  # 调整 figsize 以适应三行图

# plt.subplot(3, 1, 1)
# plt.plot(soc_list)
# plt.title('soc_list')
# plt.xlabel('Time (s)')
# plt.ylabel('soc')

# plt.subplot(3, 1, 2)
# plt.plot(control_list)
# plt.title('control_list')
# plt.xlabel('Time (s)')
# plt.ylabel('u')

# plt.subplot(3, 1, 3)
# plt.plot(driving_cycle_total)
# plt.title('driving_cycle')
# plt.xlabel('Time (s)')
# plt.ylabel('Speed')

# plt.tight_layout()  # 自动调整子图间距
# plt.savefig('results_plot.png', dpi=300)  # 保存图片，dpi可调
# # plt.show()


