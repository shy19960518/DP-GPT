import numpy as np
import pandas as pd
from tqdm import tqdm
from Mahle_Model_parallel import plot, get_next_fuel_and_soc, from_soc_to_fuel
import matplotlib.pyplot as plt
from utils import get_segments_from_driving_cycle
import pickle



np.set_printoptions(precision=5)

df = pd.read_csv('./Driving_Cycles/FTP75.csv')
driving_cycle = df.iloc[:, 0].to_numpy()

segments = get_segments_from_driving_cycle(driving_cycle)

driving_cycle = segments[-2]

################################################################################################

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


for t in tqdm(range(T - 2, -1, -1), desc="Time Step", ncols=100, leave=True, dynamic_ncols=True):
    acceleration = np.diff(driving_cycle)  
    acceleration = np.insert(acceleration, 0, 0)
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
        # if (min_cost == np.inf) and J[idx,t] != np.inf:
        if (min_cost == np.inf):
            break
    
    last_min_cost = np.inf
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


np.save("./saved_data/J.npy", J)
np.save("./saved_data/policy.npy", policy)
np.savetxt("./saved_data/J.csv", J, delimiter=",", fmt="%.4f")

################################## test ###########################################

J = np.load("./saved_data/J.npy")
SOC_max = 0.310
SOC_min = 0.250
delta_t = 1        

T = len(driving_cycle)
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


