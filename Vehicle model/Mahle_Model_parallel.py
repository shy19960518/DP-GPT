
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from scipy import interpolate
from scipy.interpolate import RegularGridInterpolator

from vehicle_parameter import veh, ice, mg1, mg2, batt


def plot(driving_cycle):

    plt.plot(driving_cycle)

    # 添加标题和标签
    plt.title('Driving Cycle Data')
    plt.xlabel('Index')
    plt.ylabel('Value')

    # 显示图形
    plt.show()


def vehicle_dynamics(v,a):

    fa = (veh.drag * veh.area * v**2)/21.15
    fr = (veh.roll[0]+veh.roll[1]*v + veh.roll[2]*v**2)*veh.mass*9.8 if v != 0 else 0
    fv = veh.mass * a
    wheel_trq = veh.whl * (fa + fr + fv)

    wheel_speed = (v/3.6) / (veh.whl * 2 * math.pi) * 60

    return wheel_trq, wheel_speed

def final_drive(wheel_speed, wheel_trq):

    fd_trq = wheel_trq / 4.14
    fd_spd = wheel_speed * 4.14

    return fd_trq, fd_spd

def controller(eng_ratio, T_dem, mg1_spd):


    eng_ratio = np.clip(eng_ratio, 0, 1)
    get_max_torque_mg1 = interpolate.interp1d(mg1.TrSpd, mg1.MaxT, kind='linear', fill_value="extrapolate")
    get_max_torque_ice = interpolate.interp1d(ice.TrSpd, ice.MaxT, kind='linear', fill_value="extrapolate")
    gb_trq_max = 0.95 * min(get_max_torque_mg1(mg1_spd), get_max_torque_ice(mg1_spd))
    gb_trq = eng_ratio * gb_trq_max
    
    if mg1_spd > ice.idle:
        return gb_trq, T_dem - gb_trq
    else:
        return 0, T_dem

def mg2_gear(mg2_trq, fd_spd):

    mg2_gear = 3.91
    mg2_spd = fd_spd * 3.91
    mg2_trq = mg2_trq/3.91

    return mg2_trq, mg2_spd

def MG2(mg2_trq, mg2_spd):


    get_max_torque_mg2 = interpolate.interp1d(mg2.TrSpd, mg2.MaxT, kind='linear', fill_value="extrapolate")
    
    u1 = min(get_max_torque_mg2(mg2_spd), mg2_trq)

    mg2_map = RegularGridInterpolator((mg2.effTrq, mg2.effSpd), mg2.effMap, method='linear', bounds_error=False, fill_value=None)

    point1 = mg2_map((u1, mg2_spd))
    point2 = u1 * mg2_spd / 9.550

    if mg2_trq > 0:
        return point1 * point2
    else:
        return point2 / point1


def battery(discharge_power_w, batt_integration):

    point1 = discharge_power_w / batt.ocv * batt.ce
    if point1 > 0:
        batt_current = (1 + batt.ir) * point1
    else:
        batt_current = (1 - batt.ir) * point1

    batt_loss = (batt_current ** 2) * batt.ir

    batt_integration = batt_integration + batt_current

    # print(batt_current / (batt.cap*3600))

    batt_soc = 0.28 - batt_integration / (batt.cap*3600)

    return batt_loss, batt_soc, batt_integration

def mg1_gear(gb_trq, fd_spd):

    mg1_spd = fd_spd * 0.95
    mg1_trq = gb_trq / 0.95

    return mg1_trq, mg1_spd

def ICE(ice_trq, ice_spd, integration):
    
    ice_trq = np.clip(ice_trq, 0, None)

    fuel_map = RegularGridInterpolator((ice.fuTrq, ice.fuSpd), ice.fuMap, method='linear', bounds_error=False, fill_value=None)
    point1 = fuel_map((ice_trq, ice_spd))

    fuel_rate_Ls = point1 * 1e-3 / ice.fuDens * 1e3
    integration = integration + fuel_rate_Ls

    point2 = ice_trq * ice_spd / 9550 
    power_engine = (point1 * 1e-3 * 44000 - point2) * 1000

    return fuel_rate_Ls, integration, power_engine

######################################################################################################################################

def battery_next_soc(discharge_power_w, current_soc):
    point1 = discharge_power_w / batt.ocv * batt.ce
    if point1 > 0:
        batt_current = (1 + batt.ir) * point1
    else:
        batt_current = (1 - batt.ir) * point1

    batt_loss = (batt_current ** 2) * batt.ir

    # 计算下一时刻的SOC
    next_soc = current_soc - batt_current / (batt.cap * 3600)

    return batt_loss, next_soc


def get_next_fuel_and_soc(u, v, a, current_soc, fuel_integration=0):

    v = v * 3.6 # m/s to km/h
    wheel_trq, wheel_speed = vehicle_dynamics(v,a)
    T_dem, n = final_drive(wheel_speed, wheel_trq)

    mg1_spd = n * 0.95
    
    gb_trq, mg2_trq = controller(u, T_dem, mg1_spd)
    mg1_trq, mg1_spd = mg1_gear(gb_trq, n)
    mg2_trq, mg2_spd = mg2_gear(mg2_trq, n)


    fuel_rate_Ls, fuel_integration, power_engine = ICE(mg1_trq, mg1_spd, fuel_integration)


    discharge_power_w = MG2(mg2_trq, mg2_spd)

    batt_loss, next_soc, = battery_next_soc(discharge_power_w, current_soc)

    return fuel_rate_Ls, next_soc

def from_soc_to_fuel(d_soc):

    d_E = -d_soc * (batt.cap * 3600) * batt.ocv

    V_fuel = d_E / (ice.fuLHV * 1e6 * ice.fuDens)
    return V_fuel


def from_v_to_n1_n2(v): # m/s

    wheel_speed = (v) / (veh.whl * 2 * math.pi) * 60
    fd_spd = wheel_speed * 4.14
    mg1_spd = fd_spd * 0.95
    mg2_spd = fd_spd * 3.91

    return mg1_spd, mg2_spd

# print(from_v_to_n1_n2(50))



# df = pd.read_csv('./Driving_Cycles/BAC_Arterial_Cycle.csv')
# driving_cycle = df.iloc[:, 0].to_numpy()
# driving_cycle = np.insert(driving_cycle, 0, 0)

# acceleration = np.diff(driving_cycle)  
# acceleration = np.insert(acceleration, 0, 0)

# driving_cycle = driving_cycle[:65]
# acceleration = acceleration[:65]


# u_list = [1] * len(driving_cycle)

# test_list = []

# fuel_integration = 0
# batt_integration = 0

# for v,a,u in zip(driving_cycle, acceleration, u_list):

#     v = v * 3.6 # m/s to km/h
#     wheel_trq, wheel_speed = vehicle_dynamics(v,a)
#     T_dem, n = final_drive(wheel_speed, wheel_trq)


#     mg1_spd = n * 0.95

#     gb_trq, mg2_trq = controller(u, T_dem, mg1_spd)

#     mg1_trq, mg1_spd = mg1_gear(gb_trq, n)
#     mg2_trq, mg2_spd = mg2_gear(mg2_trq, n)


#     fuel_rate_Ls, fuel_integration, power_engine = ICE(mg1_trq, mg1_spd, fuel_integration)

    
#     discharge_power_w = MG2(mg2_trq, mg2_spd)

#     batt_loss, batt_soc, batt_integration = battery(discharge_power_w, batt_integration)
#     # print(batt_soc)

#     test_list.append(discharge_power_w)


################################################################################################33
# fuel_integration = 0
# current_soc = 0.28
# fuel_rate_Ls_list = []
# soc_list = []
# for v,a,u in zip(driving_cycle, acceleration, u_list):

#     fuel_rate_Ls, next_soc = get_next_fuel_and_soc(u, v, a, current_soc, fuel_integration)

#     current_soc = next_soc
#     fuel_rate_Ls_list.append(fuel_rate_Ls)
#     soc_list.append(next_soc)



# # plt.plot(fuel_rate_Ls_list)
# plt.plot(test_list)
# # 添加标题和标签
# plt.title('Driving Cycle Data')
# plt.xlabel('Index')
# plt.ylabel('Value')

# # 显示图形
# plt.show()
#########################################################################################################



# def normalize(arr):
#     max_val = np.max(arr)  # 获取最大值
#     if max_val == 0:  # 避免除以0
#         return arr
#     return arr / max_val  # 归一化


# test_list = np.array(test_list, dtype=np.float32)

# # driving_cycle = normalize(driving_cycle)
# # test_list = normalize(test_list)

# # plt.plot(driving_cycle)
# plt.plot(test_list)
# # 添加标题和标签
# plt.title('Driving Cycle Data')
# plt.xlabel('Index')
# plt.ylabel('Value')

# # 显示图形
# plt.show()
