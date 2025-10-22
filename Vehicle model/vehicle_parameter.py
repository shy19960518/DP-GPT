from types import SimpleNamespace
import math
import numpy as np
import pandas as pd

veh = SimpleNamespace(
    mass=1718.375,
    whl=337 / 1000,
    roll=[0.011, 0, 0],
    frmax=0.95,
    area=2.455,
    drag=0.32
)

ice = SimpleNamespace(
    dis=1.5,
    int=0.078,
    fuLHV=43.5,
    fuDens=743,
    idle=1000,
    MaxT= pd.read_csv('./vehicle_para/ice_MaxT.csv', header=None).to_numpy().squeeze(),
    TrSpd=pd.read_csv('./vehicle_para/ice_TrSpd.csv', header=None).to_numpy().squeeze(),
    fuSpd=pd.read_csv('./vehicle_para/ice_fuSpd.csv', header=None).to_numpy().squeeze(),
    fuTrq=pd.read_csv('./vehicle_para/ice_fuTrq.csv', header=None).to_numpy().squeeze(),
    fuMap=pd.read_csv('./vehicle_para/ice_fuMap.csv', header=None).to_numpy()
)


mg1 = SimpleNamespace(
    int=0, 
    MaxT=pd.read_csv('./vehicle_para/mg1_MaxT.csv', header=None).to_numpy().squeeze(),
    TrSpd=pd.read_csv('./vehicle_para/mg1_TrSpd.csv', header=None).to_numpy().squeeze(),
    effMap=pd.read_csv('./vehicle_para/mg1_effMap.csv', header=None).to_numpy(),
    effTrq = pd.read_csv('./vehicle_para/mg1_effTrq.csv', header=None).to_numpy().squeeze(),
    effSpd = pd.read_csv('./vehicle_para/mg1_effSpd.csv', header=None).to_numpy().squeeze(),
)


mg2 = SimpleNamespace(
    MaxT = pd.read_csv('./vehicle_para/mg2_MaxT.csv', header=None).to_numpy().squeeze(),
    TrSpd = pd.read_csv('./vehicle_para/mg2_TrSpd.csv', header=None).to_numpy().squeeze(),
    effMap = pd.read_csv('./vehicle_para/mg2_effMap.csv', header=None).to_numpy(),
    effTrq = pd.read_csv('./vehicle_para/mg2_effTrq.csv', header=None).to_numpy().squeeze(),
    effSpd = pd.read_csv('./vehicle_para/mg2_effSpd.csv', header=None).to_numpy().squeeze(),
)

batt = SimpleNamespace(
    ocv = 350, 
    cap = 54.3,
    ir = 0.15, 
    ce=0.99
)


# print(ice.MaxT)
# print(mg1.TrSpd)
# print(mg1.MaxT)