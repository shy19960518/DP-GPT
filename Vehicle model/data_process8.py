import os
import pandas as pd
import numpy as np
from Mahle_Model_parallel import plot, get_next_fuel_and_soc, from_soc_to_fuel
import matplotlib.pyplot as plt
from tqdm import tqdm
from deepdiff import DeepDiff
import glob
import pickle
from natsort import natsorted

data = np.load('./dataset/zerostart_zeroend_driving_cycle_list_data/VEDdata_7/J0.npy')

np.savetxt("./saved_data/J.csv", data, delimiter=",", fmt="%.4f")
