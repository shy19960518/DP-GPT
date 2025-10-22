import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import random
class EMSdataset(Dataset):
    def __init__(self, data_list, target_list):
        """
        data_list: List of (L, 2) numpy arrays
        target_list: List of (L, 302) numpy arrays
        """
        self.data_list = [torch.tensor(data, dtype=torch.float32) for data in data_list]
        self.target_list = [torch.tensor(target, dtype=torch.float32) for target in target_list]

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx], self.target_list[idx]

def collate_fn(batch):
    PAD_VALUE = -1  

    data, target = zip(*batch)  

    data = [d.clone().detach().float() if isinstance(d, torch.Tensor) else torch.tensor(d, dtype=torch.float32) for d in data]
    target = [t.clone().detach().float() if isinstance(t, torch.Tensor) else torch.tensor(t, dtype=torch.float32) for t in target]

    max_len = max(d.shape[0] for d in data)

    data_padded = pad_sequence(data, batch_first=True, padding_value=PAD_VALUE)
    target_padded = pad_sequence(target, batch_first=True, padding_value=PAD_VALUE)

    mask = (data_padded[:, :, 0] != PAD_VALUE).float()  # (batch_size, max_len)

    return data_padded, target_padded, mask

def limit_consecutive_zeros(arr, max_zeros=20):

    arr = np.array(arr)  
    mask = (arr == 0)  
    

    zero_groups = np.split(np.where(mask)[0], np.where(np.diff(np.where(mask)[0]) != 1)[0] + 1)
    
    indices_to_remove = []  
    for group in zero_groups:
        if len(group) > max_zeros:
            indices_to_remove.extend(group[max_zeros:])  
    

    arr = np.delete(arr, indices_to_remove)
    return arr

def find_zero_indices(driving_cycle: np.ndarray):

    return np.where(driving_cycle == 0)[0]

def segment_indices(driving_cycle: np.ndarray, max_length: int = 200):

    zero_indices = find_zero_indices(driving_cycle)

    segments = []
    i = 0  

    while i < len(zero_indices) - 1:
        start_idx = zero_indices[i]

        j = i + 1
        while j < len(zero_indices) and zero_indices[j] - start_idx <= max_length:
            j += 1

        if j > i + 1:
            end_idx = zero_indices[j - 1]
            i = j - 1  
        else:
            end_idx = zero_indices[i + 1]
            i = i + 1

        segments.append((start_idx, end_idx))

    return segments

def get_segments_from_driving_cycle(driving_cycle):

    segment_indexs = segment_indices(driving_cycle)

    segments = []
    for segment_index in segment_indexs:
        segment = driving_cycle[segment_index[0]:segment_index[1]]
        segments.append(segment)
    return segments
    
    
def set_random_seed(seed):
    random.seed(seed)
    
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  
    
    
