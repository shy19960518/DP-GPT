
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from utils import EMSdataset, collate_fn, set_random_seed
import numpy as np
import random
from typing import Optional
import matplotlib.pyplot as plt
from model import DecoderOnlyTransformer
from collections import Counter




set_random_seed(123)
val_dataset = torch.load('./dataset/val_dataset.pth')

val_dataloader = DataLoader(val_dataset, batch_size=128, shuffle=True, collate_fn=collate_fn)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DecoderOnlyTransformer(in_channels=2, out_channels=110, d_model=256, n_heads=4, num_layers=8, d_ff=512, max_len=200, dropout=0.1).to(device)

model.load_state_dict(torch.load('./saved_model (copy)/model_epoch_200.pt'))

model.eval()  


total_error = 0.0  
total_valid_count = 0.0  

with torch.no_grad():  
    for batch_idx, (x, target, padding_mask) in enumerate(val_dataloader):
        x = x.to(device)
        target = target.to(device)
        padding_mask = padding_mask.to(device)  # (B, L)

        
        out = model(x, padding_mask)  # out shape: (B, L, C)
        pred = out.argmax(dim=-1)     # shape: (B, L)

        
        error = (pred - target).abs()  # shape: (B, L)

        
        valid_error = error * padding_mask

        total_error += valid_error.sum()
        total_valid_count += padding_mask.sum()



mean_error = total_error / total_valid_count
print(f"Validation mean error = {mean_error.item():.4f}")

