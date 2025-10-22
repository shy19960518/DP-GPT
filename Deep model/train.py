from tqdm import tqdm
import os
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
dataset = torch.load('./dataset/ems_dataset_based_on_dp.pt')

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])


torch.save(train_dataset, './dataset/train_dataset.pth')
torch.save(val_dataset, './dataset/val_dataset.pth')

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DecoderOnlyTransformer(in_channels=2, out_channels=110, d_model=256, n_heads=4, num_layers=8, d_ff=512, max_len=200, dropout=0.1).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.0001)
criterion = nn.CrossEntropyLoss(ignore_index=-1)


num_epochs = 200
train_losses = []
val_losses = []

save_dir = './saved_model'
os.makedirs(save_dir, exist_ok=True)

for epoch in tqdm(range(num_epochs), desc="Epoch Training", ncols=100):
    model.train()
    epoch_train_loss = 0.0

    for batch_idx, (x, label, padding_mask) in enumerate(train_dataloader):
        x = x.to(device)
        label = label.to(device).long()
        padding_mask = padding_mask.to(device)

        out = model(x, padding_mask)  # (B, L, C)
        loss = criterion(out.view(-1, 110), label.view(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_train_loss += loss.item()

    avg_train_loss = epoch_train_loss / len(train_dataloader)
    train_losses.append(avg_train_loss)

    model.eval()
    epoch_val_loss = 0.0
    with torch.no_grad():
        for x_val, label_val, padding_mask_val in val_dataloader:
            x_val = x_val.to(device)
            label_val = label_val.to(device).long()
            padding_mask_val = padding_mask_val.to(device)

            out_val = model(x_val, padding_mask_val)
            val_loss = criterion(out_val.view(-1, 110), label_val.view(-1))
            epoch_val_loss += val_loss.item()

    avg_val_loss = epoch_val_loss / len(val_dataloader)
    val_losses.append(avg_val_loss)

    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

    if (epoch + 1) % 10 == 0:
        model_path = os.path.join(save_dir, f'model_epoch_{epoch+1}.pt')
        torch.save(model.state_dict(), model_path)

np.save(os.path.join(save_dir, 'train_losses.npy'), np.array(train_losses))
np.save(os.path.join(save_dir, 'val_losses.npy'), np.array(val_losses))


