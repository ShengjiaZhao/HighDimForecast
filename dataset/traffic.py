import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms
import os, sys, shutil, copy, time
from torch.utils.data import Dataset, DataLoader


class TrafficDataset(Dataset):
    def __init__(self, train=True, max_len=32, y_dim=0):
        super(TrafficDataset, self).__init__()

        
        loader = np.load(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'new-york.npz')) 

        self.data_x = torch.from_numpy(loader['x']).type(torch.float32).permute(1, 0)
        self.data_y = torch.from_numpy(loader['travel_time']).type(torch.float32).permute(1, 0)
        
        self.indices = []
        
        len_scale = self.data_x.shape[0] // 10 
        len_scale_ = self.data_x.shape[0] // 11
        assert max_len <= len_scale // 2
        
        # Divide the data into train/test and avoid any overlap between them
        for bi in range(self.data_x.shape[0] // len_scale + 1):
            if train:
                lb = bi*len_scale
                rb = min(bi*len_scale+len_scale_, self.data_x.shape[0]) - max_len
            else:
                lb = bi*len_scale + len_scale_
                rb = min(bi*len_scale + len_scale, self.data_x.shape[0]) - max_len
            if lb < rb:
                self.indices.append(torch.arange(lb, rb))
        self.indices = torch.cat(self.indices)
        
        self.x_dim = self.data_x.shape[1]
        if y_dim <= 0:
            self.y_dim = self.data_y.shape[1]
        else:
            self.y_dim = y_dim
        self.max_len = max_len 

#         if train:
#             self.data_x = self.data_x[:-10000]
#             self.data_y = self.data_y[:-10000]
#         else:
#             self.data_x = self.data_x[-10000:]
#             self.data_y = self.data_y[-10000:]
            
    def __len__(self):
        return self.indices.shape[0]
    
    def __getitem__(self, idx):
        return self.data_x[self.indices[idx]:self.indices[idx]+self.max_len], self.data_y[self.indices[idx]:self.indices[idx]+self.max_len, :self.y_dim]
    
    