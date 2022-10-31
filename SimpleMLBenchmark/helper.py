import time
import os
from typing import List
import pandas as pd
import numpy as np
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


################## Logger Stuff ##################

class SingleTrackedParameter:
    def __init__(self) -> None:
        self.array = []
        self.last_check_time = 0 

    def start(self) -> None:
        self.last_check_time = time.process_time_ns()
    
    def stop(self) -> None:
        diff = time.process_time_ns() - self.last_check_time
        self.array.append(diff)

    def reset_tracker(self) -> None:
        self.array = []

    def invalidate_result(self) -> None:
        self.array.append(float('nan'))
    
    def get_average_ms(self) -> float:
        return np.mean(self.array) / 1_000_000
    

class Tracker:
    def __init__(self, device = 'cuda:0') -> None:
        self.deviceuse = device # device used for this test
        self.dsk_loadf = SingleTrackedParameter() # Load 1 File From Disk
        self.cpu_augmt = SingleTrackedParameter() # Augement the data and convert to tensor
        self.cpu_tslst = SingleTrackedParameter() # Convert a list of tensor into 1 stacked tensor
        self.cpu_mvgpu = SingleTrackedParameter() # Move data from CPU memory to GPU Memory
        self.gpu_fbexe = SingleTrackedParameter() # GPU Forward + Backward Pass + Gradient Update

    def simple_print(self):
        print("Test Type                    |  Speed [ms]")
        
        if "cuda" in str(self.deviceuse):
            print("CPU  - Image Augmentation    :", self.cpu_augmt.get_average_ms())
            print("CPU  - Tensor Stacking       :", self.cpu_tslst.get_average_ms())
            print("CPU  - Move data to GPU RAM  :", self.cpu_mvgpu.get_average_ms())
            print("DISK - Grab File From Disk   :", self.dsk_loadf.get_average_ms())
            print("GPU  - Overall GPU Execution :", self.gpu_fbexe.get_average_ms())
        else:
            print("CPU  - Image Augmentation    :", self.cpu_augmt.get_average_ms())
            print("CPU  - Tensor Stacking       :", self.cpu_tslst.get_average_ms())
            print("DISK - Grab File From Disk   :", self.dsk_loadf.get_average_ms())
            print("CPU  - Model CPU Execution   :", self.gpu_fbexe.get_average_ms())
        print()
        epoch_time = self.compute_score()            
        print("Average Per Epoch Execution  :", epoch_time)
        print()
        ryscore = (2000 - epoch_time) * 5
        print("Final RY-Score :", math.ceil(ryscore))
        print()
    
    def compute_score(self):
        c_aug = self.cpu_augmt.get_average_ms()
        c_tlt = self.cpu_tslst.get_average_ms()
        c_mvg = self.cpu_mvgpu.get_average_ms()
        d_lod = self.dsk_loadf.get_average_ms()
        g_exe = self.gpu_fbexe.get_average_ms()
        sum = c_aug + c_tlt + c_mvg + d_lod + g_exe
        return sum

################## Machine Learning Model Stuff ##################

class BasicModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            # 256
            nn.Conv2d(3, 16, kernel_size = 3, padding = 1, stride = 2),
            nn.LeakyReLU(0.1),
            #128
            nn.Conv2d(16, 32, kernel_size = 3, padding = 1, stride = 2),
            nn.LeakyReLU(0.1),
            #64
            nn.Conv2d(32, 128, kernel_size = 3, padding = 1, stride = 2),
            nn.LeakyReLU(0.1),
            #32
        )
        self.fin = nn.Sequential(
            nn.Linear(128 * 32 * 32, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 3),
            nn.Softmax(dim = 1)
        )
    
    def forward(self, x):
        batch_size = x.shape[0]
        x = self.conv(x)
        x = torch.reshape(x, (batch_size, -1))
        x = self.fin(x)
        return x


################## Dataset Management Stuff ##################

def GrabDataset() -> List:
    dataset = []
    ## Grab All Fire Images From Dataset
    for fnam in os.listdir("./dataset/Food-1"):
        full_fname = os.path.join('./dataset/Food-1', fnam)
        data_label = np.array([0, 0, 1])
        dataset.append( (full_fname, data_label) )

    ## Grab All Food Images From 
    for fnam in os.listdir("./dataset/Food-2"):
        full_fname = os.path.join('./dataset/Food-2', fnam)
        data_label = np.array([0, 1, 0])
        dataset.append( (full_fname, data_label) )
    
    ## Grab All Landscape Images
    for fnam in os.listdir("./dataset/Food-3"):
        full_fname = os.path.join('./dataset/Food-3', fnam)
        data_label = np.array([1, 0, 0])
        dataset.append( (full_fname, data_label) )

    print("Dataset Loading Complete !\n")
    return dataset

    
    
################## Single File Stuff ##################

if __name__ == '__main__':
    m = BasicModel()
    x = torch.rand(16, 3, 256, 256)
    y = m(x)
    print(y.shape)
    print(y)