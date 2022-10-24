import time
import pandas as pd
import numpy as np

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
    def __init__(self, device = 'cuda') -> None:
        self.deviceuse = device # device used for this test
        self.dsk_loadf = SingleTrackedParameter() # Load 1 File From Disk
        self.cpu_augmt = SingleTrackedParameter() # Augement the data and convert to tensor
        self.cpu_tslst = SingleTrackedParameter() # Convert a list of tensor into 1 stacked tensor
        self.cpu_mvgpu = SingleTrackedParameter() # Move data from CPU memory to GPU Memory
        self.gpu_fpexe = SingleTrackedParameter() # GPU Forward Pass
        self.gpu_bpexe = SingleTrackedParameter() # GPU Backward Pass + Gradient Update

    
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
            nn.Linear(128, 2),
            nn.Softmax(dim = 1)
        )
    
    def forward(self, x):
        batch_size = x.shape[0]
        x = self.conv(x)
        x = torch.reshape(x, (batch_size, -1))
        x = self.fin(x)
        return x


################## Device Checker Stuff ##################


################## Single File Stuff ##################

if __name__ == '__main__':
    m = BasicModel()
    x = torch.rand(16, 3, 256, 256)
    y = m(x)
    print(y.shape)
    print(y)