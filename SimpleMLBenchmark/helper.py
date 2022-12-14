import time
import os
from typing import List
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
    def __init__(self, device = 'cuda') -> None:
        self.deviceuse = device # device used for this test
        self.dsk_loadf = SingleTrackedParameter() # Load 1 File From Disk
        self.cpu_augmt = SingleTrackedParameter() # Augement the data and convert to tensor
        self.cpu_tslst = SingleTrackedParameter() # Convert a list of tensor into 1 stacked tensor
        self.cpu_mvgpu = SingleTrackedParameter() # Move data from CPU memory to GPU Memory
        self.gpu_fbexe = SingleTrackedParameter() # GPU Forward + Backward Pass + Gradient Update

    def simple_print(self):
        print("Test Type                    |  Speed [ms]")
        if "cpu" in str(self.deviceuse):
            print("CPU  - Image Augmentation    :", self.cpu_augmt.get_average_ms())
            print("CPU  - Tensor Stacking       :", self.cpu_tslst.get_average_ms())
            print("CPU  - Model CPU Execution   :", self.gpu_fbexe.get_average_ms())
            print("DISK - Grab File From Disk   :", self.dsk_loadf.get_average_ms())
        else:
            print("CPU  - Image Augmentation    :", self.cpu_augmt.get_average_ms())
            print("CPU  - Tensor Stacking       :", self.cpu_tslst.get_average_ms())
            print("CPU  - Move data to GPU RAM  :", self.cpu_mvgpu.get_average_ms())
            print("DISK - Grab File From Disk   :", self.dsk_loadf.get_average_ms())
            print("GPU  - Overall GPU Execution :", self.gpu_fbexe.get_average_ms())
        epoch_time = self.compute_score()            
        print()
        ryscore = (1_000 / epoch_time) * 100
        print("Average Per Epoch Execution  :", epoch_time)
        print("Final RY-Score               :", math.ceil(ryscore))
        print("Pytorch Device               :", self.deviceuse)
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
            nn.ReLU()
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


################## Check and Request Device ##################    

def check_device(device_str):
    x = torch.tensor([1])
    try:
        x = x.to(device_str)
    except:
        return False
    del x
    return True

def grab_torch_device(args) -> str:
    print()
    print("Checking Avaliable Devices...")

    avcuda = check_device('cuda')
    avdml  = check_device('dml')
    avappl = check_device('mps')

    print("CUDA      :", avcuda)
    print("DirectML  :", avdml)
    print("Apple Mx  :", avappl)
    print()
    rqcuda   = bool(args.cuda)
    rqdml    = bool(args.dml)
    rqappple = bool(args.apple)
    rqcpu    = bool(args.cpu)

    # CPU have priority in Request
    if rqcpu:
        return "cpu"

    if rqcuda:
        if avcuda:
            return "cuda"
        else:
            raise Exception("cuda is requested but no cuda device exist !")

    if rqdml:
        if avdml:
            return "dml"
        else:
            raise Exception("DirectML is not found !")
    
    if rqappple:
        if avappl:
            return "mps"
        else:
            raise Exception("This option is for Apple Mx Devices Only")

    # Automatic Best Device allocation
    if avcuda:
        return "cuda"
    if avdml:
        return "dml"
    if avappl:
        return "mps"
    return "cpu"

################## Single File Stuff ##################

if __name__ == '__main__':
    m = BasicModel()
    x = torch.rand(16, 3, 256, 256)
    y = m(x)
    print(y.shape)
    print(y)