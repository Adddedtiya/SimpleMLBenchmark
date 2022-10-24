import torch
import numpy as np
from tqdm import tqdm

from time import sleep
from helper import Tracker, BasicModel

total_epochs = 128
batch_size   = 64
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
track = Tracker(device)

print("")
print("Simple Machine Learning Device Benchmark")
print("Batch size  :", batch_size)
print("Total epoch :", total_epochs)
print("Device used :", device)
print()

model = BasicModel().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr = 0.0001)
lossFunction = torch.nn.SmoothL1Loss()

print("Starting Logging !")
print()

for epoch_counter in range(total_epochs):
    print(" Epoch :", epoch_counter)
    model.train()
    for sub_batch in tqdm(range(0, 1000, batch_size)):
        sleep(0.1)
    print("")

print("Logging Complete!, Compiling Results...")
print()