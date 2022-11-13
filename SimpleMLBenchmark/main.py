import torch
import random
from tqdm import tqdm
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TTFF
import argparse

from helper import Tracker, BasicModel, GrabDataset, grab_torch_device

# Create the parser
parser = argparse.ArgumentParser()

# Add an argument
parser.add_argument('--cuda', action='store_true')
parser.add_argument('--dml',  action='store_true')
parser.add_argument('--apple',action='store_true')
parser.add_argument('--cpu',  action='store_true')

# Parse the argument
args = parser.parse_args()

print("")
print("Simple Machine Learning Device Benchmark")

total_epochs = 128
batch_size   = 64
device = grab_torch_device(args)
track = Tracker(device)

print("Batch size  :", batch_size)
print("Total epoch :", total_epochs)
print("Device used :", device)
print()

model = BasicModel().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr = 0.0001)
lossFunction = torch.nn.L1Loss()
dataset = GrabDataset()

print("Starting Logging !")
print()

for epoch_counter in range(total_epochs):
    print(" Epoch :", epoch_counter)
    model.train()

    for batch_counter in tqdm(range(0, len(dataset), batch_size)):
        batch_dataset = dataset[batch_counter : batch_counter + 4]

        list_x = []
        list_y = []
        lss = []
        for imgFilePath, labelOneHot in batch_dataset:

            track.dsk_loadf.start()
            image = Image.open(imgFilePath)
            track.dsk_loadf.stop()

            track.cpu_augmt.start()
            image = image.resize((256, 256))
            x = TTFF.to_tensor(image)
            x = x / 255
            rotate_val = random.randint(-180, 180)
            x = TTFF.rotate(x, rotate_val)
            y = torch.from_numpy(labelOneHot)

            x = x.to(torch.float)
            y = y.to(torch.float)

            list_x.append(x)
            list_y.append(y)
            track.cpu_augmt.stop()

        track.cpu_tslst.start()
        x = torch.stack(list_x, 0)
        y = torch.stack(list_y, 0)
        track.cpu_tslst.stop()

        track.cpu_mvgpu.start()
        image = x.to(device)
        label = y.to(device)
        track.cpu_mvgpu.stop()

        optimizer.zero_grad()

        track.gpu_fbexe.start()
        output = model(image)
        loss = lossFunction(output, label)
        loss.backward()
        lss.append(loss.item())
        optimizer.step()
        track.gpu_fbexe.stop()
    print("loss :", np.average(lss))
    print("")

print("Logging Complete!, Compiling Results...")
print()
track.simple_print()