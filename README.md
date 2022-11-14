# SimpleMLBenchmark

For normal users use the "Normal Install" option, for people who want to test DirectML use the "DirectML Install"

### Please read the Mode of executions on the end of the readme

## Normal Install

1. Install Python

2. Clone this repo

3. Install prerequisites 
```
    pip install --upgrade numpy tqdm Pillow  
```

4. Install [Pytroch](https://pytorch.org/get-started/locally/)

5. Run the scoring system 

```
    cd SimpleMLBenchmark
    python main.py
```

## DirectML Install
Follow the steps below to get set up with PyTorch on DirectML. 

[WSL](https://learn.microsoft.com/en-us/windows/ai/directml/gpu-pytorch-wsl) 
[WIN](https://learn.microsoft.com/en-us/windows/ai/directml/gpu-pytorch-windows)

1. Download and install [Python 3.8](https://www.python.org/downloads/release/python-380/).

2. Install [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)

2. Clone this repo.	

3. Create conda env
```
    conda create --name pydml -y
    conda activate pydml
```

3. Install prerequisites
```
    pip install torchvision==0.9.0
    pip uninstall torch
    pip install pytorch-directml
```

> Note: The torchvision package automatically installs the torch==1.8.0 dependency, but this is not needed and will cause collisions with the pytorch-directml package. We must uninstall the torch package after installing requirements.

4. Install other libraries 

```
    pip install --upgrade numpy tqdm Pillow  
```

5. Run the scoring system 

```
    cd SimpleMLBenchmark
    python main.py
```

## Tests Modes

### Autoselect

```
    python main.py 
```

### Force cuda / rocm

```
    python main.py --cuda 
```

### Force DirectML

```
    python main.py --dml 
```

### Force Appple AI Accelerator

```
    python main.py --apple
```