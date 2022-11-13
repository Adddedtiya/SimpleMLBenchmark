# SimpleMLBenchmark

For normal users use the "Normal Install" option, for people who want to test DirectML use the "DirectML Install"

## Normal Install

1. Install Python

2. Clone this repo

3. Install prerequisites 
```
    pip install numpy tqdm Pillow  
```

4. Install [Pytroch](https://pytorch.org/get-started/locally/)

5. Run the scoring system 

```
    cd SimpleMLBenchmark
    python main.py
```

## DirectML Install
Follow the steps below to get set up with PyTorch on DirectML.

1.	Download and install [Python 3.8](https://www.python.org/downloads/release/python-380/).

2. Clone this repo.	

3. Install prerequisites
```
    pip install torchvision==0.9.0
    pip uninstall torch
    pip install pytorch-directml
```

> Note: The torchvision package automatically installs the torch==1.8.0 dependency, but this is not needed and will cause collisions with the pytorch-directml package. We must uninstall the torch package after installing requirements.

4. Install other libraries 

```
    pip install numpy tqdm Pillow  
```

5. Run the scoring system 

```
    cd SimpleMLBenchmark
    python main.py
```
