import torch
import torch.nn as nn
from torchvision import datasets
from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt
import numpy as np
import random
import os

DATA_DIR=""
print(os.listdir())
if('MNIST data' in os.listdir()):
    DATA_DIR="./MNIST data"
    print("TRUE")
else:
    DATA_DIR = "."

download_dataset = False

train_mnist = datasets.MNIST(DATA_DIR, train=True, download=download_dataset)
test_mnist = datasets.MNIST(DATA_DIR, train=False, download=download_dataset)

print(train_mnist)