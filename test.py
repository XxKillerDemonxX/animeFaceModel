import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
from scipy.signal import convolve


weight = torch.randn(4,3,3,3)
weight = weight.permute(1, 2, 3, 0).view(-1, 4)

x = torch.randn(4, 3, 9, 9)
x = torch.nn.functional.pad(x, (1, 1, 1, 1), mode = 'constant', value = 0)

unfold = nn.Unfold(kernel_size=(3, 3))


x = unfold(x)
x = x.transpose(1, 2)@weight
print(x.shape)

x = x.transpose(1, 2)
x = x.reshape(4, 9, 9, 4)
print(x.shape)