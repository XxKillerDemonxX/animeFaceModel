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
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
from scipy.signal import convolve



c = nn.ConvTranspose2d(100, 512, 4, 1, 0)


output = c(torch.randn(1, 100, 1, 1))
print(c.weight.shape)
print(c.bias.shape)
print(output.shape)


#legendary code
input_tensor = torch.randn(1, 100, 1, 1)
output_tensor = F.interpolate(input_tensor, size=(4, 4), mode='bilinear', align_corners=False)
output_tensor = output_tensor.permute(0, 2, 3, 1).reshape(-1, 100)
output_tensor = output_tensor@torch.randn(512, 100).T
output_tensor = output_tensor.reshape(1, 4, 4, 512).permute(0, 3, 1, 2)
print(output_tensor.shape)

print((torch.randn(1, 1, 1, 100)@torch.randn(100, 512*16)).shape)
print(torch.matmul(torch.randn(4, 9), torch.randn(16)).shape)
