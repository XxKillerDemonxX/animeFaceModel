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


#out, in, image, image
weight = torch.randn(1,3,4,4)
weight = weight.permute(1, 2, 3, 0).reshape(-1, 1)

#batch, in, image, image
x = torch.randn(4, 3, 9, 9)
image_height = ((x.shape[2] + (1 * 2) - 4) // 2) + 1
image_width = ((x.shape[3] + (1 * 2) - 4) // 2) + 1
x = torch.nn.functional.pad(x, (1, 1, 1, 1), mode = 'constant', value = 0)

unfold = nn.Unfold(kernel_size=(4, 4), stride = 2)


x = unfold(x)
x = x.transpose(1, 2)@weight
#out, patchsize, patches
print(x.shape)



x = x.transpose(1, 2)
x = x.reshape(4, 1, image_height, image_width)
print(x.shape)