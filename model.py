#%matplotlib inline  <- for junypter notebooks, which we are not using
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


manual_seed = 999

dataroot = "Input"
workers = 2
batch_size = 128    #batch size that will be used in training
image_size = 64     #spatial size of images
nc = 3              #number of color channels
nz = 100            #length of latent vector
ngf = 64            #depth of feature maps in generator
ndf = 64            #length of feature maps in discriminator
num_epochs = 50     #number of training epochs
lr = 0.0002         #learning rate
beta1 = 0.5         #beta1 hyperparameter
ngpu = 1            #number of gpus available


dataset = dset.ImageFolder(root = dataroot,
                           transform=transforms.Compose([transforms.Resize(image_size),
                                                            transforms.CenterCrop(image_size),
                                                            transforms.ToTensor(),
                                                            transforms.Normalize((0.5, 0.5,0.5), (0.5, 0.5, 0.5)),
                                                        ]))

dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=workers)

#what device to run on (gpu or cpu)
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")




#convolutional layer
#out_channels can also represent how many filters (kernels) are in the layer
#if 64x64 rgb image (3 channels), dimensions would be 64,64,3. in_channels would be 3
#filter_size is always smaller than image dimensions
class ConvolutionalLayer:
    def __init__(self, in_channels, out_channels, filter_size, stride=1, padding=0, bias=True, padding_mode='zeros', device=None):
        self.out_channels = out_channels
        self.weight = torch.randn(out_channels, in_channels, filter_size, filter_size)

    def __call__(self, x):
        #need some type of sliding operation here...
        #turn the image into patches (ex. a 9x9 matrix should have 9 patches of filter_size x filter_size)
        self.out = x@self.weight
    def parameters(self):
        return []









# for data in dataset:
#     print(data)
if __name__ == '__main__':
    real_batch = next(iter(dataloader))
    plt.figure(figsize=(8,8))
    plt.axis("off")
    plt.title("Training Images")
    plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device), padding=2, normalize=True).cpu(), (1, 2, 0)))
    plt.show()


