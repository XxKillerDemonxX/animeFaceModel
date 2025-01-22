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
from scipy.signal import convolve


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
class ConvolutionalLayer(nn.Module):
    def __init__(self, in_channels, out_channels, filter_size, stride=1, padding=0, bias=True, padding_mode='zeros', device=None):
        super(ConvolutionalLayer, self).__init__()
        self.out_channels = out_channels
        self.in_channels = out_channels
        self.filter_size = filter_size
        self.weight = torch.randn(out_channels, in_channels, filter_size, filter_size)
        self.kernel = (filter_size, filter_size)
        if bias==True:
            self.bias = torch.randn(filter_size, filter_size)
        else:
            self.bias = torch.zeros_like(filter_size, filter_size)
    def __call__(self, x):
        
        #x should be in form of pixels x pixels x channels x kernel x kernel
        #would be easier to turn x into pixels x pixels x 27 (if channels is 3 and filter_size is 3) using .flatten and can shape it back for dot product later using .reshape
        #x = convolve(x, np.ones(self.kernel))@self.weight    <- this doesn't work sadly

        #unfold may solve all my problems
        #with a 3x9x9 .unfold with a 3x3 kernel -> 3x9x9x3x3 (with padding) 3 channels, 3x3 patch

        #lets assume i get it in batchsize x 64x64x27
        # i dont think this is correct -> x = x.reshape(batch_size, image_size, image_size, self.in_channels, self.filter_size, self.filter_size)


        self.out = x
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


