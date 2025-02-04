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
import torch.nn.functional as F
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

#dataset
dataset = dset.ImageFolder(root = dataroot,
                           transform=transforms.Compose([transforms.Resize(image_size),
                                                            transforms.CenterCrop(image_size),
                                                            transforms.ToTensor(),
                                                            transforms.Normalize((0.5, 0.5,0.5), (0.5, 0.5, 0.5)),
                                                        ]))
#loading the dataset
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=workers)

#what device to run on (gpu or cpu)
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
print(torch.cuda.is_available())
#print(torch.__version__)


#function to add random noise for an input (supposedly it helps the model train)
def add_noise(images, noise_std=0.1):
    return images + noise_std * torch.randn_like(images)


#convolutional layer
class ConvolutionalLayer(nn.Module):
    def __init__(self, in_channels, out_channels, filter_size, stride=1, padding=0, dilation = 1, bias=True, padding_mode='zeros', device=device):
        super(ConvolutionalLayer, self).__init__()
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.filter_size = filter_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.device = device
        #initialize weights and biases
        self.weight = torch.randn(out_channels, in_channels, filter_size, filter_size, device = self.device) * ((2/in_channels) * 0.2)
        self.weight.requires_grad_() # requires grad so that gradients are calculated
        self.kernel = (filter_size, filter_size)
        if bias==True:
            self.bias = torch.randn(in_channels, filter_size, filter_size, device = self.device, requires_grad = True)
        else:
            self.bias = torch.zeros_like(in_channels, filter_size, filter_size, device = self.device, requires_grad = False)
    def __call__(self, x):
        #find the new image height and width after its reduced
        image_height = ((x.shape[2] + (self.padding * 2) - self.filter_size) // self.stride) + 1
        image_width = ((x.shape[3] + (self.padding * 2) - self.filter_size) // self.stride) + 1
        #pad all edges of the matrix with a 0 so that kernel can capture the edges
        x = torch.nn.functional.pad(x, (self.padding, self.padding, self.padding, self.padding), mode = 'constant', value = 0)
        #unfold ex. 4x3x9x9 (3x3 kernel) -> 4x27x81
        unfold = nn.Unfold(
            kernel_size=(self.filter_size, self.filter_size), 
            stride = self.stride
        )
        x = unfold(x)
        #permute to bring to in_channel, filter_size, filter_size, out_channels
        self.weight = self.weight.permute(1, 2, 3, 0)
        #view to make sure its in_channels*filter_size*filter_size, out_channels
        self.weight = self.weight.view(-1, self.out_channels)
        #permute x so that its batch_size, patches, patch_size
        #matrix multiply by weight, patch_size dimension should match in_channels*filter_size*filter_size
        x = x.permute(0, 2, 1)@self.weight # -> batch_size, patches, out_channel
        #permute it to batch_size, out_channels, patches
        x = x.permute(0, 2, 1)
        #reshape it to batch_size, out_channels, image_height, image_width
        x = x.reshape(batch_size, self.out_channels, image_height, image_width)
        #make sure weight is back to correct dimensions, or we could leave default weight shape as in_channels*filter_size*filter_size, out_channels, but this might make it easier to understand
        self.weight = self.weight.view(self.in_channels, self.filter_size, self.filter_size, self.out_channels)
        self.weight = self.weight.permute(3, 0, 1, 2)
        self.out = x
        #return the output
        return self.out
    def parameters(self):
        #return all the parameters involved (bias is not currently used, but return it anyways)
        return [self.weight, self.bias]

class ConvolutionalTransposeLayer(nn.Module):
    def __init__(self, in_channels, out_channels, filter_size,  stride=1, padding=0, output_padding = 0, groups = 0, bias=True, dilation = 1, padding_mode='zeros', device=device):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.filter_size = filter_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        self.dilation = dilation
        self.device = device
        self.weight = torch.randn(in_channels, out_channels, filter_size, filter_size, device=device) * (2/in_channels)
        self.weight.requires_grad_()
        if bias==True:
            self.bias = torch.randn(out_channels, device = self.device, requires_grad = True)
        else:
            self.bias = torch.zeros(out_channels, device = self.device, requires_grad = False)
    def get_output_size(self, input_size):
        #return the output size of an input based on stride and filter size
        return (
            (input_size - 1) * self.stride
            - 2 * self.padding + self.dilation * (self.filter_size - 1)
            + self.output_padding
            + 1
        )
    def __call__(self, x):
        #view weights as in_channels, out_channels*filter_size*filter_size
        self.weight = self.weight.view(self.in_channels, -1)
        #get the output size, we only need one because our image is a square, would need two different if not a square though
        output_size = (self.get_output_size(x.shape[2]), self.get_output_size(x.shape[3]))
        #view x as batch_size, in_channels, image_height*image_width
        #permute x to batch_size, image_height*image_width, in_channels
        x = x.view(batch_size, self.in_channels, -1).permute(0, 2, 1)
        #matrix multiply by weight
        #output is batch_size, image_height*image_width, out_channels*filter_size*filter_size
        output = x@self.weight
        #permute to batch_size, out_channels*filter_size*filter_size, image_height*image_width
        output = output.permute(0, 2, 1)
        #fold - folds the output to an image of desired dimensions
        #sums up all the matrixes ontop of each other correctly
        output = torch.nn.functional.fold(output, output_size, stride = self.stride, kernel_size=self.filter_size, padding=self.padding)
        self.out = output
        #return weight to orginal dimensions
        self.weight = self.weight.reshape(self.in_channels, self.out_channels, self.filter_size, self.filter_size)
        return self.out
    def parameters(self):    
        return [self.weight, self.bias]

class BatchNorm(nn.Module):
    
    def __init__(self, features, device = device):
        super(BatchNorm, self).__init__()
        self.device = device
        self.weight = torch.ones((1, features, 1, 1), device = device, requires_grad = True)
        self.bias = torch.zeros((1, features, 1, 1), device = device,  requires_grad = True)
    def __call__(self, x):
        #mean of all elements, doesn't mix between channels
        xmean = x.mean(dim=(0, 2, 3), keepdim = True)
        #variance of all elements, doesn't mix between channels
        xvar = x.var(dim=(0, 2, 3), keepdim = True)
        #batchnorm equation
        #add e-5 incase xvar is 0
        y = (x - xmean) / torch.sqrt(xvar + 0.00001)
        #multiply by weight and add some bias
        self.out = self.weight * y + self.bias
        return self.out
    def parameters(self):
        return [self.weight, self.bias]

class LeakyRelu():
    def __call__(self, x):
        leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        return leaky_relu(x)
class Relu():
    def __call__(self, x):
        relu = nn.ReLU(True)
        return relu(x)
class Tanh():
    def __call__(self, x):
        tanh = nn.Tanh()
        return tanh(x)
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.layers = [
            #x = 1, 100, 1, 1
            ConvolutionalTransposeLayer(nz, ngf*8, filter_size=4, stride=1, padding=0, bias=False, device=device),    BatchNorm(ngf*8), Relu(),
            #x = 1, 512, 4, 4
            ConvolutionalTransposeLayer(ngf*8, ngf*4, filter_size=4, stride=2, padding=1, bias=False, device=device), BatchNorm(ngf*4), Relu(),
            #x = 1, 256, 8, 8
            ConvolutionalTransposeLayer(ngf*4, ngf*2, filter_size=4, stride=2, padding=1,bias=False, device=device),  BatchNorm(ngf*2), Relu(),
            #x = 1, 128, 16, 16
            ConvolutionalTransposeLayer(ngf*2, ngf, filter_size=4, stride=2, padding=1, bias=False, device=device),   BatchNorm(ngf),   Relu(),
            #x = 1, 64, 32, 32
            ConvolutionalTransposeLayer(ngf, nc, filter_size=4, stride=2, padding=1, bias=False, device=device),      BatchNorm(nc),    Tanh()
            #x = 1, 3, 64, 64  
        ]
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        self.out = x
        return self.out
    def parameters(self):
        params = []
        for layer in self.layers:
            if isinstance(layer, nn.Module):  # If it's a valid module (e.g., ConvolutionalLayer)
                params += list(layer.parameters())  # Get parameters of the layer
        return params

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.layers = [
            #3 in_channels, 4 out_channels, 3 filter_size
            ConvolutionalLayer(nc, ndf, 4, 2, 1, device=device),                        LeakyRelu(),
            ConvolutionalLayer(ndf, ndf*2, 4, 2, 1, device=device),   BatchNorm(ndf*2), LeakyRelu(),
            ConvolutionalLayer(ndf*2, ndf*4, 4, 2, 1, device=device), BatchNorm(ndf*4), LeakyRelu(),
            ConvolutionalLayer(ndf*4, ndf*8, 4, 2, 1, device=device), BatchNorm(ndf*8), LeakyRelu(),
            ConvolutionalLayer(ndf*8, 1, 4, 1, 0, device=device)
        ]
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        self.out = x
        return self.out
    def parameters(self):
        params = []
        for layer in self.layers:
            if isinstance(layer, nn.Module):  # If it's a valid module (e.g., ConvolutionalLayer)
                params += list(layer.parameters())  # Get parameters of the layer
        return params


if __name__ == '__main__':
    img_list = []
    G_losses = []
    D_losses = []
    iterations = []
    iters = 1
    discriminator = Discriminator()
    generator = Generator()
    discriminator.to(device)
    generator.to(device)




    labels = torch.ones(128, 1, 1, 1, device=device)
    fake_labels = torch.zeros(128, 1, 1, 1, device=device)
    
    real_labels = 1
    optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.0001, betas=(0.5, 0.999))
    optimizerG = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    #training loop
    for i in range(10):
        for i, data in enumerate(dataloader, 0):
            images, label = data
            if images.size(0) < batch_size:
                continue

            images, label = images.to(device), label.to(device)

            optimizer.zero_grad()

            #training on real images
            images = add_noise(images)
            output = discriminator(images)
            preloss = torch.sigmoid(output)
            loss_real = F.binary_cross_entropy(preloss, labels)
            print(f"loss_real: {loss_real.data}")
            #loss.backward()

            #use generator to create fake images
            noise = torch.randn(batch_size, nz, 1, 1, device=device)
            fake = generator(noise)
            #print(fake.shape)
            fake_images = fake.detach() #detach to make sure generator weights are not updated with the discriminator

            #training on fake images
            outputG = discriminator(fake_images)
            prelossG = torch.sigmoid(outputG)
            loss_fake = F.binary_cross_entropy(prelossG, fake_labels)
            loss_total = loss_real + loss_fake
            print(f"loss_fake: {loss_fake.data}")

            #update discriminator
            #print(loss_total.data)
            loss_total.backward()
            optimizer.step()


            #training the generator, do another forward pass for the disciminator, but only update weights for generator
            outputSecond = discriminator(fake)
            preloss_generator = torch.sigmoid(outputSecond)
            loss_generator = F.binary_cross_entropy(preloss_generator, labels)
            print(f"loss_generator: {loss_generator.data}")

            #update generator
            optimizerG.zero_grad()
            loss_generator.backward()
            optimizerG.step()
            
            #save losses for plotting
            G_losses.append(loss_generator)
            D_losses.append(loss_total)
            iterations.append(iters)
            iters += 1
    D_losses_tensor = torch.tensor(D_losses)
    G_losses_tensor = torch.tensor(G_losses)
    plt.plot(iterations, D_losses_tensor.cpu(), label='Discriminator Loss', color='blue', marker='o')
    plt.plot(iterations, G_losses_tensor.cpu(), label='Generator Loss', color='red', marker='x')
    
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('Generator and Discriminator Losses over Iterations')
    plt.legend()

    # Display the plot
    plt.show()
    print("done")

    matrixI = torch.randn(128, nz, 1, 1, device=device)
    fake = generator(matrixI)
    # image_tensor = fake[0]
    # image_np = image_tensor.permute(1, 2, 0).detach().cpu().numpy()
    # image_np = (image_np + 1) / 2
    # plt.imshow(image_np)
    # plt.axis('off')  # Hide axes
    # plt.show()
    # Create a grid for displaying images
    plt.figure(figsize=(8,8))
    plt.axis("off")
    plt.title("Generated Images")
    plt.imshow(np.transpose(vutils.make_grid(fake.to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))
    plt.show()