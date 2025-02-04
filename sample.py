
import torch
import torch.optim as optim
import torchvision.datasets as dset
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
import torchvision.utils as vutils
from model import Generator, Discriminator, add_noise, ConvolutionalTransposeLayer, BatchNorm, Relu, LeakyRelu, Tanh






ngpu = 1
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")


generator = Generator()
generator.to(device)

with torch.serialization.safe_globals([Generator, ConvolutionalTransposeLayer, BatchNorm, Relu, LeakyRelu, Tanh]):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    generator = Generator()  # Make sure your model is defined
    #generator.load_state_dict(torch.load("models/generator_epoch_4.pth", map_location=device))
    generator = torch.load("models/generator_epoch_330.pth", map_location=device)

#sample the model
matrixI = torch.randn(128, 100, 1, 1, device=device)
fake = generator(matrixI)
plt.figure(figsize=(8,8))
plt.axis("off")
plt.title("Generated Images")
plt.imshow(np.transpose(vutils.make_grid(fake.to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))
plt.show()