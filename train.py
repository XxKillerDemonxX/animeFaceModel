import torch
import torch.optim as optim
import torchvision.datasets as dset
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
import torchvision.utils as vutils
from model import Generator, Discriminator, add_noise, ConvolutionalTransposeLayer, BatchNorm, Relu, LeakyRelu, Tanh

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
#print(torch.cuda.is_available())
#print(torch.__version__)





if __name__ == '__main__':
    img_list = []
    G_losses = []
    D_losses = []
    iterations = []
    iters = 1
    discriminator = Discriminator()
    generator = Generator()

    epoch_num = 300
    with torch.serialization.safe_globals([Generator, ConvolutionalTransposeLayer, BatchNorm, Relu, LeakyRelu, Tanh]):
        generator = torch.load(f"models/generator_epoch_{epoch_num}.pth", map_location=device, weights_only=False)
        discriminator = torch.load(f"models/discriminator_epoch_{epoch_num}.pth", map_location=device, weights_only=False)

    discriminator.to(device)
    generator.to(device)

    generator_path = "models/split3/generator_epoch_{epoch}.pth"
    discriminator_path = "models/split3/discriminator_epoch_{epoch}.pth"


    labels = torch.ones(128, 1, 1, 1, device=device)
    fake_labels = torch.zeros(128, 1, 1, 1, device=device)
    
    real_labels = 1
    optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.0001, betas=(0.5, 0.999))
    optimizerG = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    #training loop
    for epoch in range(30):
        print(f"epoch({epoch + 1}/30)")
        for i, data in enumerate(dataloader, 0):
            images, label = data
            if images.size(0) < batch_size:
                continue

            images, label = images.to(device), label.to(device)

            optimizer.zero_grad()

            #training on real images
            #images = add_noise(images)
            output = discriminator(images)
            preloss = torch.sigmoid(output)
            loss_real = F.binary_cross_entropy(preloss, labels)
            #print(f"loss_real: {loss_real.data}")
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
            #print(f"loss_fake: {loss_fake.data}")

            #update discriminator
            #print(loss_total.data)
            loss_total.backward()
            optimizer.step()


            #training the generator, do another forward pass for the disciminator, but only update weights for generator
            outputSecond = discriminator(fake)
            preloss_generator = torch.sigmoid(outputSecond)
            loss_generator = F.binary_cross_entropy(preloss_generator, labels)
            #print(f"loss_generator: {loss_generator.data}")

            #update generator
            optimizerG.zero_grad()
            loss_generator.backward()
            optimizerG.step()
            
            #save losses for plotting
            G_losses.append(loss_generator)
            D_losses.append(loss_total)
            iterations.append(iters)
            iters += 1

        if ((epoch+epoch_num+1)%10 == 0):
            torch.save(generator, generator_path.format(epoch=epoch_num+epoch+1))
            torch.save(discriminator, discriminator_path.format(epoch=epoch_num+epoch+1))
    
    #plot and display loss over all iterations
    D_losses_tensor = torch.tensor(D_losses)
    G_losses_tensor = torch.tensor(G_losses)
    plt.plot(iterations, D_losses_tensor.cpu(), label='Discriminator Loss', color='blue', marker='o')
    plt.plot(iterations, G_losses_tensor.cpu(), label='Generator Loss', color='red', marker='x')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('Generator and Discriminator Losses over Iterations')
    plt.legend()
    plt.show()

    #sample the model
    matrixI = torch.randn(128, nz, 1, 1, device=device)
    fake = generator(matrixI)
    plt.figure(figsize=(8,8))
    plt.axis("off")
    plt.title("Generated Images")
    plt.imshow(np.transpose(vutils.make_grid(fake.to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))
    plt.show()