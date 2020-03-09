import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
from torchvision import datasets, transforms
import torchvision.utils
import os
import sys
import time
import math

latent_dim = 100
channels = 3
batch_size = 64

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # Size : 512 x 4 x4 
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # 256 x 8 x 8
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # 128 x 16 x 16
            nn.ConvTranspose2d(128, channels, 4, 2, 1, bias=False),
            nn.Tanh()
            # 1 x 32 x 32
        )

    def forward(self, input):
        return self.main(input)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # 1 x 32 x 32
            nn.Conv2d(channels, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # 128 x 16 x 16
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # 256 x 8 x 8
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            # 512 x 4 x 4
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)



gen = Generator()
disc = Discriminator()

trainLoader = torch.utils.data.DataLoader(
    datasets.CIFAR10('../data', train=True, download=True,
                    transform=transforms.Compose([
                        transforms.Resize(32),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                    ])),
    batch_size=batch_size, shuffle=True, drop_last=True)

# put on gpu if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
gen.to(device)
disc.to(device)

gen_optimizer = optim.Adam(gen.parameters(), lr=0.0002, betas=(0.5, 0.999))
desc_optimizer = optim.Adam(disc.parameters(), lr=0.0002, betas=(0.5, 0.999))
criterion = nn.BCELoss()

for epoch in range(20):

    for i, (input_batch, _) in enumerate(trainLoader, 0):
        input_batch = input_batch.to(device)


        # ==============================#
        # Train the discriminator       #
        # ==============================#

        disc.zero_grad()

        # loss on real data
        real_desc = disc(input_batch)
        desc_loss_real = criterion(real_desc, torch.ones_like(real_desc))

        #loss on fake data
        noise = torch.randn(batch_size, latent_dim, 1, 1, device=device)
        gen_imgs = gen(noise)
        gen_desc = disc(gen_imgs)
        desc_loss_fake = criterion(gen_desc, torch.zeros_like(gen_desc))

        desc_loss = desc_loss_fake + desc_loss_real
        desc_loss.backward()
        desc_optimizer.step()

        # ============================= #
        # Train the generator           #
        # ============================= #

        gen.zero_grad()

        noise = torch.randn(batch_size, latent_dim, 1, 1, device=device)
        gen_imgs = gen(noise)
        gen_desc = disc(gen_imgs)
        gen_loss = criterion(gen_desc,  torch.ones_like(gen_desc))

        gen_loss.backward()
        gen_optimizer.step()    

        if i % 200 == 0:
            print('Epoch [{}], Step [{}], desc_loss: {:.4f}, gen_loss: {:.4f}, D(x): {:.2f}, D(G(z)): {:.2f}' 
                .format(epoch, i+1, desc_loss.item(), gen_loss.item(), 
                    real_desc.mean().item(), gen_desc.mean().item()))

    disp_imgs = (gen_imgs + 1.) / 2
    torchvision.utils.save_image(disp_imgs, './img_epoch_%.i.png' % epoch)

