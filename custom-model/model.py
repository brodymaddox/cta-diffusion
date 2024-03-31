# Written by Brody Maddox
# Code Adapted from https://www.youtube.com/watch?v=a4Yfz2FxXiY

import torch
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
import torchvision
import math

class Block(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, up=False):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        if up:
            self.conv1 = nn.Conv2d(2*in_ch, out_ch, 3, padding=1)
            self.transform = nn.ConvTranspose2d(out_ch, out_ch, 4, 2, 1)
        else:
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
            self.transform = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bnorm1 = nn.BatchNorm2d(out_ch)
        self.bnorm2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU()
    
    def forward(self, x, t):
        # First Conv
        h = self.bnorm1(self.relu(self.conv1(x)))
        # Time Embedding
        time_emb = self.relu(self.time_mlp(t))
        # Extend last 2 dimensions
        time_emb = time_emb[(..., ) + (None, ) * 2]
        # Add time channel
        h = h + time_emb
        # Second Conv
        h = self.bnorm2(self.relu(self.conv2(h)))
        # Down or Upsample
        return self.transform(h)


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    
    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class SimpleUnet(nn.Module):
    """
    A Simplified variant of the Unet Architecture
    """
    def __init__(self):
        super().__init__()
        image_channels = 1
        down_channels = (64, 128, 256, 512, 1024)
        up_channels = (1024, 512, 256, 128, 64)
        out_dim = 1
        time_emb_dim = 32

        # Time Embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU()
        )

        # Initial Projection
        self.conv0 = nn.Conv2d(image_channels, down_channels[0], 3, padding=1)

        # Downsample
        self.downs = nn.ModuleList([Block(down_channels[i], down_channels[i+1], \
                                          time_emb_dim) for i in range(len(down_channels)-1)])
        
        # Upsample
        self.ups = nn.ModuleList([Block(up_channels[i], up_channels[i+1], \
                                        time_emb_dim, up=True) for i in range(len(up_channels)-1)])
        
        self.output = nn.Conv2d(up_channels[-1], 1, out_dim)

    def forward(self, x, timestep):
        # Embed time
        t = self.time_mlp(timestep)
        # Initial conv
        x = self.conv0(x)
        # Unet
        residual_inputs = []
        for down in self.downs:
            x = down(x, t)
            residual_inputs.append(x)
        for up in self.ups:
            residual_x = residual_inputs.pop()
            # Add residual as additional channels
            x = torch.cat((x, residual_x), dim=1)
            x = up(x, t)
        return self.output(x)
    


class ChannelConditionalUNET(nn.Module):
    """
    A U-net that appends conditional data as additional channels before each double convolution block
    Before use must fill in image_sizes, conditional channels, init_cond_channels
    """
    def __init__(self):
        super().__init__()
        image_channels = 1
        self.batch_size = 8
        self.image_sizes = [(256,256), (128,128), (64,64), (32,32), (16,16)]
        self.init_cond_channels = 8
        self.conditional_channels = [8, 8, 8, 8, 8]
        self.up_conditional_channels = self.conditional_channels[::-1]
        self.up_img_sizes = self.image_sizes[::-1]
        down_channels = (64, 128, 256, 512, 1024)
        up_channels = (1024, 512, 256, 128, 64)
        out_dim = 1
        time_emb_dim = 32

        # Time Embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU()
        )

        # Initial Projection
        self.conv0 = nn.Conv2d(image_channels + self.init_cond_channels, down_channels[0], 3, padding=1)

        # Downsample
        self.downs = nn.ModuleList([Block(down_channels[i] + self.conditional_channels[i], down_channels[i+1], \
                                          time_emb_dim) for i in range(len(down_channels)-1)])
        
        # Upsample
        self.ups = nn.ModuleList([Block(up_channels[i] + int(self.up_conditional_channels[i]/2), up_channels[i+1], \
                                        time_emb_dim, up=True) for i in range(len(up_channels)-1)])
        # Quick and dirty fix, the up block doubles the inputted channels to account for residuals, so we only want to add half of the conditioning channels
        # This forces the number of conditioning channels to be even to work

        self.output = nn.Conv2d(up_channels[-1], 1, out_dim)

    def forward(self, x, condition, timestep):
        """
        Forward propogation that concatenates condition tensor at each stage
        """
        # Embed time
        t = self.time_mlp(timestep)

        # Initialize master copy of condition
        cond = condition

        # Initial condition reshape
        cond0 = cond.unsqueeze(-1).unsqueeze(-1).expand(-1,-1,self.image_sizes[0][0], self.image_sizes[0][1])
        cond0 = cond0.to('cuda')

        # Initial Concatenation
        x = torch.cat((x, cond0), dim=1)

        # Initial conv
        x = self.conv0(x)

        # Unet
        residual_inputs = []
        for i, down in enumerate(self.downs):

            # Create condition tensor and concatenate
            cond = condition.unsqueeze(-1).unsqueeze(-1).expand(-1,-1,self.image_sizes[i][0], self.image_sizes[i][0])
            cond = cond.to('cuda')
            x = torch.cat((x, cond), dim=1)

            # Conduct the down block
            x = down(x, t)

            # Store residual input for later
            residual_inputs.append(x)

        for i, up in enumerate(self.ups):

            # Create condition tensor
            cond = condition.unsqueeze(-1).unsqueeze(-1).expand(-1,-1,self.up_img_sizes[i][0], self.up_img_sizes[i][0])
            cond = cond.to('cuda')

            # Access residual input
            residual_x = residual_inputs.pop()

            # Add residual and condition as additional channels
            x = torch.cat((x, residual_x, cond), dim=1)
            
            # Conduct the up block
            x = up(x, t)

        return self.output(x)
        

