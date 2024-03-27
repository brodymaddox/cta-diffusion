# Written by Brody Maddox
# Adapted from https://www.youtube.com/watch?v=a4Yfz2FxXiY

from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import cta_dataset

def load_transformed_dataset(IMG_SIZE):
    data_transforms = [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), # Scales data to [0,1]
        transforms.Lambda(lambda t: (t*2)-1) # Scales between [-1,1]
    ]
    data_transform = transforms.Compose(data_transforms)
    data = cta_dataset.CTAngiographyNoConditionDataset(root_dir='/home/brody/Laboratory/cta-diffusion/experiments/test_exp/', csv='annotations.csv', transform=data_transform)
    return data

def show_tensor_image(image):
    reverse_transforms = transforms.Compose([
        transforms.Lambda(lambda t: (t+1)/2),
        transforms.Lambda(lambda t: t.permute(1,2,0)), #CHW to HWC
        transforms.Lambda(lambda t: t * 255.),
        transforms.Lambda(lambda t: t.numpy().astype(np.uint8)),
        transforms.ToPILImage(),
    ])

    # Take first image of batch
    if len(image.shape) == 4:
        image = image[0, :, :, :]
    plt.imshow(reverse_transforms(image))

