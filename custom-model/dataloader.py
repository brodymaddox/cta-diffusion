# Written by Brody Maddox
# Adapted from https://www.youtube.com/watch?v=a4Yfz2FxXiY

from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import cta_dataset
from PIL import Image, ImageOps

# Custom Letter Box Image Transform
class LetterboxTransform:
    def __init__(self, target_size):
        self.target_size = target_size

    def __call__(self, img):
        # Calculate new size that preserves aspect ratio
        aspect_ratio = img.width / img.height
        if aspect_ratio > 1: # Width greatest dim
            new_size = (self.target_size, int(self.target_size / aspect_ratio))
        else: # Height greatest dim
            new_size = (int(self.target_size * aspect_ratio), self.target_size)
        
        # Conduct Interpolation to aspect ratio preserved size
        img = img.resize(new_size, Image.BILINEAR)

        # Calculate padding amount
        padding = (self.target_size - new_size[0], self.target_size - new_size[1])
        left_pad = padding[0] // 2
        top_pad = padding[1] // 2
        padding = (left_pad, top_pad, self.target_size - new_size[0] - left_pad, self.target_size - new_size[1] - top_pad)

        # Apply Padding
        img = ImageOps.expand(img, padding, fill=0)
        return img



def load_transformed_dataset(IMG_SIZE):
    data_transforms = [
        LetterboxTransform(IMG_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), # Scales data to [0,1]
        transforms.Lambda(lambda t: (t*2)-1) # Scales between [-1,1]
    ]
    data_transform = transforms.Compose(data_transforms)
    data = cta_dataset.CTAngiographyNoConditionDataset(root_dir='/home/brody/Laboratory/cta-diffusion/experiments/all_condition_all_slice/', csv='annotations.csv', transform=data_transform)
    return data

def load_transformed_conditional_dataset(IMG_SIZE, conditioning_columns=[]):
    data_transforms = [
        LetterboxTransform(IMG_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), # Scales data to [0,1]
        transforms.Lambda(lambda t: (t*2)-1) # Scales between [-1,1]
    ]
    data_transform = transforms.Compose(data_transforms)
    data = cta_dataset.CTAngiographyDataset(root_dir='/home/brody/Laboratory/cta-diffusion/experiments/all_condition_all_slice/', csv='annotations.csv', transform=data_transform, conditioning=True, 
                                            condition_columns=conditioning_columns)
    return data

def show_tensor_image(image):
    reverse_transforms = transforms.Compose([
        transforms.Lambda(lambda t: (t+1)/2),
        transforms.Lambda(lambda t: t.permute(1,2,0)), #CHW to HWC
        transforms.Lambda(lambda t: t * 255.),
        transforms.Lambda(lambda t: t.numpy().astype(np.uint8)),
    ])

    # Take first image of batch
    if len(image.shape) == 4:
        image = image[0, :, :, :]
    plt.imshow(reverse_transforms(image).squeeze(), cmap='gray')

