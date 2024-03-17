# Written by Brody Maddox
# Directly lifted from https://www.youtube.com/watch?v=a4Yfz2FxXiY

from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision

def load_transformed_dataset(IMG_SIZE):
    data_transforms = [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), # Scales data to [0,1]
        transforms.Lambda(lambda t: (t*2)-1) # Scales between [-1,1]
    ]
    data_transform = transforms.Compose(data_transforms)
    train = torchvision.datasets.CIFAR10(root=".", download=True,
                                              transform=data_transform, train=True)
    test = torchvision.datasets.CIFAR10(root=".", download=True,
                                              transform=data_transform, train=False)
    return torch.utils.data.ConcatDataset([train, test])

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

