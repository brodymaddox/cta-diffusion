import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.transforms import Compose, Resize, ToTensor
import os
import pandas as pd
from PIL import Image
import numpy as np

class CTAngiographyDataset(Dataset):

    def __init__(self, root_dir, csv, transform=None, conditioning=False, condition_columns=[]):
        """
        Arguments:
            root_dir (str): path to the directory containing images
            csv (str): path to csv file containing annotations
            transform (callable, optional): optional transform to be 
                applied to each sample
            conditioning (boolean, optional): set to true to enable conditioning
            condition_columns (list(str), optional): columns from context df to condition on
        """
        self.root_dir = root_dir
        self.transform = transform 
        self.context_df = pd.read_csv(csv)
        self.conditioning = conditioning
        self.condition_columns = condition_columns

    def len(self):
        return len(self.context_df)
    
    def __getitem__(self, idx):
        """
        Required dataset structure

        experiment
        |-- images
                |-- img1.png
                |-- img2.png
        |-- annotations.csv [index, name, slice]


        annotations.csv will contain a row for every slice and its relevant clinical data -> duplicate clinical data rows.
        """
        img_name = os.path.join(self.root_dir + "/images/", self.context_df['name'][idx])
        img = read_image(img_name)
        
        if self.transform:
            img = self.transform(img)
    
        if self.conditioning:
            condition = []
            for col in self.condition_columns:
                condition.append(self.context_df[col][idx])
            condition = [float(i)/sum(condition) for i in condition] # Normalize with respect to the sum
            return img, torch.FloatTensor(condition)
        else:
            return img
        

class CTAngiographyNoConditionDataset(Dataset):
    def __init__(self, csv, root_dir, transform=None):
        """
        Arguments:
            root_dir (str): path to the directory containing images
            transform (callable, optional): optional transform to be 
                applied to each sample
        """
        self.root_dir = root_dir
        self.transform = transform
        self.annotations = pd.read_csv(csv)

    def len(self):
        return self.length
    
    def __getitem__(self, idx):
        """
        Required dataset structure

        experiment
        |-- images
                |-- img1.png
                |-- img2.png
        | annotations.csv [index, name]
        """
        img_name = os.path.join(self.root_dir + "/images/", self.context_df['name'][idx])
        img = read_image(img_name)
        
        if self.transform:
            img = self.transform(img)

        return img




