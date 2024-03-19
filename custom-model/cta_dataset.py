import torch
from torch.utils.data import Dataset
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
        |-- annotations.csv [index, name]

        """
        img_name = os.path.join(self.root_dir, self.context_df['name'][idx])
        img = Image.open(img_name)
        img_np = np.array(img)
        
        if self.transform:
            img_np = self.transform(img_np)
        

        if self.conditioning:
            condition = []
            for col in self.condition_columns:
                condition.append(self.context_df[col][idx])
            sample = {'image': img_np, 'condition': condition}
        else:
            sample = {'image': img_np}
            
        return sample




