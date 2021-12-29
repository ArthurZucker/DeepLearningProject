import numpy as np
import torch

#torch dataset library
from torch.utils.data import Dataset

#Libraries for visualizations
import albumentations as A
from albumentations.pytorch import ToTensorV2
import copy
import matplotlib.pyplot as plt
import random

class CIFAR_DINO(Dataset):
    def __init__(self, data,  n_crops, n_global_crops, transform_global1= None, transform_global2= None, transform_local= None):

        self.data = data
        self.transform_global1 = transform_global1
        self.transform_global2 = transform_global2
        self.transform_local = transform_local
        self.n_crops = n_crops
        self.n_global_crops = n_global_crops

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        #get the image at index zero
        image = np.array(self.data[idx][0])
        
        #Create a list of image crops with global and local transforms
        transformed_crops = []
        transformed_crops.append(self.transform_global1(image = image)['image'])
        transformed_crops.append(self.transform_global2(image = image)['image'])
            
        for _ in range(self.n_crops - self.n_global_crops):
            transformed_crops.append(self.transform_local(image = image)['image'])
        
        return transformed_crops
    
    
class CIFAR_BT(Dataset):
    def __init__(self, data, transform1= None, transform2= None):

        self.data = data
        self.transform1 = transform1
        self.transform2 = transform2

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        label = self.data[idx][1]

        image = np.array(self.data[idx][0])
        image1 = image
        image2 = image
        
        #Transform the same image with 2 different transforms
        if self.transform1 is not None:
            image1 = self.transform1(image = image)['image']

        if self.transform2 is not None:
            image2 = self.transform2(image = image)['image']
        

        return image1, image2, label    
    
