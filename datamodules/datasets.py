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
    
    
    
def visualize_augmentations_BT_dataset(dataset, idx=0, samples=10, cols=5):
    dataset = copy.deepcopy(dataset)
    dataset.transform1 = A.Compose([t for t in dataset.transform1 if not isinstance(t, (A.Normalize, ToTensorV2))])
    dataset.transform2 = A.Compose([t for t in dataset.transform2 if not isinstance(t, (A.Normalize, ToTensorV2))])

    rows = samples // cols
    figure, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(12, 6))
    for i in range(samples):
        image1,image2,  _ = dataset[idx]
        images = [image1,image2]
        rand = random.randint(0,1)
        ax.ravel()[i].imshow(images[rand])
        ax.ravel()[i].set_axis_off()
    plt.tight_layout()
    plt.show()  


def visualize_augmentations_DINO_dataset(dataset, idx=0, samples=10, cols=5):
    dataset = copy.deepcopy(dataset)
    dataset.transform_global1 = A.Compose([t for t in dataset.transform_global1 if not isinstance(t, (A.Normalize, ToTensorV2))])
    dataset.transform_global2 = A.Compose([t for t in dataset.transform_global2 if not isinstance(t, (A.Normalize, ToTensorV2))])
    dataset.transform_local = A.Compose([t for t in dataset.transform_local if not isinstance(t, (A.Normalize, ToTensorV2))])

    rows = samples // cols
    figure, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(12, 6))
    for i in range(samples):
        images = dataset[idx]
        rand = random.randint(0,7)
        image = np.transpose(images[rand], (1,0,2))
        ax.ravel()[i].imshow(image)
        ax.ravel()[i].set_axis_off()
    plt.tight_layout()
    plt.show()   