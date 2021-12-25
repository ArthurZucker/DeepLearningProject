import numpy as np
import torch

#torch dataset library
from torch.utils.data import Dataset

class CIFAR_DINO(Dataset):
    def __init__(self, config, data, transform_global1= None, transform_global2= None, transform_local= None):

        self.data = data
        self.transform_global1 = transform_global1
        self.transform_global2 = transform_global2
        self.transform_local = transform_local
        self.n_crops = config.n_crops
        self.n_global_crops = config.n_global_crops

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