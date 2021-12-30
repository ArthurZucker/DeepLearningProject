import numpy as np
import torch

#torch dataset library
from torch.utils.data import Dataset
from utils.transforms import DinoTransforms
from torchvision.datasets import CIFAR10
from PIL import Image

class DinoDataset(CIFAR10):
    def __init__(self, root, img_size, n_crops, n_global_crops, local_crops_scale, global_crops_scale, train, download = False):
        super().__init__(root=root, train = train, download = download)
        self.transform = DinoTransforms(img_size, n_crops, n_global_crops, local_crops_scale, global_crops_scale)

    def __getitem__(self, idx):
        image = Image.fromarray(self.data[idx]) 
        image = np.array(image)
        # Make a list of transformed images, with the first images of the list being global crops
        if self.transform is not None:
            aug_crops = self.transform(image = image)
        return aug_crops