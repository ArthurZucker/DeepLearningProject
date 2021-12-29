
import numpy as np
import torch

#torch dataset library
from torch.utils.data import Dataset
from utils.transforms import BarlowTwinsTransform
from torchvision.datasets import CIFAR10
from PIL import Image

class BarlowTwinsDataset(CIFAR10):
    def __init__(self, root, img_size, train,download = False):
        super().__init__(root=root, train = train, download = download)
        self.transform = BarlowTwinsTransform(img_size)

    def __getitem__(self, idx):
        image = Image.fromarray(self.data[idx]) # FIXME Image.fromarray might not be needed
        image = np.array(image)
        # Transform the same image with 2 different transforms
        if self.transform is not None:
            aug_image1, aug_image2 = self.transform(image = image)
        return aug_image1, aug_image2    
    
