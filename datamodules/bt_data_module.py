import os
from typing import Optional

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split

import torchvision 

from datasets import CIFAR_BT
from augmentations import BT_Transforms


class BT_DataModule(LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.batch_size = config.batch_size
        self.data_dir = config.data_dir
        
        self.transform1, self.transform2 = BT_Transforms(config.img_size)


    def prepare_data(self):
        torchvision.datasets.CIFAR10(root=self.data_dir, train=True,
                                        download=True, transform=None)
        torchvision.datasets.CIFAR10(root=self.data_dir, train=False,
                                        download=True, transform=None)

    def setup(self, stage: Optional[str] = None):
        
        # split dataset
        if stage in (None, "fit"):
            cifar_train = torchvision.datasets.CIFAR10(root=self.data_dir, train=True)
            cifar_val = torchvision.datasets.CIFAR10(root=self.data_dir, train=False)
            self.data_train = CIFAR_BT( cifar_train, transform1= self.transform1, transform2= self.transform2)
            self.data_val = CIFAR_BT( cifar_val,transform1= self.transform1, transform2= self.transform2)

    # return the dataloader for each split
    def train_dataloader(self):
        train_loader = DataLoader(self.data_train, batch_size=self.batch_size, shuffle=True)
        return train_loader

    def val_dataloader(self):
        val_loader = DataLoader(self.data_val, batch_size=self.batch_size, shuffle=False)
        return val_loader
    '''
    def test_dataloader(self):
        mnist_test = DataLoader(self.mnist_test, batch_size=self.batch_size)
        return mnist_test

    def predict_dataloader(self):
        mnist_predict = DataLoader(self.mnist_predict, batch_size=self.batch_size)
        return mnist_predict
    '''