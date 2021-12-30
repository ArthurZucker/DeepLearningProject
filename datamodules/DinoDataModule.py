import os

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
import datasets 

class DinoDataModule(LightningDataModule):
    def __init__(self, config, dataset_name = "DinoDataset"):
        super().__init__()
        self.dataset    = getattr(datasets.Dino, dataset_name)
        self.config     = config
        self.batch_size = self.config.batch_size
        self.root = os.path.join(self.config.asset_path, "CIFAR10")

    def prepare_data(self):
    #use to download
        self.dataset(
                root = self.root, img_size = self.config.input_size, n_crops = self.config.n_crops, 
                n_global_crops = self.config.n_global_crops, local_crops_scale = self.config.local_crops_scale, 
                global_crops_scale = self.config.global_crops_scale, train = True, download=True
            )

        self.dataset(
                root = self.root, img_size = self.config.input_size, n_crops = self.config.n_crops, 
                n_global_crops = self.config.n_global_crops, local_crops_scale = self.config.local_crops_scale, 
                global_crops_scale = self.config.global_crops_scale, train = False, download=True
            )

    # OPTIONAL, called for every GPU/machine (assigning state is OK)
    def setup(self, stage=None):
        # transforms
        # split dataset
        if stage in (None, "fit"):
            self.cifar_train = self.dataset(
                root = self.root, img_size = self.config.input_size, n_crops = self.config.n_crops, 
                n_global_crops = self.config.n_global_crops, local_crops_scale = self.config.local_crops_scale, 
                global_crops_scale = self.config.global_crops_scale, train = True, download=True
            )
            self.cifar_val = self.dataset(
                root = self.root, img_size = self.config.input_size, n_crops = self.config.n_crops, 
                n_global_crops = self.config.n_global_crops, local_crops_scale = self.config.local_crops_scale, 
                global_crops_scale = self.config.global_crops_scale, train = False, download=True
            )

    def train_dataloader(self):
        cifar_train = DataLoader(
            self.cifar_train,
            batch_size=self.batch_size,
            num_workers=self.config.num_workers,
            shuffle=True,
        )
        return cifar_train

    def val_dataloader(self):
        cifar_val = DataLoader(
            self.cifar_val,
            batch_size=self.batch_size,
            num_workers=self.config.num_workers,
            shuffle=False,
        )
        return cifar_val