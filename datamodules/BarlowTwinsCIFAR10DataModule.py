import os

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from datasets.BarlowTwinsDataset import BarlowTwinsDataset

# TODO add validation loader and transforms

class BarlowTwinsCIFAR10DataModule(LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.batch_size = self.config.batch_size
        self.root = os.path.join(self.config.asset_path, "CIFAR10")

    # When doing distributed training, Datamodules have two optional arguments for
    # granular control over download/prepare/splitting data:

    # OPTIONAL, called only on 1 GPU/machine
    # def prepare_data(self):
    # use to download
    # BarlowTwinsDataset(root = self.root, image_set='trainval', download=False)
    # BarlowTwinsDataset(root = self.root, image_set='val', download=False)

    # OPTIONAL, called for every GPU/machine (assigning state is OK)
    def setup(self, stage=None):
        # transforms
        # split dataset
        if stage in (None, "fit"):
            self.cifar_train = BarlowTwinsDataset(
                self.root, img_size=self.config.input_size,train =True,download =True
            )
            self.cifar_val = BarlowTwinsDataset(
                self.root, img_size=self.config.input_size,train = False,download =True
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
            shuffle=True,
        )
        return cifar_val
