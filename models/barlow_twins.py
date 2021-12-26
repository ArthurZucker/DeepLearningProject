import os

import torch
import torch.nn as nn
import torchmetrics # TODO use later

from pytorch_lightning import LightningModule

from torch.nn import functional as F
from torch.optim import Adam


class BASE_LitModule(LightningModule):
    def __init__(self, config, encoder):
        """method used to define our model parameters"""
        super().__init__()

        # optimizer parameters
        self.lr = config.lr

        # save hyper-parameters to self.hparams (auto-logged by W&B)
        # self.save_hyperparameters()

        # Loss parameter
        self.lmbda = config.lmbda

        # get backbone model and adapt it to the task
        self.in_features = 0
        self.encoder = encoder  # TODO add encoder name to the hparams, use getnet() to get the encoder
        try:
            self.in_features = list(self.encoder.children())[-1].in_features
            last_layer_name = list(self.encoder.named_children())[-1][0]    
            
            #TODO encoder might not always have those layers
            if last_layer_name == "head":
                self.encoder.head = nn.Identity()
            else:
                self.encoder.fc = nn.Identity()
        except:
            print(
                "Encoder should be a torchvision resnet model or timm's VIT"
            )  # TODO Basic VIT will probably be abit too big, we might want to use SWIN

        # Make Projector (3-layers)
        self.proj_channels = config.bt_proj_channels
        proj_layers = []

        for i in range(3):
            if i == 0:
                proj_layers.append(
                    nn.Linear(self.in_features, self.proj_channels, bias=False)
                )
            else:
                proj_layers.append(
                    nn.Linear(self.proj_channels, self.proj_channels, bias=False)
                )
            if i < 2:
                proj_layers.append(nn.BatchNorm1d(self.proj_channels))
                proj_layers.append(nn.ReLU(inplace=True))

        self.proj = nn.Sequential(*proj_layers)

    def forward(self, x1, x2):
        # Feeding the data through the encoder and projector
        z1 = self.proj(self.encoder(x1))
        z2 = self.proj(self.encoder(x2))

        return z1, z2

    def training_step(self, batch, batch_idx):
        """needs to return a loss from a single batch"""
        loss = self._get_loss(batch)

        # Log loss and metric
        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        """used for logging metrics"""
        loss = self._get_loss(batch)

        # Log loss and metric
        self.log("val_loss", loss)

        return loss

    def test_step(self, batch, batch_idx):
        """used for logging metrics"""
        loss = self._get_loss(batch)

        # Log loss and metric
        self.log("test_loss", loss)

    def configure_optimizers(self):
        """defines model optimizer"""
        return Adam(self.parameters(), lr=self.lr)

    def _get_loss(self, batch):
        """convenience function since train/valid/test steps are similar"""
        x1, x2 = batch
        z1, z2 = self(x1, x2)

        loss = bt_loss(z1, z2, self.lmbda)

        return loss

# TODO put the loss in either model/loss or create a folder in utils or a folder in the repo but it might be an over kill
def bt_loss(z1, z2, lmbda):

    # Normalize the projector's output across the batch
    norm_z1 = (z1 - z1.mean(0)) / z1.std(0)
    norm_z2 = (z2 - z2.mean(0)) / z2.std(0)

    # Cross correlation matrix
    batch_size = z1.size(0)
    cc_M = torch.einsum("bi,bj->ij", (norm_z1, norm_z2)) / batch_size

    # Invariance loss
    diag = torch.diagonal(cc_M)
    invariance_loss = ((torch.ones_like(diag) - diag) ** 2).sum()

    # Zero out the diag elements and flatten the matrix to compute the loss
    cc_M.fill_diagonal_(0)
    redundancy_loss = (cc_M.flatten() ** 2).sum()
    loss = invariance_loss + lmbda * redundancy_loss

    return loss
