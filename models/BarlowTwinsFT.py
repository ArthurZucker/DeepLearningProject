import torch.nn as nn
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from pytorch_lightning import LightningModule
from torch.nn import functional as F
import torch
from utils.agent_utils import get_net

from models.optimizers.lars import LARS

class BarlowTwinsFT(LightningModule):
    def __init__(self, network_param,optim_param, barlow_twins):
        """method used to define our model parameters
        Args: BarlowConfig : config = network parameters to use. 
        """
        super().__init__()
        # Loss
        self.loss = nn.CrossEntropyLoss()

        # Optimizer parameters
        self.lr = optim_param.lr
        self.optimizer = optim_param.optimizer
        
        # Network parameters 
        self.num_cat = network_param.num_cat
        self.proj_channels = network_param.proj_channels
                
        # Model
        self.barlow_twins = barlow_twins
        self.linear = nn.Linear(self.proj_channels, self.num_cat)

    def forward(self, x):
        
        # Feed the data through pretrained barlow twins and prediciton layer
        out, _ = self.barlow_twins(x, x)
        out = self.linear(out)

        return out

    def training_step(self, batch, batch_idx):
        """needs to return a loss from a single batch"""
        loss = self._get_loss(batch)

        # Log loss and metric
        self.log("train/loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        """used for logging metrics"""
        loss = self._get_loss(batch)

        # Log loss and metric
        self.log("val/loss", loss)

        return loss

    def configure_optimizers(self):
        """defines model optimizer"""
        optimizer = getattr(torch.optim,self.optimizer)
        optimizer = optimizer(self.parameters(), lr=self.lr)
        # scheduler = LinearWarmupCosineAnnealingLR(
        #     optimizer, warmup_epochs=5, max_epochs=40
        # )
        return optimizer #[[optimizer], [scheduler]]

    def _get_loss(self, batch):
        """convenience function since train/valid/test steps are similar"""
        x, label = batch
        out = self(x)
        loss = self.loss(out, label)

        return loss