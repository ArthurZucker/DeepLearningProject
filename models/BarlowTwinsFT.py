import torch.nn as nn
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from pytorch_lightning import LightningModule
from torch.nn import functional as F
import torch
from models.BarlowTwins import BarlowTwins
from utils.agent_utils import get_net

from models.optimizers.lars import LARS
from models.BarlowTwins import BarlowTwins
class BarlowTwinsFT(LightningModule):
    def __init__(self, network_param,optim_param):
        """method used to define our model parameters
        Args: BarlowConfig : config = network parameters to use. 
        """
        super().__init__()
        # Loss
        self.loss = nn.CrossEntropyLoss()
        # Network parameters 
        self.num_cat = network_param.num_cat
        
        # Optimizer params
        self.optim_param = optim_param
        # Model
        self.barlow_twins = BarlowTwins(network_param)
        if network_param.weight_checkpoint is not None: 
            self.barlow_twins.load_state_dict(torch.load(network_param.weight_checkpoint)["state_dict"])
        self.barlow_twins.requires_grad_(False)
        self.use_backbone_features = network_param.use_backbone_features
        
        # @TODO fix the code correctly 
        if network_param.use_backbone_features:
            # if we want to test on the backbone features, remove the proj
            self.in_features = self.barlow_twins.proj_channels
            self.barlow_twins = self.barlow_twins.encoder
            
        self.linear = nn.Linear(self.in_features, self.num_cat)

    def forward(self, x):
        # Feed the data through pretrained barlow twins and prediciton layer
        if self.use_backbone_features:
            out = self.barlow_twins(x)
        else:
            out, _  = self.barlow_twins(x, x)
        out     = self.linear(out)
        return out

    def training_step(self, batch, batch_idx):
        """needs to return a loss from a single batch"""
        loss,logits = self._get_loss(batch)
        # Log loss and metric
        self.log("train/loss", loss)

        return {"loss": loss, "logits": logits}

    def validation_step(self, batch, batch_idx):
        """used for logging metrics"""
        loss,logits = self._get_loss(batch)
        # Log loss and metric
        self.log("val/loss", loss)
        return {"loss": loss, "logits": logits}

    def configure_optimizers(self):
        """defines model optimizer"""
        optimizer = getattr(torch.optim,self.optim_param.optimizer)
        optimizer = optimizer(self.parameters(), lr=self.optim_param.lr)
        # scheduler = LinearWarmupCosineAnnealingLR(
        #     optimizer, warmup_epochs=5, max_epochs=40
        # )
        return optimizer #[[optimizer], [scheduler]]

    def _get_loss(self, batch):
        """convenience function since train/valid/test steps are similar"""
        x, label = batch
        out = self(x)
        loss = self.loss(out, label)

        return loss,out.detach()