import torch.nn as nn
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from pytorch_lightning import LightningModule
from torch.nn import functional as F
import torch

from models.Dino import Dino
from utils.agent_utils import get_net

class DinoFT(LightningModule):
    def __init__(self, network_param, optim_param):
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
        self.pretrained_dino = Dino(network_param,optim_param) #BarlowTwins(network_param)
        if network_param.weight_checkpoint is not None: 
            self.pretrained_dino.load_state_dict(torch.load(network_param.weight_checkpoint)["state_dict"])
        
        self.head_out_features = list(self.pretrained_dino.student_head.children())[0].out_features
        self.pretrained_dino.requires_grad_(False)        
        self.linear = nn.Linear(2048, self.num_cat)

    def forward(self, x):
        # Feed the data through pretrained barlow twins and prediciton layer
        out = self.pretrained_dino.student_backbone(x)
        out = self.pretrained_dino.student_head(out)
        out = self.linear(out)

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

        return loss, out.detach()