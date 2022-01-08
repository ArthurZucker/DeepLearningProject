import os

import numpy as np
import torch
import torch.nn as nn
import torchmetrics  # TODO use later
from pytorch_lightning import LightningModule
from torch.nn import functional as F
from torch.optim import Adam
from utils.agent_utils import get_net

from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from models.losses.dino_twins_loss import DinowTwinsLoss
from models.DinoTwins import DinowTwins
from utils.scheduler import cosine_scheduler

from models.custom_layers.l2norm import L2Norm

class DinowTwinsFT(LightningModule):

    def __init__(self, network_param, optim_param = None):
        '''method used to define our model parameters'''
        super().__init__()
        # Network parameters 
        self.num_cat = network_param.num_cat
        
        # optimizer/scheduler parameters
        self.optim_param = optim_param
        self.lr = self.optim_param.lr
        # initialize loss TODO get max epochs from the hparams config directly instead of model specific params
        self.loss = nn.CrossEntropyLoss()

        if network_param.backbone_parameters is not None:
            self.patch_size = network_param.backbone_parameters["patch_size"]
            
        self.pretrained_dinow_twin = DinowTwins(network_param,optim_param) #BarlowTwins(network_param)
        if network_param.weight_checkpoint is not None: 
            print(f"Loaded chekpoint from {network_param.weight_checkpoint}")
            self.pretrained_dinow_twin.load_state_dict(torch.load(network_param.weight_checkpoint)["state_dict"])
        # @TODO solve the issue, VIT already has a lat layer embedded, resnet should have one too
        self.head_out_features = self.pretrained_dinow_twin.head_in_features
        self.pretrained_dinow_twin.requires_grad_(False)        
        self.linear = nn.Linear(self.head_out_features, self.num_cat)

    def forward(self, x):
        # Feed the data through pretrained barlow twins and prediciton layer
        out = self.pretrained_dinow_twin.student_backbone(x)
        # out = self.pretrained_dinow_twin.student_head(out)
        # out = F.softmax(self.pretrained_dinow_twin.student_head(out))
        # out = self.pretrained_dinow_twin.bt_proj(out)
        # out = self.linear(torch.cat((out1,out2),1))
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

        return loss, out.detach()