import os

import numpy as np
import torch
import torch.nn as nn
import torchmetrics  # TODO use later
from pytorch_lightning import LightningModule
from torch.nn import functional as F
from torch.optim import Adam
from utils.agent_utils import get_net

from models.losses.dino_loss import DinoLoss


class DINO(LightningModule):

    def __init__(self, network_param,optim_param = None):
        '''method used to define our model parameters'''
        super().__init__()

        self.n_global_crops = network_param.n_global_crops

        self.momentum_schedule = network_param.momentum_schedule
        #self.loss = BarlowTwinsLoss
        self.loss = DinoLoss(network_param.lmbda)
        # optimizer parameters
        self.optim_param = optim_param


        # get backbone models and adapt them to the self-supervised task
        self.head_in_features = 0
        self.student_backbone = get_net(
            network_param.student_backbone,network_param
        )
        self.teacher_backbone = get_net(
            network_param.teacher_backbone,network_param
        )
        
        self.in_features = list(self.encoder.children())[-1].in_features
        name_classif = list(self.encoder.named_children())[-1][0]
        self.student_backbone._modules[name_classif] = self.teacher_backbone._modules[name_classif] = nn.Identity()
        # self.teacher_backbone._modules[name_classif]  = nn.Identity() ^^^^^^^^^ this should also do the same 
        
        # Make Projector/Head (default: 3-layers)
        self.proj_channels = network_param.dino_proj_channels
        self.out_channels = network_param.dino_out_channels
        self.proj_layers = network_param.dino_proj_layers
        proj_layers = []
        for i in range(self.proj_layers):
            # First Layer
            if i == 0:
                proj_layers.append(
                    nn.Linear(self.head_in_features, self.proj_channels, bias=False)
                )
            #Last Layer
            elif i == self.proj_channels - 1:
                proj_layers.append(
                    nn.Linear(self.proj_channels, self.out_channels, bias=False)
                )
            # Middle Layer(s)
            else:
                proj_layers.append(
                    nn.Linear(self.proj_channels, self.proj_channels, bias=False)
                )
            if i < 2:
                proj_layers.append(nn.GELU())

        #Make head (To be implemented properly after we make the head class)
        self.student_head = nn.Sequential(*proj_layers.clone())
        self.teacher_head = nn.Sequential(*proj_layers.clone())
        

    def forward(self, crops):
        
        #Student forward pass
        full_st_output = torch.empty(0).to(crops[0].device)
        for x in crops:
            out = self.student_backbone(x)
            full_st_output = torch.cat((full_st_output, out))
            
        #Teacher forward pass
        full_teacher_output = torch.empty(0).to(crops[0].device)
        for x in crops[:2]:
            out = self.teacher_backbone(x)
            full_teacher_output = torch.cat((full_teacher_output, out))
        
        #Run head on concatenated feature maps
        return self.student_head(full_st_output), self.teacher_head(full_teacher_output)

    def training_step(self, batch, batch_idx):
        '''needs to return a loss from a single batch'''
        
        #get only the global crops for the teacher
        loss = self._get_loss(batch)
        

        # Log loss and metric
        self.log('train_loss', loss)

        return loss

    def validation_step(self, batch, batch_idx):
        '''used for logging metrics'''
        loss = self._get_loss(batch)

        # Log loss and metric
        self.log('val_loss', loss)

        return loss

    def test_step(self, batch, batch_idx):
        '''used for logging metrics'''
        loss = self._get_loss(batch)

        # Log loss and metric
        self.log('test_loss', loss)
    
    def configure_optimizers(self):
        '''defines model optimizer'''
        optimizer = getattr(torch.optim,self.optim_param.optimizer)
        optimizer = optimizer(self.parameters(), lr=self.optim_param.lr)
        # scheduler = LinearWarmupCosineAnnealingLR(
        #     optimizer, warmup_epochs=5, max_epochs=40
        # )
        return optimizer
    
    def _get_loss(self, batch):
        '''convenience function since train/valid/test steps are similar'''
        student_out, teacher_out = self(batch)
        
        loss = self.dino_loss(student_out, teacher_out)

        return loss
    
    def on_epoch_end(self) -> None:

        # EMA update for the teacher
        with torch.no_grad():
            m = self.momentum_schedule[self.epoch]  # momentum parameter
            for param_q, param_k in zip(self.student_backbone.parameters(), self.teacher_backbone.parameters()):
                param_k.mul_(m).add_((1 - m) * param_q.detach())
            
            for param_q, param_k in zip(self.student_head.parameters(), self.teacher_head.parameters()):
                param_k.mul_(m).add_((1 - m) * param_q.detach())
        
        