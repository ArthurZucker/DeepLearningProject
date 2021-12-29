import os
import numpy as np

import torch
import torch.nn as nn
import torchmetrics # TODO use later

from pytorch_lightning import LightningModule

from torch.nn import functional as F
from torch.optim import Adam

from models.losses.dino_loss import DinoLoss
class DINO(LightningModule):

    def __init__(self, config, student_backbone, teacher_backbone, student_head, teacher_head, dino_loss):
        '''method used to define our model parameters'''
        super().__init__()

        # optimizer parameters
        self.lr = config.lr

        self.n_global_crops = config.n_global_crops

        # save hyper-parameters to self.hparams (auto-logged by W&B)
        # self.save_hyperparameters()


        # get backbone models and adapt them to the self-supervised task
        self.head_in_features = 0
        self.student_backbone = student_backbone
        self.teacher_backbone = teacher_backbone
        
        self.in_features = list(self.encoder.children())[-1].in_features
        name_classif = list(self.encoder.named_children())[-1][0]
        self.student_backbone._modules[name_classif] = self.teacher_backbone._modules[name_classif] = nn.Identity()
        # self.teacher_backbone._modules[name_classif]  = nn.Identity() ^^^^^^^^^ this should also do the same 
        
        # Make Projector/Head (default: 3-layers)
        self.proj_channels = config.dino_proj_channels
        self.out_channels = config.dino_out_channels
        self.proj_layers = config.dino_proj_layers
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
        
        self.dino_loss = dino_loss

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
        return Adam(self.parameters(), lr=self.lr)
    
    def _get_loss(self, batch):
        '''convenience function since train/valid/test steps are similar'''
        student_out, teacher_out = self(batch)
        
        loss = self.dino_loss(student_out, teacher_out)

        return loss
    
    
    
