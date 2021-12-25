import os

import torch
import torch.nn as nn
import torchmetrics # TODO use later

from pytorch_lightning import LightningModule

from torch.nn import functional as F
from torch.optim import Adam

class DINO(LightningModule):

    def __init__(self, config, student_backbone, teacher_backbone, student_head, teacher_head):
        '''method used to define our model parameters'''
        super().__init__()

        # optimizer parameters
        self.lr = config.lr

        # save hyper-parameters to self.hparams (auto-logged by W&B)
        # self.save_hyperparameters()


        # get backbone models and adapt them to the self-supervised task
        self.head_in_features = 0
        self.student_backbone = student_backbone
        self.teacher_backbone = teacher_backbone
        try:
            self.head_in_features = list(self.student_backbone.children())[-1].in_features
            last_layer_name = list(self.student_backbone.named_children())[-1][0]

            if last_layer_name == 'head':
                self.student_backbone.head = nn.Identity()
                self.teacher_backbone.head = nn.Identity()
            else:
                self.student_backbone.fc = nn.Identity()
                self.teacher_backbone.fc = nn.Identity()
        except:
            print("student_backbone should be a torchvision resnet model or timm's VIT")

        
        #Make head (To be implemented properly after we make the head class)
        self.student_head = student_head
        self.teacher_head = teacher_head

    def forward(self, student_crops, teacher_crops):
        
        #Student forward pass
        full_st_output = torch.empty(0).to(student_crops[0].device)
        for x in student_crops:
            out = self.student_backbone(x)
            full_st_output = torch.cat((full_st_output, out))
            
        #Teacher forward pass
        full_teacher_output = torch.empty(0).to(teacher_crops[0].device)
        for x in teacher_crops:
            out = self.teacher_backbone(x)
            full_teacher_output = torch.cat((full_teacher_output, out))
        
        #Run head on concatenated features
        return self.student_head(full_st_output), self.teacher_head(full_teacher_output)
    
    #######################################################################################################
    #NOTHING BELLOW HAS BEEN IMPLEMENTED YET
    #######################################################################################################

    def training_step(self, batch, batch_idx):
        '''needs to return a loss from a single batch'''
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
        x1, x2 = batch
        z1, z2 = self(x1, x2)
        
        loss = bt_loss(z1, z2, self.lmbda)

        return loss