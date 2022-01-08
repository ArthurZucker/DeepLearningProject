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
from models.losses.dino_loss import DinoLoss
from utils.scheduler import cosine_scheduler, Cosine_Scheduler

from models.custom_layers.l2norm import L2Norm


class Dino(LightningModule):
    def __init__(self, network_param, optim_param=None):
        """method used to define our model parameters"""
        super().__init__()

        self.n_global_crops = network_param.n_global_crops

        # initialize current epoch/iteration
        self.curr_iteration = 0

        if network_param.backbone_parameters is not None:
            self.patch_size = network_param.backbone_parameters["patch_size"]
        
        # optimizer/scheduler parameters
        self.optim_param = optim_param

        # initialize momentum scheduler. This is overwritten by the configure optimizers method
        self.momentum_schedule = cosine_scheduler(**optim_param.scheduler_parameters)

        # initialize loss TODO get max epochs from the hparams config directly instead of model specific params
        self.loss = DinoLoss(network_param, optim_param.max_epochs)

        if network_param.backbone_parameters is not None:
            self.patch_size = network_param.backbone_parameters["patch_size"]
        # get backbone models
        self.student_backbone = get_net(
            network_param.backbone, network_param.backbone_parameters
        )
        self.teacher_backbone = get_net(
            network_param.backbone, network_param.backbone_parameters
        )

        # Adapt models to the self-supervised task
        self.head_in_features = list(self.student_backbone.modules())[-1].in_features
        name_classif = list(self.student_backbone.named_children())[-1][0]
        self.student_backbone._modules[
            name_classif
        ] = (
            nn.Identity()
        )  # self.teacher_backbone._modules[name_classif] = nn.Identity()
        self.teacher_backbone._modules[
            name_classif
        ] = nn.Identity()  

        # Make Projector/Head (default: 3-layers)
        self.proj_dim = network_param.proj_dim
        self.out_dim = network_param.out_dim
        self.proj_layers_num = network_param.proj_layers
        self.bottleneck_dim = network_param.bottleneck_dim

        # Make heads with same architecture on both networks
        self.student_head = self._get_head()
        self.teacher_head = self._get_head()

        # teacher does not require gradient
        self.teacher_backbone.requires_grad_(False)
        self.teacher_head.requires_grad_(False)

        self.teacher_backbone.load_state_dict(self.student_backbone.state_dict())
        self.teacher_head.load_state_dict(self.student_head.state_dict())

    def forward(self, crops):

        # Student forward pass
        full_st_output = torch.empty(0).to(crops[0].device)
        for x in crops:
            out = self.student_backbone(x)
            full_st_output = torch.cat((full_st_output, out))

        # Teacher forward pass
        full_teacher_output = torch.empty(0).to(crops[0].device)
        for x in crops[:2]:
            out = self.teacher_backbone(x)
            full_teacher_output = torch.cat((full_teacher_output, out))

        # Run head on concatenated feature maps
        return self.student_head(full_st_output), self.teacher_head(full_teacher_output)

    def training_step(self, batch, batch_idx):
        """needs to return a loss from a single batch"""
        # update iteration parameter
        self.curr_iteration += 1

        loss = self._get_loss(batch)

        # EMA update for the teacher
        with torch.no_grad():
            # update momentum according to schedule and curr_iteration
            m = self.momentum_schedule[self.curr_iteration]
            # update all the teacher's parameters
            for param_q, param_k in zip(
                self.student_backbone.parameters(), self.teacher_backbone.parameters()
            ):
                param_k.mul_(m).add_((1 - m) * param_q.detach())

            for param_q, param_k in zip(
                self.student_head.parameters(), self.teacher_head.parameters()
            ):
                param_k.mul_(m).add_((1 - m) * param_q.detach())

        # Log loss and metric
        self.log("train/loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        """used for logging metrics"""
        loss = self._get_loss(batch)

        # Log loss and metric
        self.log("val/loss", loss)

        return loss

    def test_step(self, batch, batch_idx):
        """used for logging metrics"""
        loss = self._get_loss(batch)

        # Log loss and metric
        self.log("test/loss", loss)

    def configure_optimizers(self):
        """defines model optimizer"""
        optimizer = getattr(torch.optim, self.optim_param.optimizer)

        optimizer = optimizer(
            self.parameters(),
            lr=self.optim_param.lr * self.trainer.datamodule.batch_size / 256,
        )

        self.optim_param.scheduler_parameters["max_epochs"] = self.trainer.max_epochs
        self.optim_param.scheduler_parameters["niter_per_ep"] = len(
            self.trainer.datamodule.train_dataloader()
        )
        self.momentum_schedule = cosine_scheduler(
            **self.optim_param.scheduler_parameters
        )
        #self.lr_scheduler_array = cosine_scheduler(
        #    self.optim_param.lr * self.trainer.datamodule.batch_size / 256,
        #    self.optim_param.min_lr,
        #    self.optim_param.max_epochs,
        #    len(self.trainer.datamodule.train_dataloader()),
        #    self.optim_param.warmup_epochs,
        #)
        #scheduler = Cosine_Scheduler(optimizer, self.lr_scheduler_array)
        scheduler = LinearWarmupCosineAnnealingLR(
           optimizer, warmup_epochs=10, max_epochs=self.optim_param.max_epochs, 
           warmup_start_lr=0.1*(self.optim_param.lr * self.trainer.datamodule.batch_size / 256),
           eta_min = 0.1*(self.optim_param.lr * self.trainer.datamodule.batch_size / 256)
         )
        
        return [[optimizer], [scheduler]]

    def _get_loss(self, batch):
        """convenience function since train/valid/test steps are similar"""
        student_out, teacher_out = self(batch)

        loss = self.loss(student_out, teacher_out, self.current_epoch)

        return loss

    def _get_head(self):
        # first layer
        proj_layers = [nn.Linear(self.head_in_features, self.proj_dim), nn.GELU()]
        for i in range(self.proj_layers_num - 2):
            proj_layers.append(nn.Linear(self.proj_dim, self.proj_dim))
            proj_layers.append(nn.GELU())
        proj_layers += [nn.Linear(self.proj_dim, self.bottleneck_dim), nn.GELU()]
        # last layer
        proj_layers += [
            L2Norm(),
            nn.Linear(self.bottleneck_dim, self.out_dim, bias=False),
        ]
        return nn.Sequential(*proj_layers)
