import os
import numpy as np

import torch
import torch.nn as nn
import torchmetrics # TODO use later

from pytorch_lightning import LightningModule

from torch.nn import functional as F
from torch.optim import Adam

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
    
    
    
    
class DINO_Loss(nn.Module):
    def __init__(self, config): 
        super().__init__()
       
        
        self.n_crops = config.n_crops
        self.n_global_crops = config.n_global_crops
        self.center_momentum = config.center_momentum
        self.student_temp = config.student_temp
        self.teacher_temp = config.teacher_temp
        
        #the centering operation requires tu update a buffer of centers
        self.register_buffer("center", torch.zeros(1, config.out_dim))
        
        # Without a warmup on the teacher temperature, training becomes unstable
        #To be reviewed and fixed
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(config.warmup_teacher_temp,
                        self.teacher_temp, config.warmup_teacher_temp_epochs),
            np.ones(config.nepochs - config.warmup_teacher_temp_epochs) * self.teacher_temp
        ))
        
    def forward(self, student_out, teacher_out, epoch):
        #keep variable for centering
        teacher_out_center = teacher_out.clone()
        
        #Perform sharpening on student's ouput 
        student_out =  student_out / self.student_temp
    
        #Perform centering and sharpening on the teachers' output
        teacher_temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax((teacher_out - self.center)/ teacher_temp, dim=-1).detach()
        
        #Get each image's output separately
        student_out = student_out.chunk(self.n_crops)
        teacher_out = teacher_out.chunk(self.n_global_crops)
        
        #Here we have one output per image 
        total_loss = 0
        num_losses = 0
        for t_idx, t_out in enumerate(teacher_out):
            for s_idx, s_out in enumerate(student_out):
                if t_idx == s_idx:
                    #We don't compute the loss when the image is the same for s and t
                    continue
                
                #Sum is over features dimension
                loss = torch.sum(-t_out * F.log_softmax(s_out, dim=-1), dim=-1)
                num_losses += 1
                #Mean is over batch dimension
                total_loss += loss.mean()
        
        #update the center
        self.update_center(teacher_out_center)
        return total_loss/num_losses
                
                
    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        Copy pasted from DINO github. Comented part are only required with distributed gpu
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        #dist.all_reduce(batch_center)
        batch_center = batch_center / (len(teacher_output)) #* dist.get_world_size())

        # ema update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)