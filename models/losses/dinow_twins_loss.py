import torch
import torch.nn as nn
from torch.nn import functional as F
from models.losses.barlow_twins import CrossCorrelationMatrixLoss

import numpy as np


import torch
import torch.nn as nn
from torch.nn import functional as F
from models.losses.barlow_twins import CrossCorrelationMatrixLoss

import numpy as np

class DinowTwinsLoss2(nn.Module):
    def __init__(self, network_param, max_epochs): 
        super().__init__()
        
        self.n_crops = network_param.n_crops
        self.n_global_crops = network_param.n_global_crops
        self.center_momentum = network_param.center_momentum
        self.student_temp = network_param.student_temp
        self.teacher_temp = network_param.teacher_temp
        #the centering operation requires tu update a buffer of centers
        self.register_buffer("center", torch.zeros(1, network_param.out_channels))
        
        # Without a warmup on the teacher temperature, training becomes unstable
        #To be reviewed and fixed
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(network_param.warmup_teacher_temp,
                        self.teacher_temp, network_param.warmup_teacher_temp_epochs),
            np.ones(max_epochs - network_param.warmup_teacher_temp_epochs) * self.teacher_temp
        ))
        
        #Get the BarlowTwins loss and scaling parameter
        self.bt_loss = CrossCorrelationMatrixLoss(network_param.lmbda)
        self.bt_beta = network_param.bt_beta
        

    def forward(self, student_out, teacher_out, epoch):
        #keep variable for centering
        teacher_out_center = teacher_out.clone().detach()
        
        #Perform sharpening on student's ouput 
        student_out =  student_out / self.student_temp
    
        #Perform centering and sharpening on the teachers' output
        teacher_temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax((teacher_out - self.center)/ teacher_temp, dim=-1).detach()
        
        #Get each image's output separately
        # barlow_out = barlow_out.chunk(self.n_crops)
        student_out = student_out.chunk(self.n_crops)
        teacher_out = teacher_out.chunk(self.n_global_crops)
        self.teacher_distrib = np.array([torch.softmax(teacher_out[i], dim =-1).detach().cpu().numpy() for i in range(len(teacher_out))])
        self.student_distrib = np.array([torch.softmax(student_out[i], dim =-1).detach().cpu().numpy() for i in range(len(student_out))])
        #Here we have one output per image 
        dino_loss = 0
        num_dino_losses = 0
        for t_idx, t_out in enumerate(teacher_out):
            for s_idx in range(len(student_out)):
                if t_idx == s_idx:
                    #We don't compute the loss when the image is the same for s and t
                    continue
                
                #Sum is over features dimension
                loss = torch.sum(-t_out * F.log_softmax(student_out[s_idx], dim=-1), dim=-1)
                num_dino_losses += 1
                #Mean is over batch dimension
                dino_loss += loss.mean()

        #Get BarlowTwins loss
        loss_bt = 0 
        num_bt_losses = 0
        for z1_idx in range(len(student_out)): 
            for z2_idx in range(len(student_out)):
                if z1_idx == z2_idx:
                    continue
                loss_bt += self.bt_beta * self.bt_loss(student_out[z1_idx], student_out[z2_idx])
                num_bt_losses += 1

        
        #update the center
        self.update_center(teacher_out_center)
        return dino_loss/num_dino_losses, loss_bt/num_bt_losses
                
                
    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        Copy pasted from DINO github. Commented part are only required with distributed gpu
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        #dist.all_reduce(batch_center)
        batch_center = batch_center / (len(teacher_output)) #* dist.get_world_size())

        # ema update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)
class DinowTwinsLoss(nn.Module):
    def __init__(self, network_param, max_epochs): 
        super().__init__()
        
        self.n_crops = network_param.n_crops
        self.n_global_crops = network_param.n_global_crops
        self.center_momentum = network_param.center_momentum
        self.student_temp = network_param.student_temp
        self.teacher_temp = network_param.teacher_temp
        #the centering operation requires tu update a buffer of centers
        self.register_buffer("center", torch.zeros(1, network_param.out_channels))
        
        # Without a warmup on the teacher temperature, training becomes unstable
        #To be reviewed and fixed
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(network_param.warmup_teacher_temp,
                        self.teacher_temp, network_param.warmup_teacher_temp_epochs),
            np.ones(max_epochs - network_param.warmup_teacher_temp_epochs) * self.teacher_temp
        ))
        
        #Get the BarlowTwins loss and scaling parameter
        self.bt_loss = CrossCorrelationMatrixLoss(network_param.lmbda)
        self.bt_beta = network_param.bt_beta
        

    def forward(self, student_out, teacher_out, barlow_out, epoch):
        #keep variable for centering
        teacher_out_center = teacher_out.clone().detach()
        
        #Perform sharpening on student's ouput 
        student_out =  student_out / self.student_temp
    
        #Perform centering and sharpening on the teachers' output
        teacher_temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax((teacher_out - self.center)/ teacher_temp, dim=-1).detach()
        
        #Get each image's output separately
        barlow_out = barlow_out.chunk(self.n_crops)
        student_out = student_out.chunk(self.n_crops)
        teacher_out = teacher_out.chunk(self.n_global_crops)
        self.teacher_distrib = np.array([torch.softmax(teacher_out[i], dim =-1).detach().cpu().numpy() for i in range(len(teacher_out))])
        self.student_distrib = np.array([torch.softmax(student_out[i], dim =-1).detach().cpu().numpy() for i in range(len(student_out))])
        #Here we have one output per image 
        dino_loss = 0
        num_dino_losses = 0
        for t_idx, t_out in enumerate(teacher_out):
            for s_idx in range(len(student_out)):
                if t_idx == s_idx:
                    #We don't compute the loss when the image is the same for s and t
                    continue
                
                #Sum is over features dimension
                loss = torch.sum(-t_out * F.log_softmax(student_out[s_idx], dim=-1), dim=-1)
                num_dino_losses += 1
                #Mean is over batch dimension
                dino_loss += loss.mean()

        #Get BarlowTwins loss
        loss_bt = 0 
        num_bt_losses = 0
        for z1_idx in range(len(barlow_out)): 
            for z2_idx in range(len(barlow_out)):
                if z1_idx == z2_idx:
                    continue
                loss_bt += self.bt_beta * self.bt_loss(barlow_out[z1_idx], barlow_out[z2_idx])
                num_bt_losses += 1

        
        #update the center
        self.update_center(teacher_out_center)
        return dino_loss/num_dino_losses, loss_bt/num_bt_losses
                
                
    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        Copy pasted from DINO github. Commented part are only required with distributed gpu
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        #dist.all_reduce(batch_center)
        batch_center = batch_center / (len(teacher_output)) #* dist.get_world_size())

        # ema update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)