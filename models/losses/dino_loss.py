import torch
import torch.nn as nn
from torch.nn import functional as F

import numpy as np

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