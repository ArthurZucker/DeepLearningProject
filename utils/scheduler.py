import numpy as np
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim import Optimizer


def cosine_scheduler(base_value, final_value, max_epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(max_epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == max_epochs * niter_per_ep
    return schedule

    
class Cosine_Scheduler(_LRScheduler):
    def __init__(self, optimizer:Optimizer, cosine_scheduler_array, last_epoch=-1):
        """
        cosine_scheduler is an array of float coefficient corresponding to the learnng rate
        """
        self.cosine_scheduler_array = cosine_scheduler_array
        self.iteration = 0
        super().__init__(optimizer, last_epoch)
        
        

        
    def get_lr(self):
        lr = [self.cosine_scheduler_array[self.iteration] for i in range(len(self.optimizer.param_groups))]
        self.iteration += 1
        return lr