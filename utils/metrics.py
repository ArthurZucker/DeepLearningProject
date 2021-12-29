import torchmetrics
import wandb
import numpy as np
import torch

"""
https://torchmetrics.readthedocs.io/en/stable/references/modules.html#base-class MODULE METRICS
"""

class MetricsModule():

    def __init__(self, n_classes=None) -> None:
        self.metric = torchmetrics.Accuracy(num_classes=n_classes,average="weighted",compute_on_step=False).cuda()

    def update_metrics(self, x, y):
        preds = torch.argmax(x, dim=1)
        self.metric(preds, y)

    def log_metrics(self, name):
        metric = self.metric.compute()
        wandb.log({f"{name}/Accuracy": metric})
        # Reseting internal state such that metric ready for new data
        self.metric.reset()
        self.metric.cuda()