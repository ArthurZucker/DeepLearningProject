from seaborn.matrix import heatmap
import wandb
from pytorch_lightning.callbacks import Callback

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from utils.metrics import MetricsModule
class LogBarlowPredictionsCallback(Callback):
    def __init__(self,log_pred_freq) -> None:
        super().__init__()
        self.log_pred_freq = log_pred_freq

    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        """Called when the training batch ends."""
        # Let's log 20 sample image predictions from first batch
        if batch_idx == 0 and pl_module.current_epoch % self.log_pred_freq == 0:
            self.log_images("train", batch, 5, outputs)

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        """Called when the training batch ends."""

        # Let's log 20 sample image predictions from first batch
        if batch_idx == 0 and pl_module.current_epoch % self.log_pred_freq == 0:
            self.log_images("val", batch, 5, outputs)


    def log_images(self, name, batch, n, outputs):

        x1, x2 = batch
        image1 = x1[:n].cpu().detach().numpy()
        image2 = x2[:n].cpu().detach().numpy()

        samples1 = []
        samples2 = []
        mean = np.array([0.485, 0.456, 0.406])  # TODO this is not beautiful
        std = np.array([0.229, 0.224, 0.225])

        for i in range(n):

            bg1 = image1[i].transpose((1, 2, 0))
            bg1 = std * bg1 + mean
            bg1 = np.clip(bg1, 0, 1)


            bg2 = image2[i].transpose((1, 2, 0))
            bg2 = std * bg2 + mean
            bg2 = np.clip(bg2, 0, 1)

            samples1.append(wandb.Image(bg1))
            samples2.append(wandb.Image(bg2))
            
        wandb.log({f"{name}/x1": samples1})
        wandb.log({f"{name}/x2": samples2}) #TODO merge graphs   

        # FIXME should not be required but memory leaks were found
        del samples1,samples2,x1,x2




class LogBarlowCCMatrixCallback(Callback):
    """Logs the cross correlation matrix obtain 
    when computing the loss. This gives us an idea of 
    how the network learns. 
    TODO : when should we log ? 
    TODO : should we average over batches only? Or epoch? 
    For now, the average over the epoch will be computed
    as a moving average. 
    A hook should be registered on the loss, using a new argument in the loss 
    loss.cc_M which will be stored each time and then deleted
    
    """
    def __init__(self,log_ccM_freq) -> None:
        super().__init__()
        self.log_ccM_freq = log_ccM_freq
        self.cc_M  = None

    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        """Called when the training batch ends."""
        # Let's log 20 sample image predictions from first batch
        if self.cc_M is not None : 
            self.cc_M += (pl_module.loss.cc_M - self.cc_M)/(batch_idx+1) 
        else: 
            self.cc_M =  pl_module.loss.cc_M
        del pl_module.loss.cc_M 

        if batch_idx == 0 and pl_module.current_epoch % self.log_ccM_freq == 0:
            self.log_cc_M("train")

    def on_val_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        """Called when the training batch ends."""
        # Let's log 20 sample image predictions from first batch
        if self.cc_M is not None : 
            self.cc_M += (pl_module.loss.cc_M - self.cc_M)/(batch_idx+1) 
        else: 
            self.cc_M =  pl_module.loss.cc_M
        del pl_module.loss.cc_M 

        if batch_idx == 0:
            self.log_cc_M("val")

    def log_cc_M(self,name):
        heatmap = self.cc_M
        ax = sns.heatmap(heatmap, cmap="rainbow",cbar=False)
        plt.title(f"Cross correlation matrix")
        ax.set_axis_off()
        wandb.log({f"cc_Matrix/{name}" : (wandb.Image(plt))})
        plt.close()
        self.cc_M = None

class LogMetricsCallBack(Callback):
    def __init__(self):
        pass

    def on_fit_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:

        self.num_classes = pl_module.num_cat
        self.metrics_train= MetricsModule( self.num_classes)
        self.metrics_val = MetricsModule( self.num_classes)

    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        """Called when the train batch ends."""

        _, y = batch
        self.metrics_train.update_metrics(outputs["logits"], y)

    def on_train_epoch_end(self, trainer, pl_module):
        """Called when the train epoch ends."""

        self.metrics_train.log_metrics("train")

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        """Called when the validation batch ends."""

        _, y = batch
        self.metrics_val.update_metrics(outputs["logits"], y)

    def on_validation_epoch_end(self, trainer, pl_module):
        """Called when the validation epoch ends."""

        self.metrics_val.log_metrics("val")