import wandb
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import WandbLogger
import numpy as np

class LogBarlowPredictionsCallback(Callback):

    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        """Called when the training batch ends."""

        # `outputs` comes from `LightningModule.validation_step`
        # which corresponds to our model predictions in this case

        # Let's log 20 sample image predictions from first batch
        if batch_idx == 0:
            self.log_images("train", batch, 5, outputs)

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
            
        wandb.log({"x1": samples1})
        wandb.log({"x2":samples2}) #TODO merge graphs
