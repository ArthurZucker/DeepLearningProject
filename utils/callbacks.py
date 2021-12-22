import wandb
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import WandbLogger
import numpy as np


class LogPredictionsCallback(Callback):
    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        pass
