import pytorch_lightning as pl
import wandb
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
)

from utils.callbacks import (
    LogAttentionMapsCallback,
    LogBarlowPredictionsCallback,
    LogBarlowCCMatrixCallback,
    LogDinoImagesCallback,
    LogDinowCCMatrixCallback,
    LogMetricsCallBack,
)

from agents.BaseTrainer import BaseTrainer


class trainer(BaseTrainer):
    def __init__(self, config, run):
        super().__init__(config, run)

    def run(self):
        super().run()
        trainer = pl.Trainer(
            logger=self.wb_run,  # W&B integration
            callbacks=self.get_callbacks(),
            gpus=self.config.gpu,  # use all available GPU's
            max_epochs=self.config.max_epochs,  # number of epochs
            precision=self.config.precision,  # train in half precision
            accelerator="auto",
            check_val_every_n_epoch=self.config.val_freq,
            fast_dev_run=self.config.dev_run,
            # accumulate_grad_batches=self.config.accumulate_size,
            log_every_n_steps=1,
        )
        trainer.logger = self.wb_run
        trainer.fit(self.model, datamodule=self.datamodule)

    def get_callbacks(self):

        callbacks = [RichProgressBar(), LearningRateMonitor()]
        
        if "BarlowTwins" == self.config.arch :
            callbacks += [
                LogBarlowPredictionsCallback(self.config.log_pred_freq),
                LogBarlowCCMatrixCallback(self.config.log_ccM_freq),
            ]

        elif self.config.arch == "Dino" or self.config.arch == "DinoTwins":
            callbacks += [LogDinoImagesCallback(self.config.log_pred_freq)]

        if self.config.arch == "DinoTwins":
            callbacks += [LogDinowCCMatrixCallback(self.config.log_dino_freq)]

        if self.encoder == "vit":
            callbacks += [
                LogAttentionMapsCallback(
                    self.config.attention_threshold, self.config.nb_attention
                )
            ]

        if "FT" in self.config.arch:
            callbacks += [LogMetricsCallBack()]
            monitor = "val/accuracy"
            mode = "max"
        else:
            monitor = "val/loss"
            mode = "min"
        wandb.define_metric(monitor, summary=mode)
        if "Dino" in self.config.arch:
            save_top_k = -1
            every_n_epochs = 20
        else:
            save_top_k = 5
            every_n_epochs = 1

        if self.config.testing:  # don't need to save if we are just testing
            save_top_k = 0

        callbacks += [
            ModelCheckpoint(
                monitor=monitor,
                mode=mode,
                filename="{epoch:02d}-{val/loss:.2f}",
                verbose=True,
                dirpath=self.config.weights_path + f"/{str(wandb.run.name)}",
                save_top_k=save_top_k,
                every_n_epochs=every_n_epochs,
            )
        ]  # our model checkpoint callback

        return callbacks
