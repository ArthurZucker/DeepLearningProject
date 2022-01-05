import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    RichProgressBar,
    LearningRateMonitor,
)
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from utils.callbacks import (
    LogBarlowPredictionsCallback,
    LogBarlowCCMatrixCallback,
    LogMetricsCallBack,
    LogDinoImagesCallback,
    LogDinoDistribCallback,
    LogDinowCCMatrixCallback,
    LogAttentionMapsCallback
)

from agents.BaseTrainer import BaseTrainer


class trainer(BaseTrainer):
    def __init__(self, config, run):
        super().__init__(config, run)

    def run(self):
        super().run()

        trainer = pl.Trainer(
            logger=self.wb_run,  # W&B integration
            callbacks=[
                #  ModelCheckpoint(
                #      monitor="val/loss",
                #     mode="min",
                #    filename=str(self.config.dataset) + "-{epoch:02d}-{val_loss:.2f}",
                #   every_n_epochs=20,
                #  verbose=True,
                #),  # our model checkpoint callback
                RichProgressBar(),
                LearningRateMonitor(),
                LogDinoImagesCallback(self.config.log_pred_freq),
                LogDinoDistribCallback(self.config.log_dino_freq),
                LogAttentionMapsCallback(self.config.attention_threshold,self.config.nb_attention),
                # LogMetricsCallBack(),
                # LogBarlowPredictionsCallback(self.config.log_pred_freq) , # FIXME only add these if we are using barlow
                # LogBarlowCCMatrixCallback(self.config.log_ccM_freq) # FIXME memory error had to remove it
                LogDinowCCMatrixCallback(self.config.log_dino_freq)
            ],  # logging of sample predictions
            gpus=self.config.gpu,  # use all available GPU's
            max_epochs=self.config.max_epochs,  # number of epochs
            precision=self.config.precision,  # train in half precision
            accelerator="auto",
            check_val_every_n_epoch=self.config.val_freq,
            fast_dev_run=self.config.dev_run,
            # accumulate_grad_batches=self.config.accumulate_size,
            log_every_n_steps=1,
            default_root_dir=f"{self.config.weights_path}/{self.wb_run.name}",
            # limit_train_batches=4,
            # detect_anomaly = True,
        )
        trainer.logger = self.wb_run
        trainer.fit(self.model, datamodule=self.datamodule)
