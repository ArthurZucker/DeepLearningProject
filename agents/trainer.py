import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, RichProgressBar, LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from utils.agent_utils import get_datamodule, get_net
from utils.callbacks import LogBarlowPredictionsCallback, LogBarlowCCMatrixCallback

from agents.BaseTrainer import BaseTrainer

class trainer(BaseTrainer):
    def __init__(self, config, run):
        super().__init__(config, run)

    def run(self):    

        super().run()

        # ------------------------
        # 3 INIT TRAINER
        # ------------------------
        # trainer = pl.Trainer.from_argparse_args(self.config)

        trainer = pl.Trainer(
            logger=self.wb_run,  # W&B integration
            callbacks=[
                ModelCheckpoint(monitor="val/loss", mode="min", verbose=True),  # our model checkpoint callback
                RichProgressBar(),
                LearningRateMonitor(),
                LogBarlowCCMatrixCallback(1)
                # LogBarlowPredictionsCallback() FIXME memory error had to remove it
            ],  # logging of sample predictions
            gpus=self.config.gpu,  # use all available GPU's
            max_epochs=self.config.max_epochs,  # number of epochs
            precision=self.config.precision,  # train in half precision
            accelerator="auto",
            check_val_every_n_epoch=self.config.val_freq,
            fast_dev_run=self.config.dev_run,
            # accumulate_grad_batches=self.config.accumulate_size,
            log_every_n_steps=1
            # detect_anomaly = True,
        )
        trainer.logger = self.wb_run
        trainer.fit(self.model, datamodule=self.datamodule)