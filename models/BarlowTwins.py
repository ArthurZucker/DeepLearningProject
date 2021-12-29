import torch.nn as nn
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from pytorch_lightning import LightningModule
from torch.nn import functional as F
from torch.optim import Adam
from utils.agent_utils import get_net

from models.losses.barlow_twins import BarlowTwinsLoss
from models.optimizers.lars import LARS

class BarlowTwins(LightningModule):
    def __init__(self, config):
        """method used to define our model parameters"""
        super().__init__()

        self.loss = BarlowTwinsLoss
        # optimizer parameters
        self.lr = config.lr
        self.log_pred_freq = config.log_pred_freq
        # save hyper-parameters to self.hparams (auto-logged by W&B)
        # self.save_hyperparameters()

        # Loss parameter
        self.lmbda = config.lmbda

        # get backbone model and adapt it to the task
        self.encoder = get_net(
            config, config.encoder
        )  # TODO add encoder name to the hparams, use getnet() to get the encoder
        self.in_features = list(self.encoder.children())[-1].in_features
        name_classif = list(self.encoder.named_children())[-1][0]
        self.encoder._modules[name_classif] = nn.Identity()

        # Make Projector (3-layers)
        self.proj_channels = config.bt_proj_channels
        proj_layers = []

        for i in range(3):
            if i == 0:
                proj_layers.append(
                    nn.Linear(self.in_features, self.proj_channels, bias=False)
                )
            else:
                proj_layers.append(
                    nn.Linear(self.proj_channels, self.proj_channels, bias=False)
                )
            if i < 2:
                proj_layers.append(nn.BatchNorm1d(self.proj_channels))
                proj_layers.append(nn.ReLU(inplace=True))

        self.proj = nn.Sequential(*proj_layers)

    def forward(self, x1, x2):
        # Feeding the data through the encoder and projector
        z1 = self.proj(self.encoder(x1))
        z2 = self.proj(self.encoder(x2))

        return z1, z2

    def training_step(self, batch, batch_idx):
        """needs to return a loss from a single batch"""
        loss = self._get_loss(batch)

        # Log loss and metric
        self.log("train/loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        """used for logging metrics"""
        loss = self._get_loss(batch)

        # Log loss and metric
        self.log("val/loss", loss)

        return loss

    def configure_optimizers(self):
        """defines model optimizer"""
        optimizer = LARS(self.parameters(), lr=self.lr)
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer, warmup_epochs=10, max_epochs=40
        )
        return [[optimizer], [scheduler]]

    def _get_loss(self, batch):
        """convenience function since train/valid/test steps are similar"""
        x1, x2 = batch
        z1, z2 = self(x1, x2)

        loss = self.loss(z1, z2, self.lmbda)

        return loss
