from dataclasses import dataclass
from typing import List, ClassVar
import numpy as np
from simple_parsing.helpers import list_field
import random 
import torch
import simple_parsing
import os
"""Dataclass allows to have arguments and easily use wandb's weep module.

Each arguments can be preceded by a comment, which will describe it when you call 'main.pu --help
An example of every datatype is provided. Some of the available arguments are also available.
Most notably, the agent, dataset, optimizer and loss can all be specified and automatically parsed
"""


@dataclass
class Hparams:
    """Hyperparameters of Your Model"""
    # data directory
    data_dir = './data'
    # learning rate 
    lr : float = 3e-4
    # batch size 
    batch_size = 128
    # number of channels to use for projection
    bt_proj_channels: int = 2048 # TODO run a sweep for that?
    # backbone to use. Should match a specific format to be defined later
    encoder: str = "resnet50"
    #method 
    method : str= "barlow"
    


@dataclass
class DatasetParams:
    """Dataset Parameters"""

    default_root: ClassVar[str] = "/dataset"  # the default root directory to use.

    dataset: str = "CIFAR10"  # laptop,pistol
    """ dataset name: only [cifar10] for now """

    root_dir: str = default_root  # dataset root directory


@dataclass
class OptimizerParams:
    """Optimization parameters"""

    optimizer: str = "adam"  # Optimizer (adam, rmsprop)
    lr: float = 0.0001  # learning rate, default=0.0002
    lr_sched_type: str = "step"  # Learning rate scheduler type.
    z_lr_sched_step: int = 100000  # Learning rate schedule for z.
    lr_iter: int = 10000  # Learning rate operation iterations
    normal_lr_sched_step: int = 100000  # Learning rate schedule for normal.
    beta1: float = 0.0  # beta1 for adam. default=0.5
    batchSize: int = 4  # input batch size

        
@dataclass
class BarlwoConfig:
    """Hyperparameters specific to Barlow Twin Model.
    Used when the `arch` option is set to "Barlow" in the hparams
    """
    # Image size, assumes square images
    img_size: int = 32
    # number of crops/global_crops
    n_crops: int = 8
    # number of global crops 
    n_global_crops: int = 2
    # scale range of the global crops 
    global_crops_scale: List[int]   = list_field(0.5,1)
    local_crops_scale:  List[float] = list_field(0.08, 0.5)
    # lambda coefficient used for FIXME ??????????
    lmbda: float = 0.05

@dataclass
class DinoConfig:
    """Hyperparameters specific to the DINO Model.
    Used when the `arch` option is set to "Barlow" in the hparams
    """
    pass



@dataclass
class Parameters:
    """base options."""

    # Dataset parameters.
    # dataset: DatasetParams = DatasetParams()
    # Set of parameters related to the optimizer.
    # optimizer: OptimizerParams = OptimizerParams()
    # GAN Settings
    hparams: Hparams = Hparams()

    def __post_init__(self):
        """Post-initialization code"""
        # Mostly used to set some values based on the chosen hyper parameters
        # since we will use different models, backbones and datamodules

        # Set render number of channels
        if self.hparams.method == "barlow":
            self.network_param : BarlwoConfig = BarlwoConfig()  # TODO later we might need to do something
            
        # Set random seed
        if self.hparams.seed_everything is None:
            self.hparams.seed_everything = random.randint(1, 10000)
        print("Random Seed: ", self.hparams.seed_everything)
        random.seed(self.hparams.seed_everything)
        torch.manual_seed(self.hparams.seed_everything)
        if not self.hparams.gpu != 0:
            torch.cuda.manual_seed_all(self.hparams.seed_everything)

    @classmethod
    def parse(cls):
        parser = simple_parsing.ArgumentParser()
        parser.add_arguments(cls, dest="parameters")
        args = parser.parse_args()
        instance: Parameters = args.parameters
        return instance
