from dataclasses import dataclass
from typing import Dict, List, ClassVar, Optional, Any
import numpy as np
from simple_parsing.helpers import list_field, choice, dict_field
import random

import torch
import torch.optim

import simple_parsing
import os
from os import path as osp

"""Dataclass allows to have arguments and easily use wandb's weep module.

Each arguments can be preceded by a comment, which will describe it when you call 'main.pu --help
An example of every datatype is provided. Some of the available arguments are also available.
Most notably, the agent, dataset, optimizer and loss can all be specified and automatically parsed
"""

@dataclass
class Hparams:
    """Hyperparameters of Your Model"""

    wandb_project: str = "test-deep-learning"  # name of the project
    wandb_entity: str = "dinow-twins"  # name of the wandb entity, here our team
    save_dir: str = osp.join(os.getcwd(), "wandb")  # directory to save wandb outputs
    arch: str = "DinowTwins"  # choice("BarlowTwinsFT","BarlowTwins", "Dino", "DinowTwins", default="BarlowTwins")  # training method, either Barlow, Dino, or DinowTwin
    # datamodule to use, for now we only have one dataset, CIFAR10
    datamodule: str = "DinoDataModule"
    dataset: Optional[str] = "DinoDataset"
    agent: str = "trainer"  # agent to use for training
    seed_everything: Optional[int] = None  # seed for the whole run
    input_size: tuple = (32, 32)  # resize coefficients (H,W) for classic transforms
    tune_lr: bool = False  # tune the model on first run
    tune_batch_size: bool = False  # tune the model on first run
    gpu: int = 1  # number or gpu
    precision: int = 32  # precision
    val_freq: int = 1  # validation frequency
    dev_run: bool = False  # developpment mode, only run 1 batch of train val and test
    accumulate_size: int = 512  # gradient accumulation batch size
    # maximum number of epochs
    max_epochs: int = 400
    # path to download pascal voc
    asset_path: str = osp.join(os.getcwd(), "assets")
    # log_pred_freq
    log_pred_freq: int = 10
    # log cc_M matrix frequency
    log_ccM_freq: int = 1
    # log output frrequency for dino
    log_dino_freq: int = 1
    # path to save weights
    weights_path: str = osp.join(os.getcwd(), "weights")


@dataclass
class DatasetParams:
    """Dataset Parameters"""

    # Image size, assumes square images
    num_workers: int = 20  # number of workers for dataloadersint
    input_size: tuple = (32, 32)  # image_size
    batch_size: int = 256  # batch_size
    asset_path: str = osp.join(os.getcwd(), "assets")  # path to download the dataset

    # Dino params
    # number of crops/global_crops
    n_crops: int = 5 #TODO already defined in the model.... 
    # number of global crops
    n_global_crops: int = 2
    # scale range of the crops
    global_crops_scale: List[int] = list_field(0.5, 1)
    local_crops_scale: List[float] = list_field(0.05, 0.5)

    



@dataclass
class OptimizerParams:
    """Optimization parameters"""

    optimizer: str = "Adam"  # Optimizer (adam, rmsprop)
    lr: float = 5e-4  # learning rate, default=0.0002
    lr_sched_type: str = "step"  # Learning rate scheduler type.
    z_lr_sched_step: int = 100000  # Learning rate schedule for z.
    lr_iter: int = 10000  # Learning rate operation iterations
    normal_lr_sched_step: int = 100000  # Learning rate schedule for normal.
    betas: List[float] = list_field(0.9, 0.999)  # beta1 for adam. default=(0.9, 0.999)
    scheduler_parameters: Dict[str, Any] = dict_field(
        dict(
            base_value=0.9995,
            final_value=1,
            max_epochs=0,
            niter_per_ep=0,
            warmup_epochs=0,
            start_warmup_value=0,
        )
    )


@dataclass
class BarlowConfig:
    """Hyperparameters specific to Barlow Twin Model.
    Used when the `arch` option is set to "Barlow" in the hparams
    """

    # number of channels to use for projection
    bt_proj_channels: int = 2048  # TODO run a sweep for that?
    # encoder for barlow
    encoder: str = choice("resnet50", "swinS", default="resnet50")
    # lambda coefficient used to scale the scale of the redundancy loss
    # so it doesn't overwhelm the invariance loss
    lmbda: float = 5e-3

    pretrained_encoder: bool = False

    # number of classes to use for the fine tuning task
    num_cat: int = 10
    # model checkpoint used in classification fine tuning
    weight_checkpoint: Optional[str] = osp.join(
        os.getcwd(),
        "wandb/test-deep-learning/lebgzheo/checkpoints/epoch=189-step=4749.ckpt",
    )


@dataclass
class DinoConfig:
    """Hyperparameters specific to the DINO Model.
    Used when the `arch` option is set to "Barlow" in the hparams
    """

    student_backbone: str = "resnet50" 
    teacher_backbone: str = student_backbone
    proj_layers: int = 3
    proj_channels: int = 2048
    bottleneck_dim: int = 256
    out_channels: int = 4096
    # number of crops/global_crops
    n_crops: int = 5
    # number of global crops
    n_global_crops: int = 2
    # scale range of the crops
    global_crops_scale: List[int] = list_field(0.5, 1)
    local_crops_scale: List[float] = list_field(0.05, 0.5)
    warmup_teacher_temp_epochs: int = 10  # Default 30
    student_temp: float = 0.1
    teacher_temp: float = 0.07  # Default 0.04, can be linearly increased to 0.07 but then it becomes unstable
    warmup_teacher_temp: float = (
        0.04  # would be different from techer temp if we used a warmup for this param
    )
    center_momentum: float = 0.9  # Default 0.9
    max_epochs: int = 400  # This is redundant with the hparms max_epochs
    # number of classes to use for the fine tuning task
    num_cat: int = 10

    weight_checkpoint: Optional[str] = osp.join(
        os.getcwd(),
        "/home/arthur/Work/MVA-S1/DeepLearning/DeepLearningProject/weights/dino/epoch=191-step=37631.ckpt"
        #"/home/arthur/Work/MVA-S1/DeepLearning/DeepLearningProject/weights/dino/epoch=386-step=75851.ckpt"
        #"wandb/test-deep-learning/15nz03bf/checkpoints/epoch=124-step=24499.ckpt"
        #"weights/DinoDataset-epoch=39-val_loss=0.00.ckpt"
        #"weights/15nz0fepoch=78-step=15483.ckpt",
    )

    backbone_parameters: Optional[str] = None

    if student_backbone == "vit":
        backbone_parameters: Dict[str, Any] = dict_field(
            dict(
                image_size = 32,
                patch_size = 8,
                num_classes = 10,
                dim = 2048,
                depth = 6,
                heads = 16,
                mlp_dim = 2048,
                dropout = 0.1,
                emb_dropout = 0.1
            )
    )

@dataclass
class DinoTwinConfig:
    """Hyperparameters specific to the DINO Model.
    Used when the `arch` option is set to "Barlow" in the hparams
    """

    student_backbone: str = choice("resnet50", "swinS", default="resnet50")
    teacher_backbone: str = choice("resnet50", "swinS", default="resnet50")
    proj_layers: int = 3
    proj_channels: int = 2048
    bottleneck_dim: int = 256
    out_channels: int = 2048
    # number of crops/global_crops
    n_crops: int = 5
    # number of global crops
    n_global_crops: int = 2
    # scale range of the crops
    global_crops_scale: List[int] = list_field(0.5, 1)
    local_crops_scale: List[float] = list_field(0.05, 0.5)
    warmup_teacher_temp_epochs: int = 10  # Default 30
    student_temp: float = 0.1
    teacher_temp: float = 0.07  # Default 0.04, can be linearly increased to 0.07 but then it becomes unstable
    warmup_teacher_temp: float = (
        0.04  # would be different from techer temp if we used a warmup for this param
    )
    center_momentum: float = 0.9  # Default 0.9

    # barlow twin scale
    lmbda: float = 5e-3

    # scale for the BT loss
    bt_beta: float = 5e-3 * 0.5
    max_epochs: int = 400  # This is redundant with the hparms max_epochs
    # number of classes to use for the fine tuning task
    num_cat: int = 10

    weight_checkpoint: Optional[str] = osp.join(
        os.getcwd(),
        "wandb/test-deep-learning/2z4ulgmh/checkpoints/epoch=79-step=15679.ckpt",
    )

@dataclass
class Parameters:
    """base options."""

    # Dataset parameters.
    # dataset: DatasetParams = DatasetParams()
    # Set of parameters related to the optimizer.
    # optimizer: OptimizerParams = OptimizerParams()
    # GAN Settings
    hparams: Hparams = Hparams()
    optim_param: OptimizerParams = OptimizerParams()

    def __post_init__(self):
        """Post-initialization code"""
        # Mostly used to set some values based on the chosen hyper parameters
        # since we will use different models, backbones and datamodules

        # Set render number of channels
        if "BarlowTwins" in self.hparams.arch:
            self.network_param: BarlowConfig    = BarlowConfig()
        elif "Dinow" in self.hparams.arch:
            self.network_param: DinoTwinConfig  = DinoTwinConfig()
        elif  "Dino" in self.hparams.arch:
            self.network_param: DinoConfig      = DinoConfig()
        # Set random seed
        if self.hparams.seed_everything is None:
            self.hparams.seed_everything = random.randint(1, 10000)

        self.data_param: DatasetParams = DatasetParams()
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
