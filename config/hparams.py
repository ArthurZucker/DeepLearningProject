import os
import random
from dataclasses import dataclass
from os import path as osp
from typing import Any, ClassVar, Dict, List, Optional

import numpy as np
import simple_parsing
import torch
import torch.optim
from simple_parsing.helpers import Serializable, choice, dict_field, list_field

"""Dataclass allows to have arguments and easily use wandb's weep module.

Each arguments can be preceded by a comment, which will describe it when you call 'main.pu --help
An example of every datatype is provided. Some of the available arguments are also available.
Most notably, the agent, dataset, optimizer and loss can all be specified and automatically parsed
"""


@dataclass
class Hparams: 
    """Hyperparameters of Your Model"""

    # ----------------------
    # Wandb Parameters
    # ----------------------
    testing               : bool          = True
    wandb_project         : str           = f"{'test'*testing}-deep-learning"     # name of the project
    wandb_entity          : str           = "dinow-twins"       # name of the wandb entity,
    save_dir              : str           = osp.join(os.getcwd(), "wandb") # directory to save wandb outputs
    arch                  : str           = "Dino"              # choice("BarlowTwinsFT","BarlowTwins", "Dino", "DinoTwins", default="BarlowTwins")
    datamodule            : str           = "DinoDataModule"    # datamodule used. 
    # The same module is used for dino/dinotwins and a different one is used for barlow twins
    # dataset used. The same dataset is used for dino/dinotwins and a different one is used for barlow twins 
    # Moreover, the datasets are different depending on the task: SSL or Eval.
    dataset               : Optional[str] = "DinoDataset"       # dataset : has to correspond to a file name
    agent                 : str           = "trainer"           # agent used for training, only one is available now
    seed_everything       : Optional[int] = None                # seed for the whole run, if None a random seed will be selected
    
    # --------------------
    # Training parameters
    # --------------------
    tune_lr               : bool          = False   # tune the model's learning rate 
    tune_batch_size       : bool          = False   # tune the model's batch size 
    gpu                   : int           = 1       # gpu index
    precision             : int           = 32      # precision
    val_freq              : int           = 1       # validation frequency
    dev_run               : bool          = False   # developpment mode, only run 1 batch of train val and test
    accumulate_size       : int           = 1024    # gradient accumulation batch size
    max_epochs            : int           = 400     # number of epochs
    asset_path            : str           = osp.join(os.getcwd(), "assets") # path to download data

    # --------------------
    # Logging parameters
    # --------------------
    log_pred_freq         : int           = 10      # log_pred_freq
    log_ccM_freq          : int           = 1       # log cc_M matrix frequency
    log_dino_freq         : int           = 1       # log output frrequency for dino
    weights_path          : str           = osp.join(os.getcwd(), "weights") # path to save weights
    attention_threshold   : float         = 0.8     # threshold used to get attention map of multiple heads
    nb_attention          : int           = 5       # number of images used to display attention maps


@dataclass
class DatasetParams: 
    """Dataset Parameters"""

    num_workers        : int         = 20           # number of workers for dataloadersint
    input_size         : tuple       = (32, 32)     # image_size
    batch_size         : int         = 128          # batch_size
    asset_path         : str         = osp.join(os.getcwd(), "assets")  # path to download the dataset
    n_crops            : int         = 5            # number of crops
    n_global_crops     : int         = 2            # number of global crops
    global_crops_scale : List[int]   = list_field(0.5, 1)       # scale range of the global crops
    local_crops_scale  : List[float] = list_field(0.05, 0.5)    # scale range of the local crops


@dataclass
class OptimizerParams: 
    """Optimization parameters"""

    optimizer           : str            = "AdamW"  # Optimizer default vit: AdamW, default resnet50: Adam
    lr                  : float          = 5e-4     # learning rate, default = 5e-4
    min_lr              : float          = 5e-6     # min lr reached at the end of the cosine schedule
    lr_sched_type       : str            = "step"   # Learning rate scheduler type.
    betas               : List[float]    = list_field(0.9, 0.999)  # beta1 for adam. default = (0.9, 0.999)
    max_epochs          : int            = 400      # number of epochs
    warmup_epochs       : int            = 10       # number of warmup epochs for the learning rate and centering in Dino/Dino Twins

    scheduler_parameters: Dict[str, Any] = dict_field(
        dict(
            base_value         = 0.9995,
            final_value        = 1,
            max_epochs         = 0,
            niter_per_ep       = 0,
            warmup_epochs      = 0,
            start_warmup_value = 0,
        )
    )
    lr_scheduler_parameters: Dict[str, Any] = dict_field(
        dict(
            base_value         = 0,
            final_value        = 0,
            max_epochs         = 0,
            niter_per_ep       = 0,
            warmup_epochs      = 10,
            start_warmup_value = 0,
        )
    )


@dataclass
class BarlowConfig: 
    """Hyperparameters specific to Barlow Twin Model.
    Used when the `arch` option is set to "Barlow" in the hparams
    """

    bt_proj_dim : int = 2048  # number of channels to use for projection
    encoder     : str = choice("resnet50", "swinS", default="resnet50") # backbone encoder for barlow twins
    # lambda coefficient used to scale the scale of the redundancy loss so it doesn't overwhelm the invariance loss
    lmbda                 : float         = 5e-3
    pretrained_encoder    : bool          = False
    use_backbone_features : bool          = True
    num_cat               : int           = 10     # number of classes to use for the fine tuning task
    weight_checkpoint     : Optional[str] = osp.join(os.getcwd(),)
    # "wandb/test-deep-learning/lebgzheo/checkpoints/epoch=189-step=4749.ckpt", # model checkpoint used in evaluation phase


@dataclass
class DinoConfig: 
    """Hyperparameters specific to the DINO Model.
    Used when the `arch` option is set to "Barlow" in the hparams
    """
    backbone                   : str         = "vit"
    proj_layers               : int         = 3
    proj_dim                  : int         = 2048
    bottleneck_dim            : int         = 256
    out_dim                   : int         = 4096
    n_crops                   : int         = 5         # number of crops
    n_global_crops            : int         = 2         # number of global crops
    global_crops_scale        : List[int]   = list_field(0.5, 1) # scale range of the global crops
    local_crops_scale         : List[float] = list_field(0.05, 0.5) # scale range of the local crops
    warmup_teacher_temp_epochs: int         = 10        # Default 30
    student_temp              : float       = 0.1
    teacher_temp              : float       = 0.07      # Default 0.04, can be linearly increased to 0.07 but then it becomes unstable
    warmup_teacher_temp       : float       = 0.04      # (starting teacher temp) different from teacher temp only if we use a warmup
    center_momentum           : float       = 0.9       # Default 0.9
    num_cat                   : int         = 10        # number of classes to use for the fine tuning task
    pretrained                : bool        = False  


    weight_checkpoint: Optional[str] = osp.join(os.getcwd(),"weights/dino/epoch=386-step=75851.ckpt",) # model checkpoint used in evaluation phase
    backbone_parameters: Optional[str] = None

    if backbone == "vit":
        backbone_parameters: Dict[str, Any]    = dict_field(
                dict(
                    image_size  = 32,
                    patch_size  = 4,
                    num_classes = 0,
                    dim         = 192,
                    depth       = 4,
                    heads       = 6,
                    mlp_dim     = 1024,
                    dropout     = 0.1,
                    emb_dropout = 0.1,
                )
        )


@dataclass
class DinoTwinConfig: 
    """Hyperparameters specific to the DINO Model.
    Used when the `arch` option is set to "Barlow" in the hparams
    """

    student_backbone            : str         = "resnet50"
    teacher_backbone            : str         = student_backbone
    proj_layers                 : int         = 3
    proj_dim                    : int         = 2048
    bottleneck_dim              : int         = 256
    out_dim                     : int         = 2048
    n_crops                     : int         = 5                       # number of crops
    n_global_crops              : int         = 2                       # number of global crops
    global_crops_scale          : List[int]   = list_field(0.5, 1)      # scale range of the global crops
    local_crops_scale           : List[float] = list_field(0.05, 0.5)   # scale range of the local crops
    warmup_teacher_temp_epochs  : int         = 10                      # Default 30
    student_temp                : float       = 0.1
    teacher_temp                : float       = 0.07                    # (final teacher temp)Default 0.04, can be linearly increased to 0.07 but then Dino becomes unstable
    warmup_teacher_temp         : float       = 0.04                    # (starting teacher temp) different from teacher temp only if we use a warmup
    center_momentum             : float       = 0.9                     # Default 0.9

    lmbda                       : float = 5e-3                          # lambda coefficient used to scale the scale of the redundancy loss so it doesn't overwhelm the invariance loss

    
    bt_beta                     : float             = 5e-3 * 0.5    # scaling of the barlow twins loss. Default is meant to get bt_loss = 1/2 * Dino_loss at the begining of training
    num_cat                     : int               = 10            # number of classes to use for the fine tuning task
    backbone_parameters         : Dict[str, Any]    = None
    if  student_backbone == "vit":
        backbone_parameters     : Dict[str, Any]    = dict_field(
            dict(
                image_size      = 32,
                patch_size      = 4,
                num_classes     = 0,
                dim             = 192,
                depth           = 4,
                heads           = 6,
                mlp_dim         = 1024,
                dropout         = 0.1,
                emb_dropout     = 0.1,
            )
        )

    weight_checkpoint           : Optional[str] = osp.join(
        os.getcwd(),
                                # "weights/dinowtwins_2heads/epoch=68-step=13523.ckpt",
    )                           # model checkpoint used in evaluation phase


@dataclass
class Parameters: 
    """base options."""

    hparams    : Hparams         = Hparams()
    optim_param: OptimizerParams = OptimizerParams()

    def __post_init__(self): 
        """Post-initialization code"""
        # Mostly used to set some values based on the chosen hyper parameters
        # since we will use different models, backbones and datamodules

        # Set render number of channels
        if   "BarlowTwins" in self.hparams.arch: 
            self.network_param: BarlowConfig = BarlowConfig()
        elif "DinoTwins" in self.hparams.arch: 
            self.network_param: DinoTwinConfig = DinoTwinConfig()
        elif "Dino" == self.hparams.arch or "DinoFT" == self.hparams.arch: 
            self.network_param: DinoConfig = DinoConfig()
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
        args        = parser.parse_args()
        instance: Parameters = args.parameters
        return instance
