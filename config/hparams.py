from dataclasses import dataclass
from typing import List
import numpy as np
from simple_parsing.helpers import list_field

import os
"""Dataclass allows to have arguments and easily use wandb's weep module.

Each arguments can be preceded by a comment, which will describe it when you call 'main.pu --help
An example of every datatype is provided. Some of the available arguments are also available.
Most notably, the agent, dataset, optimizer and loss can all be specified and automatically parsed
"""


@dataclass
class hparams:
    """Hyperparameters of Your Model"""
    # learning rate 
    lr : float = 3e-4
    # lambda coefficient used for FIXME
    lmbda: float = 0.05
    # number of channels to use for projection
    bt_proj_channels: int = 2048 # TODO run a sweep for that?
    
    
    ###Probably needs a different organization but I put the new parameters here
    # assumes square images
    img_size: int = 32
    #number of crops/global_crops
    n_crops = 8
    n_global_crops = 2
    