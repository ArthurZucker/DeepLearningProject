"""
Auteur: Arthur Zucker
- Capture configuration
- Update with argvs
- launches training through agent
"""
from __future__ import absolute_import, division

import wandb
from simple_parsing import ArgumentParser

from agents import *
from config.hparams import hparams

from pytorch_lightning.loggers import WandbLogger
# from apex import amp
parser = ArgumentParser()
# automatically add arguments for all the fields of the classes in hparams:
parser.add_arguments(hparams, dest="hparams")
args = parser.parse_args()


def main():
    # initialize wandb instance
    wandb_run = WandbLogger(config=vars(args.hparams), project="test-project", entity="dinow-twins", allow_val_change=True)
    config = wandb_run.experiment.config
    # Create the Agent and pass all the configuration to it then run it..
    agent_class = globals()[config.agent]
    agent = agent_class(config, wandb_run)
    # run the model
    agent.run()



if __name__ == "__main__":
    main()
