from pytorch_lightning import LightningModule
from vit_pytorch import ViT

def vit(vit_parameters):
    return ViT(**vit_parameters)
