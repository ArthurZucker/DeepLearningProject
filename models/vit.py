from pytorch_lightning import LightningModule
from vit_pytorch import ViT


class vit(LightningModule):
    def __init__(self, config) -> None:
        super().__init__()
        self.net = ViT(
                    image_size = 32,
                    patch_size = 8,
                    num_classes = 1000,
                    dim = 2048,
                    depth = 6,
                    heads = 16,
                    mlp_dim = 2048,
                    dropout = 0.1,
                    emb_dropout = 0.1
                    )

    def forward(self, x):
        return self.net(x)