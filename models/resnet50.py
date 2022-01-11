from pytorch_lightning import LightningModule
from torchvision.models import resnet50


class Resnet50(LightningModule):
    def __init__(self, config) -> None:
        super().__init__()
        self.net = resnet50(pretrained=False)

    def forward(self, x):
        return self.net(x)
