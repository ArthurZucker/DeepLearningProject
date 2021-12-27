from torch import nn
import importlib
from torch import optim
from torch.nn import MarginRankingLoss

def get_net(args, name=None):
    """
    Get Network Architecture based on arguments provided
    """
    name = name if name is not None else args.arch
    module = "models." + name
    mod = importlib.import_module(module)
    net = getattr(mod, name)
    return net(args)


def get_datamodule(args):
    """
    Fetch Network Function Pointer
    """
    module = "datamodule." + args.datamodule
    mod = importlib.import_module(module)
    net = getattr(mod, (args.datamodule))
    return net(args)
