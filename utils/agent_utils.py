from torch import nn
import importlib
from torch import optim
from torch.nn import MarginRankingLoss

def get_net(arch,network_param, optimizer_param = None):
    """
    Get Network Architecture based on arguments provided
    """
    # FIXME this iss fucking strange the import needs to be done twice to work
    try: 
        mod = importlib.import_module(f"models.{arch}")
    except:
        mod = importlib.import_module(f"models.{arch}")
    net = getattr(mod,arch)
    return net(network_param,optimizer_param)


def get_datamodule(datamodule,data_param):
    """
    Fetch Network Function Pointer
    """
    module = "datamodules." + datamodule
    mod = importlib.import_module(module)
    net = getattr(mod, datamodule)
    return net(data_param)
