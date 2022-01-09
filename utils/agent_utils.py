import importlib

def get_net(arch,network_param, optimizer_param = None):
    """
    Get Network Architecture based on arguments provided
    """
    # FIXME this iss fucking strange the import needs to be done twice to work
    mod = importlib.import_module(f"models.{arch}")
    net = getattr(mod,arch)
    if optimizer_param is not None:
        return net(network_param,optimizer_param)
    else : 
        return net(network_param)


def get_datamodule(datamodule,data_param,dataset = None):
    """
    Fetch Network Function Pointer
    """
    module = "datamodules." + datamodule
    mod = importlib.import_module(module)
    net = getattr(mod, datamodule)
    return net(data_param,dataset)
