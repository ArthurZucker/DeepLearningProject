import torch
def get_attention(attention):
    """Defines a hook for a transfomer architecture
    It should be registered on a layer such that the output contains the attention
    The model will then store the attentions in a list of atention for the batch of images
    This might be heavy and shall be modified @TODO improve memory performances
    """
    def hook(model, input, output):
        attention.append(output.cpu().detach().numpy())
    return hook