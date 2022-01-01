import torch.nn as nn 

class L2Norm(nn.Module):
    def forward(self, x, eps = 1e-6):
        norm = x.norm(dim = 1, keepdim = True).clamp(min = eps)
        return x / norm