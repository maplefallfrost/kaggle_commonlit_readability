import torch.nn as nn
import torch.nn.functional as F

class Normalize(nn.Module):
    def __init__(self, p_norm=1):
        super().__init__()
        self.p_norm = p_norm
    
    def forward(self, x):
        return F.normalize(x, p=self.p_norm, dim=1)