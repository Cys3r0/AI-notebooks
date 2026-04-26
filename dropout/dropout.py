import torch.nn as nn
from torch import Tensor
import torch
#TODO Train a simple MNIST model with and without this dropout

class DropOut(nn.Module):
    def __init__(self, p:float = 0.5):
        super().__init__()
        self.p = p #p of dropping element

    def forward(self, x: Tensor) -> Tensor:
        if not self.training:
            return x
        
        keep_p = 1 - self.p
        mask = torch.bernoulli(torch.full_like(x, keep_p))
        return x * mask * 1/(keep_p)
