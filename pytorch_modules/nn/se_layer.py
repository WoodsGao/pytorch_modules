import torch.nn as nn

from .activation import Mish


class SELayer(nn.Module):
    def __init__(self, filters, div=4):
        super(SELayer, self).__init__()
        self.block = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(filters, filters // div, 1),
            Mish(),
            nn.Conv2d(filters // div, filters, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.weight(x)
