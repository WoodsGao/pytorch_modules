import torch
import torch.nn as nn

from ..nn import ConvNormAct, SeparableConvNormAct
from ..utils import initialize_weights


class MiniNet(nn.Module):
    def __init__(self, num_classes=10, drop_rate=0.2):
        super(MiniNet, self).__init__()
        self.stages = nn.ModuleList([
            ConvNormAct(3, 32, 7, stride=2),
            ConvNormAct(32, 64, stride=2),
            ConvNormAct(64, 128, stride=2),
            ConvNormAct(128, 256, stride=2),
            ConvNormAct(256, 512, stride=2),
        ])

        initialize_weights(self)

    def forward(self, x):
        features = []
        for stage in self.stages:
            x = stage(x)
            features.append(x)

        return tuple(features)
