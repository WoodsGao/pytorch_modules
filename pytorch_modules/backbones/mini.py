import math
import torch
import torch.nn as nn
from . import BasicModel
from ..nn import CNS, SeparableCNS


class MiniNet(BasicModel):
    def __init__(self, num_classes=10, drop_rate=0.2):
        super(MiniNet, self).__init__()
        self.stages = nn.ModuleList([
            CNS(3, 32, 7, stride=2),
            SeparableCNS(32, 64, stride=2),
            SeparableCNS(64, 128, stride=2),
            SeparableCNS(128, 256, stride=2),
            SeparableCNS(256, 512, stride=2),
        ])
        self._avg_pooling = nn.AdaptiveAvgPool2d(1)
        self._dropout = nn.Dropout(drop_rate)
        self._fc = nn.Linear(512, num_classes)

        self._initialize_weights()

    def forward(self, x):
        for stage in self.stages:
            x = stage(x)
        # Pooling and final linear layer
        x = self._avg_pooling(x)
        x = torch.flatten(x, 1)
        x = self._dropout(x)
        x = self._fc(x)
        return x
