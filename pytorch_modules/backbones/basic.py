import math
import torch
import torch.nn as nn
from ..nn import Swish, SimpleSwish


class BasicModel(nn.Module):
    def __init__(self):
        super(BasicModel, self).__init__()

    def initialize_weights(self, module=None):
        if module is None:
            module = self.modules()
        else:
            module = module.modules()
        for m in module:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode="fan_out",
                                        nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0.01)
                nn.init.zeros_(m.bias)

    def freeze(self, module=None):
        if module is None:
            module = self
        for param in module.parameters():
            param.requires_grad = False
        module.eval()

    def freeze_bn(self, module=None, half=False):
        if module is None:
            module = self.modules()
        else:
            module = module.modules()
        for m in module:
            if isinstance(m, nn.BatchNorm2d):
                if half:
                    m.eval().half()
                else:
                    m.eval()

    def set_swish(self, simple=False):
        if simple:
            for m in self.modules():
                if isinstance(m, Swish):
                    m = SimpleSwish()
        else:
            for m in self.modules():
                if isinstance(m, SimpleSwish):
                    m = Swish()

    def forward(self, x):
        return x
