import math
import torch
import torch.nn as nn
from ..nn import Swish, SimpleSwish, Identity, ada_group_norm


class BasicModel(nn.Module):
    def __init__(self):
        super(BasicModel, self).__init__()

    def initialize_weights(self, module=None):
        if module is None:
            module = self.modules()
        else:
            module = module.modules()
        for m in module:
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight,
                                        mode="fan_out",
                                        nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def fuse_bn(self, module=None, replace_by_gn=False):
        if module is None:
            module = self
        last_conv = None
        for name, m in module.named_children():
            if isinstance(m, nn.Conv2d):
                last_conv = name
            elif isinstance(m, nn.BatchNorm2d):
                if last_conv is None:
                    continue
                conv = module._modules[last_conv]
                w = conv.weight
                mean = m.running_mean
                var_sqrt = torch.sqrt(m.running_var + m.eps)

                beta = m.weight
                gamma = m.bias

                if conv.bias is not None:
                    b = conv.bias
                else:
                    b = mean.new_zeros(mean.shape)

                w = w * (beta / var_sqrt).reshape([conv.out_channels, 1, 1, 1])
                b = (b - mean) / var_sqrt * beta + gamma
                module._modules[last_conv] = nn.Conv2d(conv.in_channels,
                                                       conv.out_channels,
                                                       conv.kernel_size,
                                                       conv.stride,
                                                       conv.padding,
                                                       conv.dilation, 
                                                       conv.groups,
                                                       bias=True)
                module._modules[last_conv].weight = nn.Parameter(w)
                module._modules[last_conv].bias = nn.Parameter(b)
                if replace_by_gn:
                    module._modules[name] = ada_group_norm(conv.out_channels, conv.in_channels // conv.groups)
                else:
                    module._modules[name] = Identity()
                last_conv = None
            else:
                last_conv = None
                self.fuse_bn(m, replace_by_gn)

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
