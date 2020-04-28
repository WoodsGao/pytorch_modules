import torch
import torch.nn as nn

from ..nn import Identity


def fuse(module):
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
            module._modules[name] = Identity()
            last_conv = None
        else:
            last_conv = None
            fuse(m)
