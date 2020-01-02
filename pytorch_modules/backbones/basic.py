import torch
import torch.nn as nn
from ..nn import Swish, SimpleSwish, Identity


def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight,
                                    mode="fan_out",
                                    nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)


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


def replace_bn(module, sync=True):
    if sync:
        for name, m in module.named_children():
            if isinstance(m, nn.BatchNorm2d):
                sync_bn = nn.SyncBatchNorm(m.num_features)
                sync_bn.running_var = m.running_var
                sync_bn.running_mean = m.running_mean
                sync_bn.weight = m.weight
                sync_bn.bias = m.bias
    else:
        for name, m in module.named_children():
            if isinstance(m, nn.SyncBatchNorm):
                normal_bn = nn.BatchNorm(m.num_features)
                normal_bn.running_var = m.running_var
                normal_bn.running_mean = m.running_mean
                normal_bn.weight = m.weight
                normal_bn.bias = m.bias


def freeze(module):
    for param in module.parameters():
        param.requires_grad = False
    module.eval()


def set_swish(module, simple=False):
    if simple:
        for m in module:
            if isinstance(m, Swish):
                m = SimpleSwish()
    else:
        for m in module:
            if isinstance(m, SimpleSwish):
                m = Swish()


def imagenet_normalize(x):
    x -= torch.FloatTensor([0.485, 0.456, 0.406]).reshape(1, 3, 1,
                                                          1).to(x.device)
    x /= torch.FloatTensor([0.229, 0.224, 0.225]).reshape(1, 3, 1,
                                                          1).to(x.device)
    return x
