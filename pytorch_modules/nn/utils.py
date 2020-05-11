import torch
import torch.nn as nn
import torch.nn.functional as F

from .activation import Identity, Mish

BN_MOMENTUM = 0.1


def build_conv2d(inplanes,
                 planes,
                 ksize=3,
                 stride=1,
                 groups=1,
                 dilation=1,
                 bias=False):
    return nn.Conv2d(inplanes,
                     planes,
                     ksize,
                     stride=stride,
                     padding=(ksize - 1) // 2 * dilation,
                     groups=groups,
                     dilation=dilation,
                     bias=bias)


class ConvNormAct(nn.Sequential):
    def __init__(self,
                 inplanes,
                 planes,
                 ksize=3,
                 stride=1,
                 groups=1,
                 dilation=1,
                 activate=nn.ReLU(inplace=True)):
        super(ConvNormAct, self).__init__(
            build_conv2d(inplanes,
                         planes,
                         ksize,
                         stride=stride,
                         groups=groups,
                         dilation=dilation,
                         bias=False),
            nn.BatchNorm2d(planes, momentum=BN_MOMENTUM),
            activate if activate is not None else Identity(),
        )


class SeparableConvNormAct(nn.Sequential):
    def __init__(self,
                 inplanes,
                 planes,
                 ksize=3,
                 stride=1,
                 dilation=1,
                 activate=nn.ReLU(inplace=True)):
        super(SeparableConvNormAct, self).__init__(
            ConvNormAct(inplanes,
                        inplanes,
                        ksize,
                        stride=stride,
                        groups=inplanes,
                        dilation=dilation,
                        activate=None),
            ConvNormAct(inplanes, planes, 1, activate=activate),
        )


class SeparableConv(nn.Sequential):
    def __init__(self,
                 inplanes,
                 planes,
                 ksize=3,
                 stride=1,
                 dilation=1,
                 bias=True):
        super(SeparableConv, self).__init__(
            ConvNormAct(inplanes,
                        inplanes,
                        ksize,
                        stride=stride,
                        groups=inplanes,
                        dilation=dilation,
                        activate=None),
            build_conv2d(inplanes, planes, 1, bias=bias),
        )
