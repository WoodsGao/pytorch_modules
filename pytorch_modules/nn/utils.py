import torch
import torch.nn as nn
import torch.nn.functional as F
from .activation import Swish, Identity


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
                 activate=nn.ReLU(True),
                 bn_momentum=0.1):
        super(ConvNormAct, self).__init__(
            nn.Conv2d(inplanes,
                      planes,
                      ksize,
                      stride=stride,
                      padding=(ksize - 1) // 2 * dilation,
                      groups=groups,
                      dilation=dilation,
                      bias=False),
            nn.BatchNorm2d(planes, momentum=bn_momentum),
            activate if activate is not None else Identity(),
        )


class SeparableConvNormAct(nn.Sequential):
    def __init__(self,
                 inplanes,
                 planes,
                 ksize=3,
                 stride=1,
                 dilation=1,
                 activate=nn.ReLU(True)):
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
                 activate=nn.ReLU(True),
                 bias=True):
        super(SeparableConv, self).__init__(
            ConvNormAct(inplanes,
                        inplanes,
                        ksize,
                        stride=stride,
                        groups=inplanes,
                        dilation=dilation,
                        activate=None),
            nn.Conv2d(inplanes, planes, 1, bias=bias),
        )
