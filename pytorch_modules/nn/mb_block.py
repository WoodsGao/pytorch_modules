import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import build_conv2d
from .activation import Swish, Identity
from .drop_connect import DropConnect


class MBConvBlock(nn.Module):
    """
    Mobile Inverted Residual Bottleneck Block
    """
    def __init__(self,
                 inplanes,
                 planes,
                 ksize=3,
                 stride=1,
                 dilation=1,
                 expand_ratio=6,
                 se_ratio=0.25,
                 drop_rate=0.2,
                 bn_momentum=1e-2,
                 bn_eps=1e-3):
        super().__init__()
        self.id_skip = (stride == 1 and inplanes == planes)
        if self.id_skip:
            self.drop_connect = DropConnect(
                drop_rate) if drop_rate > 0 else Identity()

        # Expansion phase
        self.expand_ratio = expand_ratio
        midplanes = inplanes * expand_ratio
        if expand_ratio != 1:
            self._expand_conv = build_conv2d(inplanes, midplanes, 1)
            self._bn0 = nn.BatchNorm2d(midplanes,
                                       momentum=bn_momentum,
                                       eps=bn_eps)
        self._depthwise_conv = build_conv2d(
            midplanes,
            midplanes,
            ksize,
            stride,
            dilation=dilation,
            groups=midplanes,
        )
        self._bn1 = nn.BatchNorm2d(midplanes, momentum=bn_momentum, eps=bn_eps)

        # Squeeze and Excitation layer, if desired
        self.se_ratio = se_ratio
        if se_ratio > 0:
            num_squeezed_channels = max(1, int(inplanes * se_ratio))
            self._se_reduce = build_conv2d(midplanes,
                                           num_squeezed_channels,
                                           1,
                                           bias=True)
            self._se_expand = build_conv2d(num_squeezed_channels,
                                           midplanes,
                                           1,
                                           bias=True)

        # Output phase
        self._project_conv = build_conv2d(midplanes, planes, 1)
        self._bn2 = nn.BatchNorm2d(planes, momentum=bn_momentum, eps=bn_eps)
        self._swish = Swish()

    def forward(self, inputs):
        """
        :param inputs: input tensor
        :param drop_connect_rate: drop connect rate (float, between 0 and 1)
        :return: output of block
        """

        # Expansion and Depthwise Convolution
        x = inputs
        if self.expand_ratio != 1:
            x = self._expand_conv(x)
            x = self._bn0(x)
            x = self._swish(x)
        x = self._depthwise_conv(x)
        x = self._bn1(x)
        x = self._swish(x)

        # Squeeze and Excitation
        if self.se_ratio > 0:
            x_squeezed = F.adaptive_avg_pool2d(x, 1)
            x_squeezed = self._se_expand(
                self._swish(self._se_reduce(x_squeezed)))
            x = torch.sigmoid(x_squeezed) * x

        x = self._bn2(self._project_conv(x))

        # Skip connection and drop connect
        if self.id_skip:
            return inputs + self.drop_connect(x)
        else:
            return x
