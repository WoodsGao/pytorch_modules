import torch.nn as nn

from .utils import build_conv2d


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 groups=1,
                 base_width=64,
                 dilation=1):
        super(BasicBlock, self).__init__()
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = build_conv2d(inplanes, planes, 3, stride, groups,
                                  dilation)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = build_conv2d(planes, planes * self.expansion, 3, 1,
                                  groups, dilation)
        self.bn2 = nn.BatchNorm2d(planes * self.expansion)
        self.stride = stride
        if stride != 1 or inplanes != planes * self.expansion:
            self.downsample = nn.Sequential(
                build_conv2d(inplanes, planes * self.expansion, 1, stride),
                nn.BatchNorm2d(planes * self.expansion),
            )
        else:
            self.downsample = None

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 groups=1,
                 base_width=64,
                 dilation=1):
        super(Bottleneck, self).__init__()
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = build_conv2d(inplanes, width, 1)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = build_conv2d(width, width, 3, stride, groups, dilation)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = build_conv2d(width, planes * self.expansion, 1)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride
        if stride != 1 or inplanes != planes * self.expansion:
            self.downsample = nn.Sequential(
                build_conv2d(inplanes, planes * self.expansion, 1, stride),
                nn.BatchNorm2d(planes * self.expansion),
            )
        else:
            self.downsample = None

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
