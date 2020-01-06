import math
import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url
from . import imagenet_normalize
from ..utils import initialize_weights
from ..nn import build_conv2d, MBConvBlock, Swish

model_urls = [
    'http://storage.googleapis.com/public-models/efficientnet/efficientnet-b0-355c32eb.pth',
    'http://storage.googleapis.com/public-models/efficientnet/efficientnet-b1-f1951068.pth',
    'http://storage.googleapis.com/public-models/efficientnet/efficientnet-b2-8bb594d6.pth',
    'http://storage.googleapis.com/public-models/efficientnet/efficientnet-b3-5fb5a3c3.pth',
    'http://storage.googleapis.com/public-models/efficientnet/efficientnet-b4-6ed6700e.pth',
    'http://storage.googleapis.com/public-models/efficientnet/efficientnet-b5-b6417697.pth',
    'http://storage.googleapis.com/public-models/efficientnet/efficientnet-b6-c76e70fd.pth',
    'http://storage.googleapis.com/public-models/efficientnet/efficientnet-b7-dcc49843.pth',
]

params_list = [
    # Coefficients:   width,depth,res,dropout
    (1.0, 1.0, 224, 0.2),
    (1.0, 1.1, 224, 0.2),
    (1.1, 1.2, 256, 0.3),
    (1.2, 1.4, 320, 0.3),
    (1.4, 1.8, 384, 0.4),
    (1.6, 2.2, 448, 0.4),
    (1.8, 2.6, 512, 0.5),
    (2.0, 3.1, 608, 0.5),
]


class EfficientNet(nn.Module):
    """
    An EfficientNet model. Most easily loaded with the .from_name or .from_pretrained methods
    """
    def __init__(self,
                 layers=[1, 2, 2, 3, 3, 4, 1],
                 widths=[32, 16, 24, 40, 80, 112, 192, 320],
                 se_ratio=0.25,
                 drop_rate=0.2,
                 num_classes=1000,
                 replace_stride_with_dilation=None,
                 bn_momentum=1e-2,
                 bn_eps=1e-3):
        super().__init__()

        ksizes = [3, 3, 5, 3, 5, 5, 3]
        strides = [1, 2, 2, 2, 1, 2, 1]
        expand_ratios = [1, 6, 6, 6, 6, 6, 6]

        # resnet like dilation
        dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        assert len(replace_stride_with_dilation) == 3
        replace_stride_with_dilation = [False, False] + replace_stride_with_dilation

        # Stem
        last_width = widths.pop(0)
        self._conv_stem = build_conv2d(3, last_width, 3, stride=2)
        self._bn0 = nn.BatchNorm2d(last_width,
                                   momentum=bn_momentum,
                                   eps=bn_eps)
        self._swish = Swish()

        # Build blocks
        self._blocks = nn.ModuleList([])
        self.stages = nn.ModuleList([])
        stage = [self._conv_stem, self._bn0, self._swish]
        last_idx = 0
        for idx, (depth, width, ksize, stride, expand_ratio) in enumerate(
                zip(layers, widths, ksizes, strides, expand_ratios)):
            if stride > 1:
                stage += self._blocks[last_idx:]
                self.stages.append(nn.Sequential(*stage))
                stage = []
                last_idx = len(self._blocks)
         
            self._blocks.append(
                MBConvBlock(last_width,
                            width,
                            ksize,
                            1 if replace_stride_with_dilation[len(self.stages)] else stride,
                            dilation=dilation,
                            expand_ratio=expand_ratio,
                            se_ratio=se_ratio,
                            drop_rate=drop_rate * idx /
                            (len(self._blocks) + 1),
                            bn_momentum=bn_momentum,
                            bn_eps=bn_eps))
            last_width = width
            if replace_stride_with_dilation[len(self.stages)] and stride == 2:
                dilation *= 2
            for _ in range(depth - 1):
                self._blocks.append(
                    MBConvBlock(last_width,
                                width,
                                ksize,
                                1,
                                dilation=dilation,
                                expand_ratio=expand_ratio,
                                se_ratio=se_ratio,
                                drop_rate=drop_rate * idx /
                                (len(self._blocks) + 1),
                                bn_momentum=bn_momentum,
                                bn_eps=bn_eps))
        stage += self._blocks[last_idx:]
        self.stages.append(nn.Sequential(*stage))

        # Head
        self._conv_head = build_conv2d(width, width * 4, 1)
        self._bn1 = nn.BatchNorm2d(width * 4, momentum=bn_momentum, eps=bn_eps)

        # Final linear layer
        self._avg_pooling = nn.AdaptiveAvgPool2d(1)
        self._dropout = nn.Dropout(0.2)
        self._fc = nn.Linear(width * 4, num_classes)
        initialize_weights(self)

    def forward(self, x):
        x = imagenet_normalize(x)
        """ Calls extract_features to extract features, applies final linear layer, and returns logits. """
        # Stem
        for stage in self.stages:
            x = stage(x)

        # Head
        x = self._conv_head(x)
        x = self._bn1(x)
        x = self._swish(x)

        # Pooling and final linear layer
        x = self._avg_pooling(x)
        x = torch.flatten(x, 1)
        x = self._dropout(x)
        x = self._fc(x)
        return x


def efficientnet(model_id=0, pretrained=False, progress=True, **kwargs):
    assert model_id >= 0 and model_id <= 7
    params = params_list[model_id]
    layers = [math.ceil(l * params[1]) for l in [1, 2, 2, 3, 3, 4, 1]]
    widths = [w * params[0] for w in [32, 16, 24, 40, 80, 112, 192, 320]]
    int_widths = [max(16, int(w + 4) // 8 * 8) for w in widths]
    widths = [
        iw + 8 if iw < 0.9 * w else iw for w, iw in zip(widths, int_widths)
    ]
    drop_rate = params[3]
    model = EfficientNet(layers, widths, drop_rate=drop_rate, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[model_id],
                                              progress=progress)
        model.load_state_dict(state_dict, strict=False)
    return model
