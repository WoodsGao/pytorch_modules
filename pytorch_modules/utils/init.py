import torch.nn as nn


def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            nn.init.normal_(m.weight, std=0.001)
            for name, _ in m.named_parameters():
                if name in ['bias']:
                    nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.ConvTranspose2d):
            nn.init.normal_(m.weight, std=0.001)
            for name, _ in m.named_parameters():
                if name in ['bias']:
                    nn.init.constant_(m.bias, 0)
