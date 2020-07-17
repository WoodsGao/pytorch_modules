import random

import numpy as np
import torch
import torch.nn as nn


def initialize_weights(module: nn.Module):
    """
    Initialize the weights of a module.

    Args:
        module (nn.Module): a torch.nn.Module instance.
    """
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


def set_seed(seed: int = 666):
    """
    Set random seed for random/numpy/torch.

    Args:
        seed (int, optional): random seed. Defaults to 666.
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
