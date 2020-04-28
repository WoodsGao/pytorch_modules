import sys
import types

import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from ..nn import BasicBlock, Bottleneck, ConvNormAct, SeparableConvNormAct

sys.setrecursionlimit(9000000)

MODULES = (
    nn.Conv2d,
    nn.BatchNorm2d,
    nn.Linear,
    Bottleneck,
    BasicBlock,
    ConvNormAct,
    SeparableConvNormAct,
)


def checkpoint_forward(self, x):
    if self.training:
        if not x.requires_grad:
            x.requires_grad_()
        return checkpoint(self.dummy_forward, x)
    else:
        return self.dummy_forward(x)


def convert_to_ckpt_model(module, recursion=3):
    # if isinstance(module, MODULES):
    #     module.dummy_forward = module.forward
    #     module.forward = types.MethodType(checkpoint_forward, module)
    #     return True
    # if isinstance(module, nn.Sequential):
    #     if False not in [
    #             isinstance(m, MODULES) for name, m in module.named_children()
    #     ]:
    #         module.dummy_forward = module.forward
    #         module.forward = types.MethodType(checkpoint_forward, module)
    #         return True
    # if recursion >= 0:
    #     for name, m in module.named_children():
    #         convert_to_ckpt_model(m, recursion - 1)
    # return True
    module.dummy_forward = module.forward
    module.forward = types.MethodType(checkpoint_forward, module)
    return True
