import types
import torch.nn as nn
from ..nn import MBConvBlock, Bottleneck, BasicBlock, ConvNormAct, SeparableConvNormAct
from torch.utils.checkpoint import checkpoint

MODULES = (
    nn.Conv2d,
    nn.BatchNorm2d,
    nn.Linear,
    MBConvBlock,
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


def convert_to_ckpt_model(module):
    if isinstance(module, MODULES):
        module.dummy_forward = module.forward
        module.forward = types.MethodType(checkpoint_forward, module)
        return True
    if isinstance(module, nn.Sequential):
        if False not in [
                isinstance(m, MODULES) for name, m in module.named_children()
        ]:
            module.dummy_forward = module.forward
            module.forward = types.MethodType(checkpoint_forward, module)
            return True

    for name, m in module.named_children():
        convert_to_ckpt_model(m)
    return True
