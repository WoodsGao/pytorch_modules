import types

import torch
import torch.nn as nn
import torch.quantization as Q
from torch.quantization import QuantStub, DeQuantStub

# def fuse(module):
#     last_conv = None
#     for name, m in module.named_children():
#         if isinstance(m, nn.Conv2d):
#             last_conv = name
#         elif isinstance(m, nn.BatchNorm2d):
#             if last_conv is None:
#                 continue
#             conv = module._modules[last_conv]
#             w = conv.weight
#             mean = m.running_mean
#             var_sqrt = torch.sqrt(m.running_var + m.eps)
#
#             beta = m.weight
#             gamma = m.bias
#
#             if conv.bias is not None:
#                 b = conv.bias
#             else:
#                 b = mean.new_zeros(mean.shape)
#
#             w = w * (beta / var_sqrt).reshape([conv.out_channels, 1, 1, 1])
#             b = (b - mean) / var_sqrt * beta + gamma
#             module._modules[last_conv] = nn.Conv2d(conv.in_channels,
#                                                    conv.out_channels,
#                                                    conv.kernel_size,
#                                                    conv.stride,
#                                                    conv.padding,
#                                                    conv.dilation,
#                                                    conv.groups,
#                                                    bias=True)
#             module._modules[last_conv].weight = nn.Parameter(w)
#             module._modules[last_conv].bias = nn.Parameter(b)
#             module._modules[name] = nn.Identity()
#             last_conv = None
#         else:
#             last_conv = None
#             fuse(m)


def tensor_quant(quant, x):
    if isinstance(x, torch.Tensor):
        return quant(x)
    if isinstance(x, (list, tuple)):
        return [tensor_quant(quant, i) for i in x]
    return x


def tensor_dequant(dequant, x):
    if isinstance(x, torch.Tensor):
        return dequant(x)
    if isinstance(x, (list, tuple)):
        return [tensor_dequant(dequant, i) for i in x]
    return x


def quant_forward(self, x):
    x = tensor_quant(self.quant, x)
    x = self.dummy_forward(x)
    x = tensor_dequant(self.dequant, x)
    return x


def replace_quant_forward(module):

    module.quant = QuantStub()
    module.dequant = DeQuantStub()
    module.dummy_forward = module.forward
    module.forward = types.MethodType(quant_forward, module)


def quant_functional_forward(self, x):
    def _qadd(a, b):
        print(a, b)
        return self.QF.add(a, b)

    # print(self)
    _add = torch.Tensor.__add__
    torch.Tensor.__add__ = _qadd
    x = self.dummy_forward(x)
    torch.Tensor.__add__ = _add
    return x


def replace_quant_functions(module):
    for name, m in module.named_children():
        if len(m._modules) == 0 or isinstance(
                m, (nn.Sequential, nn.quantized.FloatFunctional)):
            continue
        # print(name, type(m))
        module._modules[name].dummy_forward = module._modules[name].forward
        module._modules[name].QF = torch.nn.quantized.FloatFunctional()
        module._modules[name].forward = types.MethodType(
            quant_functional_forward, module._modules[name])
        # module._modules[name]._qadd = types.MethodType(_qadd, module._modules[name])
        replace_quant_functions(m)

