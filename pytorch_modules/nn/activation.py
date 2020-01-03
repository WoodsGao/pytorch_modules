import torch
import torch.nn as nn
import torch.nn.functional as F


class CReLU(nn.Module):
    def forward(self, x):
        return torch.cat([F.relu(x), F.relu(-x)], 1)


class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


class Identity(nn.Module):
    def forward(self, x):
        return x


class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class Swish(nn.Module):
    def forward(self, x):
        return SwishImplementation.apply(x)


class SimpleSwish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


def set_swish(module, simple=False):
    if simple:
        for m in module:
            if isinstance(m, Swish):
                m = SimpleSwish()
    else:
        for m in module:
            if isinstance(m, SimpleSwish):
                m = Swish()