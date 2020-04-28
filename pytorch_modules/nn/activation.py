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


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)
