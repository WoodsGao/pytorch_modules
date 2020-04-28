import torch
import torch.nn as nn


class DropConnect(nn.Module):
    def __init__(self, p=0.5):
        super(DropConnect, self).__init__()
        assert p < 1
        self.p = p

    def forward(self, x):
        if not self.training:
            return x
        random_tensor = (1 - self.p) + torch.rand(
            [x.size(0), 1, 1, 1], dtype=x.dtype, device=x.device)
        binary_tensor = torch.floor(random_tensor)
        x.mul_(binary_tensor / (1 - self.p))
        return x
