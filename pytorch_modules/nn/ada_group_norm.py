import torch.nn as nn
from . import Identity


def ada_group_norm(planes, weight_len=None):
    if weight_len is None:
        weight_len = planes
    if weight_len % 4 != 0:
        return Identity()
    else:
        c = weight_len
        p = 1
        while c % 2 == 0 and c > p:
            c /= 2
            p *= 2
        return nn.GroupNorm(int(p), planes)

