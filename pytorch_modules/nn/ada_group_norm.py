import torch.nn as nn
from . import Identity


def ada_group_norm(planes):
    if planes % 4 != 0:
        return Identity()
    else:
        c = planes
        p = 1
        while c % 2 == 0 and c > p:
            c /= 2
            p *= 2
        return nn.GroupNorm(int(p), planes)

