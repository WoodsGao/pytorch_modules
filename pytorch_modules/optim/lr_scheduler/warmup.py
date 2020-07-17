import torch.optim
from torch.optim.lr_scheduler import  LambdaLR


class WarmupLR(LambdaLR):

    def __init__(self, optimizer, steps=10):
        super(WarmupLR, self).__init__(optimizer, lambda epoch: epoch / steps if epoch < steps else 1)
