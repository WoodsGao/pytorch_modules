import torch.nn as nn
from .build import build_modules


class BoneNeckHead(nn.Module):
    """    
    A torch model with backbone, neck and head.
    Suitable for most classification/detection/segmentation tasks.
    There should only be one backbone.
    There can be multiple necks, and multiple necks will become a sequential model.
    There can be more multiple heads, and multiple heads will be connected to the neck in parallel.
    """
    def __init__(self, backbone, neck, head, *args, **kwargs):
        super(BoneNeckHead, self).__init__()
        self.backbone = build_modules(backbone)
        self.neck = build_modules(neck)
        if isinstance(self.neck, list):
            self.neck = nn.Sequential(*self.neck)
        self.head = build_modules(head)
        if isinstance(self.head, list):
            self.head = nn.ModuleList(self.head)

    def forward(self, x):
        x = self.backbone(x)
        x = self.neck(x)
        if isinstance(self.head, nn.ModuleList):
            x = [head(x) for head in self.head]
        else:
            x = self.head(x)
        return x
