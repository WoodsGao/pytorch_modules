from pytorch_modules.backbones.resnet import resnet50, resnet18
import torch
from torchvision.models import ResNet

t = torch.rand([2, 3, 224, 224])
model = resnet18(pretrained=True)
o = model(t)
print(o.size())
o.mean().backward()

for s in model.stages:
    t = s(t)
    print(t.size())