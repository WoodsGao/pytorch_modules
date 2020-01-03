from pytorch_modules.backbones import efficientnet
import torch

t = torch.rand([2, 3, 224, 224])
model = efficientnet(0, pretrained=True, replace_stride_with_dilation=[True, False, True])
o = model(t)
print(o.size())
o.mean().backward()
# print(model._conv_stem.state_dict(), model.stages[0].state_dict())
for s in model.stages:
    t = s(t)
    print(t.size())