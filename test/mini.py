from pytorch_modules.backbones.mini import MiniNet
import torch

t = torch.rand([2, 3, 224, 224])
model = MiniNet()
o = model(t)
print(o.size())
o.mean().backward()

for s in model.stages:
    t = s(t)
    print(t.size())