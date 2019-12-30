from pytorch_modules.nn import MBConvBlock
import torch

t = torch.rand([2, 32, 64, 64])
model = MBConvBlock(32, 128, 5, 2)
o = model(t)
print(o.size())
o.mean().backward()

