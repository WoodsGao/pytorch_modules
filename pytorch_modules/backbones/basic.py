import torch


def imagenet_normalize(x):
    x = x - torch.FloatTensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1).to(
        x.device)
    x = x / torch.FloatTensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1).to(
        x.device)
    return x
