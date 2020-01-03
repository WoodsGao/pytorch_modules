def freeze(module):
    for param in module.parameters():
        param.requires_grad = False
    module.eval()