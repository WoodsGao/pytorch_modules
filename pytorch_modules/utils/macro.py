import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.autograd.set_detect_anomaly(True)

IMG_EXT = ['.jpg', '.jpeg', '.png', '.tiff', '.JPEG']