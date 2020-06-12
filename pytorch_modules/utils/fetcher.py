from queue import Empty, Full, Queue
from threading import Lock, Thread
from time import sleep, time

import torch

from . import device as default_device


# Referencehttps://github.com/NVIDIA/apex/blob/master/examples/imagenet/main_amp.py
class Fetcher:
    def __init__(self, loader, post_fetch_fn=None, device=None):
        self.idx = 0
        self.loader = loader
        self.loader_iter = iter(loader)
        self.post_fetch_fn = post_fetch_fn
        self.device = device if device is not None else default_device
        if self.device == 'cuda':
            self.stream = torch.cuda.Stream()
        self.preload()

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        return self

    def preload(self):
        try:
            self.batch = next(self.loader_iter)
        except StopIteration:
            self.batch = None
            self.loader_iter = iter(self.loader)
            return None
        if self.device == 'cuda':
            with torch.cuda.stream(self.stream):
                self.batch = [
                    b.cuda(non_blocking=True)
                    if isinstance(b, torch.Tensor) else b for b in self.batch
                ]

    def __next__(self):
        if self.device == 'cuda':
            torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.batch
        self.preload()
        if batch is None:
            raise StopIteration
        if self.post_fetch_fn is not None:
            batch = self.post_fetch_fn(batch)
        return batch
