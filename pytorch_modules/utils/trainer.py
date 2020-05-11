import os
import os.path as osp
import shutil

import torch
import torch.distributed as dist
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from . import convert_to_ckpt_model, device

amp = None
try:
    from apex import amp
    from apex.parallel import DistributedDataParallel as DDP
    from apex.parallel import convert_syncbn_model
except ImportError:
    pass
if device == 'cuda':
    torch.backends.cudnn.benchmark = True


class Trainer:
    def __init__(self,
                 model,
                 fetcher,
                 loss_fn,
                 workdir,
                 accumulate=1,
                 adam=False,
                 lr=1e-3,
                 weight_decay=1e-3,
                 warmup=10,
                 lr_decay=1e-3,
                 resume=True,
                 weights='',
                 mixed_precision=False,
                 checkpoint=False):
        self.accumulate_count = 0
        self.metrics = 0
        self.epoch = 0
        self._lr = lr
        self.warmup = warmup
        self.lr_decay = lr_decay
        self.accumulate = accumulate
        self.fetcher = fetcher
        self.loss_fn = loss_fn
        self.workdir = workdir
        os.makedirs(workdir, exist_ok=True)

        if amp is None:
            self.mixed_precision = False
        else:
            print('amp loaded')
            self.mixed_precision = mixed_precision
        model = model.to(device)
        if adam:
            optimizer = optim.AdamW(model.parameters(),
                                    lr=lr,
                                    weight_decay=weight_decay,
                                    eps=1e-4 if self.mixed_precision else 1e-8)
        else:
            optimizer = optim.SGD(model.parameters(),
                                  lr=lr,
                                  momentum=0.9,
                                  weight_decay=weight_decay)
        self.adam = adam
        self.model = model
        self.optimizer = optimizer
        if weights:
            self.load(weights, resume)
        if checkpoint:
            convert_to_ckpt_model(self.model)
        if self.mixed_precision:
            self.model, self.optimizer = amp.initialize(self.model,
                                                        self.optimizer,
                                                        opt_level='O1',
                                                        verbosity=0)
            print('amp initialized')
        if dist.is_initialized():
            if amp is None:
                self.model = DDP(
                    self.model, device_ids=[int(os.environ.get('LOCAL_RANK'))])
            else:
                convert_syncbn_model(self.model)
                self.model = DDP(self.model, delay_allreduce=True)

        self.optimizer.zero_grad()
        if dist.is_initialized():
            self.model.require_backward_grad_sync = False

    def lr_schedule(self):
        if self.epoch < self.warmup:
            lr = self._lr * (self.epoch + 1) / (self.warmup + 1)
        else:
            lr = self._lr * (1 - self.lr_decay)**(self.epoch - self.warmup)
        print('lr change to: %6lf' % lr)
        self.set_lr(lr)

    def set_lr(self, lr):
        for pg in self.optimizer.param_groups:
            pg['lr'] = lr

    def load(self, weights, resume):
        state_dict = torch.load(weights, map_location=device)
        if resume:
            if self.adam:
                if 'adam' in state_dict:
                    self.optimizer.load_state_dict(state_dict['adam'])
            else:
                if 'sgd' in state_dict:
                    self.optimizer.load_state_dict(state_dict['sgd'])
            if 'metrics' in state_dict:
                self.metrics = state_dict['metrics']
            if 'epoch' in state_dict:
                self.epoch = state_dict['epoch']
        self.model.load_state_dict(state_dict['model'], strict=False)
        self.lr_schedule()
        torch.cuda.empty_cache()

    def save(self, best=False):
        if dist.is_initialized():
            if dist.get_rank() > 0:
                return False
        state_dict = {
            'model':
            self.model.module.state_dict()
            if dist.is_initialized() else self.model.state_dict(),
            'metrics':
            self.metrics,
            'epoch':
            self.epoch
        }
        if self.adam:
            state_dict['adam'] = self.optimizer.state_dict()
        else:
            state_dict['sgd'] = self.optimizer.state_dict()
        cwd = os.getcwd()
        os.chdir(self.workdir)
        save_path = 'epoch_%d.pth' % self.epoch
        torch.save(state_dict, save_path)
        if osp.exists('last.pth'):
            os.remove('last.pth')
        os.symlink(save_path, 'last.pth')
        if best:
            if osp.exists('best.pth'):
                os.remove('best.pth')
            os.symlink(save_path, 'best.pth')
        os.chdir(cwd)

    def step(self):
        # lr warmup and decay
        self.lr_schedule()
        print('Epoch: %d' % self.epoch)
        self.model.train()
        total_loss = 0
        pbar = tqdm(self.fetcher)
        for idx, (inputs, targets) in enumerate(pbar):
            if inputs.size(0) < 2:
                continue
            self.accumulate_count += 1
            batch_idx = idx + 1
            if self.accumulate_count % self.accumulate == 0 and dist.is_initialized(
            ):
                self.model.require_backward_grad_sync = True
            outputs = self.model(inputs)
            loss = self.loss_fn(outputs, targets, self.model)
            if torch.isnan(loss):
                print('nan loss')
                continue
            total_loss += loss.item()
            loss /= self.accumulate
            # Compute gradient
            if self.mixed_precision:
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            mem = torch.cuda.memory_cached() / 1E9 if torch.cuda.is_available(
            ) else 0  # (GB)
            pbar.set_description('mem: %8g, loss: %8g' %
                                 (mem, total_loss / batch_idx))
            if self.accumulate_count % self.accumulate == 0:
                self.accumulate_count = 0
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10)
                self.optimizer.step()
                self.optimizer.zero_grad()
                if dist.is_initialized():
                    self.model.require_backward_grad_sync = False
        torch.cuda.empty_cache()
        self.epoch += 1
