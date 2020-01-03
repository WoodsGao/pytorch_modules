import os
import torch
import torch.optim as optim
import torch.distributed as dist
from tqdm import tqdm
from . import device
from . import convert_to_ckpt_model

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
                 weights='',
                 accumulate=1,
                 adam=False,
                 lr=1e-3,
                 warmup=10,
                 lr_decay=1e-3,
                 mixed_precision=False):
        self.accumulate_count = 0
        self.metrics = 0
        self.epoch = 0
        self._lr = lr
        self.warmup = warmup
        self.lr_decay = lr_decay
        lr = self.lr_schedule()
        self.accumulate = accumulate
        self.fetcher = fetcher
        self.loss_fn = loss_fn
        if amp is None:
            self.mixed_precision = False
        else:
            self.mixed_precision = mixed_precision
        model = model.to(device)
        if adam:
            optimizer = optim.AdamW(model.parameters(),
                                    lr=lr,
                                    weight_decay=1e-5,
                                    eps=1e-4 if self.mixed_precision else 1e-8)
        else:
            optimizer = optim.SGD(model.parameters(),
                                  lr=lr,
                                  momentum=0.9,
                                  weight_decay=1e-5)
        self.adam = adam
        self.model = model
        self.optimizer = optimizer
        if weights:
            self.load(weights)
        convert_to_ckpt_model(self.model)
        if self.mixed_precision:
            self.model, self.optimizer = amp.initialize(
                self.model,
                self.optimizer,
                opt_level='O1',
                verbosity=0)
            print('amp initialized')
        if dist.is_initialized():
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
        return lr

    def load(self, weights):
        state_dict = torch.load(weights, map_location=device)
        if self.adam:
            if 'adam' in state_dict:
                self.optimizer.load_state_dict(state_dict['adam'])
        else:
            if 'sgd' in state_dict:
                self.optimizer.load_state_dict(state_dict['sgd'])
        if 'm' in state_dict:
            self.metrics = state_dict['m']
        if 'e' in state_dict:
            self.epoch = state_dict['e']
        self.model.load_state_dict(state_dict['model'], strict=False)
        lr = self.lr_schedule()
        for pg in self.optimizer.param_groups:
            pg['lr'] = lr

    def save(self, save_path_list):
        if len(save_path_list) == 0:
            return False
        state_dict = {
            'model':
            self.model.module.state_dict()
            if dist.is_initialized() else self.model.state_dict(),
            'm':
            self.metrics,
            'e':
            self.epoch
        }
        if self.adam:
            state_dict['adam'] = self.optimizer.state_dict()
        else:
            state_dict['sgd'] = self.optimizer.state_dict()
        for save_path in save_path_list:
            torch.save(state_dict, save_path)

    def run_epoch(self):
        print('Epoch: %d' % self.epoch)
        self.model.train()
        total_loss = 0
        pbar = tqdm(enumerate(self.fetcher), total=len(self.fetcher))
        for idx, (inputs, targets) in pbar:
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
        # lr warmup and decay
        lr = self.lr_schedule()
        for pg in self.optimizer.param_groups:
            pg['lr'] = lr
