import torch
from omegaconf import DictConfig, ListConfig
from pytorch_lightning import LightningModule

from .build import build_modules


class BasicLitModel(LightningModule):
    """
    A sample of lightning module which includes a torch model, an optimizer and datasets.
    """
    def __init__(self, cfg: DictConfig):
        """
        Initialization.

        Args:
            cfg (DictConfig): the whole config of a project, including 'model', 'data', 'optimizer', etc.
        """
        super(BasicLitModel, self).__init__()
        self.cfg = cfg
        self.model = build_modules(self.cfg.model)

    def setup(self, step):
        self.train_dataset = build_modules(self.cfg.data.train)
        if self.cfg.data.get('val'):
            self.val_dataset = build_modules(self.cfg.data.val, augments=None)

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizers, schedulers = [], []

        def configure_optimizer_scheduler(cfg: DictConfig):
            """
            Build an optimizer of selected parameters.
            And build schedulers for this optimizer.

            Args:
                cfg (DictConfig): config includes 'module'(unnecessary)

            """
            if cfg.module:
                params = eval(cfg.module).parameters()
            else:
                params = self.parameters()
            optimizer = build_modules(cfg, params=params)
            scheduler = build_modules(cfg.scheduler, optimizer=optimizer)
            if scheduler is None:
                scheduler = []
            if not isinstance(scheduler, list):
                scheduler = [scheduler]
            optimizers.append(optimizer)
            schedulers.extend(scheduler)

        if isinstance(self.cfg.optimizer, ListConfig):
            list(map(configure_optimizer_scheduler, self.cfg.optimizer))
        else:
            configure_optimizer_scheduler(self.cfg.optimizer)
        return optimizers, schedulers

    def train_dataloader(self):
        train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.cfg.data.batch_size,
            num_workers=self.cfg.data.num_workers,
            shuffle=self.cfg.data.shuffle,
            collate_fn=self.train_dataset.collate_fn if hasattr(
                self.train_dataset, 'collate_fn') else None)
        return train_loader

    def val_dataloader(self):
        val_loader = torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.cfg.data.batch_size,
            num_workers=self.cfg.data.num_workers,
            collate_fn=self.val_dataset.collate_fn if hasattr(
                self.val_dataset, 'collate_fn') else None)
        return val_loader
