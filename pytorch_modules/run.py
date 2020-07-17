import argparse
import logging
import os
import os.path as osp
import sys

import hydra
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_modules.engine import build_model


def train(cfg):
    print(cfg.pretty())
    tb_logger = TensorBoardLogger(save_dir=cfg.general.save_dir)

    model = build_model(cfg)

    early_stopping = pl.callbacks.EarlyStopping(
        **cfg.callbacks.early_stopping.params)
    model_checkpoint = pl.callbacks.ModelCheckpoint(
        **cfg.callbacks.model_checkpoint.params)

    trainer = pl.Trainer(logger=tb_logger,
                         early_stop_callback=early_stopping,
                         checkpoint_callback=model_checkpoint,
                         **cfg.trainer)
    trainer.fit(model)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config_path', type=str, help='Your config file path.')
    parser.add_argument('--strict',
                        action='store_true',
                        help='Strict mode for hydra.')

    opt, left = parser.parse_known_args()
    sys.argv = sys.argv[:1] + left
    print(sys.path)
    hydra_wrapper = hydra.main(config_path=osp.join(os.getcwd(), opt.config_path), strict=opt.strict)
    hydra_wrapper(train)()
