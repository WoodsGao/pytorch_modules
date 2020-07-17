# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from typing import Any

from omegaconf import Config, ListConfig

from .load import load_obj


def build_modules(cfg: Config, *args, **kwargs) -> Any:
    """
    Build a module or a set of modules from hydra config.

    Args:
        cfg (omegaconf.Config): hydra config

    Returns:
        an instance of cfg.class_name
    """
    if isinstance(cfg, ListConfig):
        return list(map(lambda cfg: build_modules(cfg, *args, **kwargs), cfg))
    else:
        if cfg.get('params'):
            return load_obj(cfg.class_name)(*args, **cfg.params, **kwargs)
        else:
            return load_obj(cfg.class_name)(*args, **kwargs)
