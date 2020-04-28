# pytorch_modules

## Introduction

A neural network toolkit built on pytorch/opencv/numpy that includes neural network layers, modules, loss functions, optimizers, data loaders, data augmentation, etc.

## Features

 - Advanced neural network modules/loss functions/optimizers
 - Ultra-efficient trainer and dataloader that allows you to take full advantage of GPU

## Installation

    sudo pip3 install pytorch_modules
    
or

    sudo python3 setup.py install

## Usage

### pytorch_modules.utils

Includes a variety of utils for pytorch model training.
See [woodsgao/pytorch_segmentation](https://github.com/woodsgao/pytorch_segmentation) as a tutorial.

### pytorch_modules.nn

This module contains a variety of neural network layers, modules and loss functions.

    import torch
    from pytorch_modules.nn import ResBlock
    
    # NCHW tensor
    inputs = torch.ones([8, 8, 224, 224])
    block = ResBlock(8, 16)
    outputs = block(inputs)

### pytorch_modules.backbones

This module includes a series of modified backbone networks.

    import torch
    from pytorch_modules.backbones import ResNet
    
    # NCHW tensor
    inputs = torch.ones([8, 8, 224, 224])
    model = ResNet(32)
    outputs = model.stages[0](inputs)

### pytorch_modules.datasets

This module includes a series of dataset classes integrated from `pytorch_modules.datasets.BasicDataset` which is integrated from `torch.utils.data.Dataset` .
The loading method of `pytorch_modules.datasets.BasicDataset` is modified to cache data with `LMDB` to speed up data loading. This allows your gpu to be fully used for model training without spending a lot of time on data loading and data augmentation. 
Please see the corresponding repository for detailed usage.
