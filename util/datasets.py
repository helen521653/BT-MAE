# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

import os
import PIL

from torchvision import datasets, transforms

import torch

import types
import math

import sys

if not hasattr(torch, "_six"):
    torch._six = types.SimpleNamespace()
    import collections.abc as container_abcs
    torch._six.container_abcs = container_abcs
    torch._six.string_classes = (str,)
    torch._six.int_classes = (int,)
    torch._six.FileNotFoundError = FileNotFoundError
    torch._six.inf = math.inf
    sys.modules['torch._six'] = torch._six

from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


from torch.utils.data import Dataset
from datasets import load_dataset

import numpy as np
# --- Fix for deprecated NumPy aliases (np.float, np.int, etc.) ---
for alias in ("float", "int", "bool", "complex", "object"):
    if not hasattr(np, alias):
        setattr(np, alias, eval(alias))

class HuggingFaceDataset(Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item['image']
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        label = item.get('label', 0)
        return image, label


def build_dataset(is_train, use_hf, is_test, args):
    transform = build_transform(is_train, args)
    
    if use_hf:
        if is_train:
            ds = load_dataset("ilee0022/ImageNet100", split='train')
        else:
        
            if is_test:
                ds = load_dataset("ilee0022/ImageNet100", split='test')
            else:
                ds = load_dataset("ilee0022/ImageNet100", split='validation')
        
        dataset = HuggingFaceDataset(ds, transform=transform)
    else:
        root = os.path.join(args.data_path, 'train' if is_train else 'val')
        dataset = datasets.ImageFolder(root, transform=transform)

    # print(dataset)

    return dataset


def build_transform(is_train, args):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    # train transform
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation='bicubic',
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        return transform

    # eval transform
    t = []
    if args.input_size <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
    size = int(args.input_size / crop_pct)
    t.append(
        transforms.Resize(size, interpolation=PIL.Image.BICUBIC),  # to maintain same ratio w.r.t. 224 images
    )
    t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)
