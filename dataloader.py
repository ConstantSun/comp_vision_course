# coding: utf-8

import numpy as np

import torch
import torch.utils.data

import torchvision
import torchvision.models
import torchvision.transforms
import timm
model = timm.create_model("eca_nfnet_l0", pretrained=True, num_classes=10 )

config = model.default_cfg
img_size = config["test_input_size"][-1] if "test_input_size" in config else config["input_size"][-1]



def get_loader(batch_size, num_workers):
    mean = np.array([0.4914, 0.4822, 0.4465])
    std = np.array([0.2470, 0.2435, 0.2616])

    train_transform = timm.data.transforms_factory.transforms_imagenet_eval(
        img_size=img_size,
        interpolation=config["interpolation"],
        mean=config["mean"],
        std=config["std"],
        crop_pct=config["crop_pct"],
    )
    test_transform = timm.data.transforms_factory.transforms_imagenet_eval(
        img_size=img_size,
        interpolation=config["interpolation"],
        mean=config["mean"],
        std=config["std"],
        crop_pct=config["crop_pct"],
    )

    dataset_dir = '~/.torchvision/datasets/CIFAR10'
    train_dataset = torchvision.datasets.CIFAR10(
        dataset_dir, train=True, transform=train_transform, download=True)
    test_dataset = torchvision.datasets.CIFAR10(
        dataset_dir, train=False, transform=test_transform, download=True)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
    )
    return train_loader, test_loader
