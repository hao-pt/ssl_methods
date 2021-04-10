import os

import numpy as np

import torch
import torchvision
from torchvision import transforms
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

from data_transform import TransformTwice
from data_samplers import TwoStreamBatchSampler, relabel_dataset
from data_transform import RandomTranslateWithReflect

NO_LABEL = -1

def cifar10(datadir):
    channel_stats = dict(mean=[0.4914, 0.4822, 0.4465],
                         std=[0.2470,  0.2435,  0.2616])
    train_transformation = TransformTwice(transforms.Compose([
        RandomTranslateWithReflect(4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(**channel_stats)
    ]))
    eval_transformation = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(**channel_stats)
    ])

    return {
        'train_transformation': train_transformation,
        'eval_transformation': eval_transformation,
        'datadir': datadir,
        'num_classes': 10
    }

DATASET_ZOO = {"cifar10": cifar10} 
def create_data_loaders(datadir, cfgs):
    """
    Load images and sample data from train and val set

    Returns:
    ---
    - train_loader
    - val_loader
    """
    data_transformer = DATASET_ZOO[cfgs.dataset]
    train_transformation, eval_transformation, _, _ = data_transformer(datadir).values()
    
    traindir = os.path.join(datadir, "train")
    evaldir = os.path.join(datadir, "val")

    dataset = torchvision.datasets.ImageFolder(traindir, train_transformation)

    if cfgs.labels:
        # read labeled examples
        with open(cfgs.labels) as f:
            labels = dict(line.split(' ') for line in f.read().splitlines())
        labeled_idxs, unlabeled_idxs = relabel_dataset(dataset, labels)

    if cfgs.exclude_unlabeled:
        sampler = SubsetRandomSampler(labeled_idxs)
        batch_sampler = BatchSampler(sampler, cfgs.batch_size, drop_last=True)
    elif cfgs.labeled_batch_size:
        batch_sampler = TwoStreamBatchSampler(
            unlabeled_idxs, labeled_idxs, cfgs.batch_size, cfgs.labeled_batch_size)

    train_loader = torch.utils.data.DataLoader(dataset,
                                               batch_sampler=batch_sampler,
                                               num_workers=cfgs.workers,
                                               pin_memory=True)

    eval_loader = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder(evaldir, eval_transformation),
        batch_size=cfgs.batch_size,
        shuffle=False,
        num_workers=2 * cfgs.workers,  # Needs images twice as fast
        pin_memory=True,
        drop_last=False)

    return train_loader, eval_loader

if __name__ == "__main__":
    from config import Config
    cfg = Config()
    train_loader, val_loader = create_data_loaders("datadir/cifar10", cfg)
    print(len(train_loader.dataset), train_loader.dataset)