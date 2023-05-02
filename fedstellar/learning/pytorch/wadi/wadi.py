# 
# This file is part of the fedstellar framework (see https://github.com/enriquetomasmb/fedstellar).
# Copyright (c) 2022 Enrique Tomás Martínez Beltrán.
#
import os
import sys
from math import floor

# To Avoid Crashes with a lot of nodes
import torch.multiprocessing
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import transforms
from torchvision.datasets import MNIST, utils
import urllib.request
import numpy as np
torch.multiprocessing.set_sharing_strategy("file_system")


class WADI(MNIST):
    def __init__(self, sub_id, number_sub, root_dir, train=True):
        super(MNIST, self).__init__(root_dir, transform=None, target_transform=None)
        self.sub_id = sub_id
        self.number_sub = number_sub
        self.download_link = 'XXXX'
        self.files = ["X_train.npy","y_train.npy","X_test.npy","y_test.npy"]
        self.train = train
        self.root = root_dir

        if not os.path.exists(f'{self.root}/WADI/X_train.npy') or not os.path.exists(f'{self.root}/WADI/y_train.npy') or not os.path.exists(f'{self.root}/WADI/X_test.npy') or not os.path.exists(f'{self.root}/WADI/y_test.npy'):
            self.dataset_download()

        if self.train:
            data_file = self.training_file
            self.data, self.targets = torch.from_numpy(np.load(f'{self.root}/WADI/X_train.npy')), torch.from_numpy(np.load(f'{self.root}/WADI/y_train.npy'))
            self.data = self.data.to(torch.float32)
            self.targets = self.targets.to(torch.float32)
        else:
            data_file = self.test_file
            self.data, self.targets = torch.from_numpy(np.load(f'{self.root}/WADI/X_test.npy')), torch.from_numpy(np.load(f'{self.root}/WADI/y_test.npy'))
            self.data = self.data.to(torch.float32)
            self.targets = self.targets.to(torch.float32)

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])
        return img, target

    def dataset_download(self):
        paths = [f'{self.root}/WADI/']
        for path in paths:
            if not os.path.exists(path):
                os.makedirs(path)

        # download files
        for file in self.files:
            urllib.request.urlretrieve(
                os.path.join(f'{self.download_link}', file),
                os.path.join(f'{self.root}/WADI/', file))


#######################################
#    FederatedDataModule for WADI     #
#######################################


class WADIDataModule(LightningDataModule):
    """
    LightningDataModule of partitioned WADI.

    Args:
        sub_id: Subset id of partition. (0 <= sub_id < number_sub)
        number_sub: Number of subsets.
        batch_size: The batch size of the data.
        num_workers: The number of workers of the data.
        val_percent: The percentage of the validation set.
    """

    # Singleton
    wadi_train = None
    wadi_val = None

    def __init__(
            self,
            sub_id=0,
            number_sub=1,
            batch_size=32,
            num_workers=4,
            val_percent=0.1,
            root_dir=None,
    ):
        super().__init__()
        self.sub_id = sub_id
        self.number_sub = number_sub
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_percent = val_percent
        self.root_dir = root_dir


        self.train = WADI(sub_id=self.sub_id, number_sub=self.number_sub, root_dir=root_dir, train=True)
        self.test = WADI(sub_id=self.sub_id, number_sub=self.number_sub, root_dir=root_dir, train=False)


        if len(self.test) < self.number_sub:
            raise ("Too much partitions")

        # Training / validation set
        trainset = self.train
        rows_by_sub = floor(len(trainset) / self.number_sub)
        tr_subset = Subset(
            trainset, range(self.sub_id * rows_by_sub, (self.sub_id + 1) * rows_by_sub)
        )
        wadi_train, wadi_val = random_split(
            tr_subset,
            [
                round(len(tr_subset) * (1 - self.val_percent)),
                round(len(tr_subset) * self.val_percent),
            ],
        )

        # Test set
        testset = self.test
        rows_by_sub = floor(len(testset) / self.number_sub)
        te_subset = Subset(
            testset, range(self.sub_id * rows_by_sub, (self.sub_id + 1) * rows_by_sub)
        )

        # DataLoaders
        self.train_loader = DataLoader(
            wadi_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )
        self.val_loader = DataLoader(
            wadi_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
        self.test_loader = DataLoader(
            te_subset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
        print(
            "Train: {} Val:{} Test:{}".format(
                len(wadi_train), len(wadi_val), len(te_subset)
            )
        )

    def train_dataloader(self):
        """ """
        return self.train_loader

    def val_dataloader(self):
        """ """
        return self.val_loader

    def test_dataloader(self):
        """ """
        return self.test_loader
