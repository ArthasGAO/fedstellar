# 
# This file is part of the fedstellar framework (see https://github.com/enriquetomasmb/fedstellar).
# Copyright (c) 2022 Enrique Tomás Martínez Beltrán.
#
import json
import os
import shutil
import sys
from math import floor
from collections import defaultdict

# To Avoid Crashes with a lot of nodes
import torch.multiprocessing
from PIL import Image
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Subset, random_split, Dataset
from torchvision.datasets import MNIST, utils
from torchvision import transforms
import numpy as np

torch.multiprocessing.set_sharing_strategy("file_system")


class FEMNIST(MNIST):
    def __init__(self, sub_id, number_sub, root_dir, train=True, transform=None, target_transform=None, download=False):
        super(MNIST, self).__init__(root_dir, transform=transform, target_transform=target_transform)
        self.sub_id = sub_id
        self.number_sub = number_sub
        self.download = download
        self.download_link = 'https://media.githubusercontent.com/media/GwenLegate/femnist-dataset-PyTorch/main/femnist.tar.gz'
        self.file_md5 = '60433bc62a9bff266244189ad497e2d7'
        self.train = train
        self.root = root_dir
        self.training_file = f'{self.root}/FEMNIST/processed/femnist_train.pt'
        self.test_file = f'{self.root}/FEMNIST/processed/femnist_test.pt'

        if not os.path.exists(f'{self.root}/FEMNIST/processed/femnist_test.pt') or not os.path.exists(f'{self.root}/FEMNIST/processed/femnist_train.pt'):
            if self.download:
                self.dataset_download()
            else:
                raise RuntimeError('Dataset not found, set parameter download=True to download')
        else:
            print('FEMNIST dataset already downloaded')

        if self.train:
            data_file = self.training_file
        else:
            data_file = self.test_file

        # Whole dataset
        data_and_targets = torch.load(data_file)
        self.data, self.targets = data_and_targets[0], data_and_targets[1]

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])
        img = Image.fromarray(img.numpy(), mode='F')
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def dataset_download(self):
        paths = [f'{self.root}/FEMNIST/raw/', f'{self.root}/FEMNIST/processed/']
        for path in paths:
            if not os.path.exists(path):
                os.makedirs(path)

        # download files only if they do not exist
        print('Downloading FEMNIST dataset...')
        filename = self.download_link.split('/')[-1]
        utils.download_and_extract_archive(self.download_link, download_root=f'{self.root}/FEMNIST/raw/', filename=filename, md5=self.file_md5)

        files = ['femnist_train.pt', 'femnist_test.pt']
        for file in files:
            # move to processed dir
            shutil.move(os.path.join(f'{self.root}/FEMNIST/raw/', file), f'{self.root}/FEMNIST/processed/')


#######################################
#    FEMNISTDataModule for FEMNIST    #
#######################################


class FEMNISTDataModule(LightningDataModule):
    """
    LightningDataModule of partitioned FEMNIST.

    This dataset is derived from the Leaf repository
    (https://github.com/TalwalkarLab/leaf) pre-processing of the Extended MNIST
    dataset, grouping examples by writer. Details about Leaf were published in
    "LEAF: A Benchmark for Federated Settings" https://arxiv.org/abs/1812.01097.

    The FEMNIST dataset is naturally non-iid

    IMPORTANT: The data is generated using ./preprocess.sh -s niid --sf 0.05 -k 0 -t sample (small-sized dataset)

    Details: 62 different classes (10 digits, 26 lowercase, 26 uppercase), images are 28 by 28 pixels (with option to make them all 128 by 128 pixels), 3500 users

    Args:

    """

    # Singleton
    femnist_train = None
    femnist_val = None

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

        transform_data = transforms.Compose(
            [
                transforms.CenterCrop((96, 96)),
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize((28, 28)),
                transforms.ColorJitter(contrast=3),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))]
        )

        self.train = FEMNIST(sub_id=self.sub_id, number_sub=self.number_sub, root_dir=root_dir, train=True, transform=transform_data, target_transform=None, download=True)
        self.test = FEMNIST(sub_id=self.sub_id, number_sub=self.number_sub, root_dir=root_dir, train=False, transform=transform_data, target_transform=None, download=True)

        if len(self.test) < self.number_sub:
            raise ValueError("Too many partitions")

        # Training / validation set
        trainset = self.train
        rows_by_sub = floor(len(trainset) / self.number_sub)
        tr_subset = Subset(
            trainset, range(self.sub_id * rows_by_sub, (self.sub_id + 1) * rows_by_sub)
        )
        femnist_train, femnist_val = random_split(
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
            femnist_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )
        self.val_loader = DataLoader(
            femnist_val,
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
                len(femnist_train), len(femnist_val), len(testset)
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
