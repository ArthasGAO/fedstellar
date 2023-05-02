#
# This file is part of the fedstellar framework (see https://github.com/enriquetomasmb/fedstellar).
# Copyright (c) 2022 Enrique Tomás Martínez Beltrán.
#
import os
import zipfile
import ast
from math import floor

import pandas as pd
# To Avoid Crashes with a lot of nodes
import torch.multiprocessing
from lightning import LightningDataModule
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset, random_split, Dataset
from torchvision.datasets import utils

torch.multiprocessing.set_sharing_strategy("file_system")


class SYSCALL(Dataset):
    def __init__(self, sub_id, number_sub, root_dir, train=True, transform=None, target_transform=None, download=False):
        self.transform = transform
        self.target_transform = target_transform
        self.sub_id = sub_id
        self.number_sub = number_sub
        self.download = download
        self.download_link = 'https://files.ifi.uzh.ch/CSG/research/fl/data/syscall.zip'
        self.train = train
        self.root = root_dir
        self.training_file = f'{self.root}/syscall/processed/syscall_train.pt'
        self.test_file = f'{self.root}/syscall/processed/syscall_test.pt'

        if not os.path.exists(f'{self.root}/syscall/processed/syscall_test.pt') or not os.path.exists(f'{self.root}/syscall/processed/syscall_train.pt'):
            if self.download:
                self.dataset_download()
                self.process()
            else:
                raise RuntimeError('Dataset not found, set parameter download=True to download')
        else:
            print('SYSCALL dataset already downloaded and processed.')

        if self.train:
            data_file = self.training_file
        else:
            data_file = self.test_file

        # Whole dataset
        data_and_targets = torch.load(data_file)
        self.data, self.targets = data_and_targets[0], data_and_targets[1]

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])
        if self.transform is not None:
            img = img
        if self.target_transform is not None:
            target = target
        return img, target

    def dataset_download(self):
        paths = [f'{self.root}/syscall/raw/', f'{self.root}/syscall/processed/']
        for path in paths:
            if not os.path.exists(path):
                os.makedirs(path, exist_ok=True)

        # download files only if they do not exist
        print('Downloading SYSCALL dataset...')
        filename = self.download_link.split('/')[-1]
        utils.download_and_extract_archive(self.download_link, download_root=f'{self.root}/syscall/raw/', filename=filename)

        with zipfile.ZipFile(f'{self.root}/syscall/raw/{filename}', 'r') as zip_ref:
            zip_ref.extractall(f'{self.root}/syscall/raw/')

    def process(self):
        print('Processing SYSCALL dataset...')
        df = pd.DataFrame()
        files = os.listdir(f'{self.root}/syscall/raw/')
        feature_name = 'system calls frequency_1gram-scaled'
        for f in files:
            if '.csv' in f:
                fi_path = f'{self.root}/syscall/raw/{f}'
                csv_df = pd.read_csv(fi_path, sep='\t')
                feature = [ast.literal_eval(i) for i in csv_df[feature_name]]
                csv_df[feature_name] = feature
                df = pd.concat([df, csv_df])
        df['maltype'] = df['maltype'].replace(to_replace='normalv2', value='normal')
        classes_to_targets = {}
        t = 0
        for i in set(df['maltype']):
            classes_to_targets[i] = t
            t += 1
        classes = list(classes_to_targets.keys())

        for c in classes_to_targets:
            df['maltype'] = df['maltype'].replace(to_replace=c, value=classes_to_targets[c])

        all_targes = torch.tensor(df['maltype'].tolist())
        all_data = torch.tensor(df[feature_name].tolist())

        x_train, x_test, y_train, y_test = train_test_split(all_data, all_targes, test_size=0.15, random_state=42)
        train = [x_train, y_train, classes_to_targets, classes]
        test = [x_test, y_test, classes_to_targets, classes]

        # save to files
        train_file = f'{self.root}/syscall/processed/syscall_train.pt'
        test_file = f'{self.root}/syscall/processed/syscall_test.pt'

        # save to processed dir
        if not os.path.exists(train_file):
            torch.save(train, train_file)
        if not os.path.exists(test_file):
            torch.save(test, test_file)


#######################################
#    SYSCALLDataModule for SYSCALL    #
#######################################


def sort_dataset(dataset):
    sorted_indexes = dataset.targets.sort()[1]
    dataset.targets = (dataset.targets[sorted_indexes])
    dataset.data = dataset.data[sorted_indexes]
    return dataset


class SYSCALLDataModule(LightningDataModule):
    """
    LightningDataModule of partitioned SYSCALL.

    Args:

    """

    # Singleton
    syscall_train = None
    syscall_val = None

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

        self.train = SYSCALL(sub_id=self.sub_id, number_sub=self.number_sub, root_dir=root_dir, train=True, download=True)
        self.test = SYSCALL(sub_id=self.sub_id, number_sub=self.number_sub, root_dir=root_dir, train=False, download=True)

        if len(self.test.data) < self.number_sub:
            raise ValueError("Too many partitions")

        # Training / validation set
        trainset = self.train
        rows_by_sub = floor(len(trainset.data) / self.number_sub)
        tr_subset = Subset(
            trainset, range(self.sub_id * rows_by_sub, (self.sub_id + 1) * rows_by_sub)
        )
        syscall_train, syscall_val = random_split(
            tr_subset,
            [
                round(len(tr_subset) * (1 - self.val_percent)),
                round(len(tr_subset) * self.val_percent),
            ],
        )

        # Test set
        testset = self.test
        rows_by_sub = floor(len(testset.data) / self.number_sub)
        te_subset = Subset(
            testset, range(self.sub_id * rows_by_sub, (self.sub_id + 1) * rows_by_sub)
        )

        # DataLoaders
        self.train_loader = DataLoader(
            syscall_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )
        self.val_loader = DataLoader(
            syscall_val,
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
                len(syscall_train), len(syscall_val), len(te_subset)
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
