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
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Subset, random_split, Dataset
from torchvision.datasets import MNIST, utils
from torchvision import transforms
import numpy as np
from fedstellar.learning.pytorch.changeablesubset import ChangeableSubset
torch.multiprocessing.set_sharing_strategy("file_system")


class FEMNIST(MNIST):
    def __init__(self, train=True, transform=None, target_transform=None, download=True):
        self.root = f"{sys.path[0]}/data"
        super(MNIST, self).__init__(self.root, transform=transform, target_transform=target_transform)

        self.download = download
        self.download_link = 'https://media.githubusercontent.com/media/GwenLegate/femnist-dataset-PyTorch/main/femnist.tar.gz'
        self.file_md5 = '60433bc62a9bff266244189ad497e2d7'
        self.train = train
        
        self.training_file = f'{self.root}/FEMNIST/processed/femnist_train.pt'
        self.test_file = f'{self.root}/FEMNIST/processed/femnist_test.pt'

        if not os.path.exists(f'{self.root}/FEMNIST/processed/femnist_test.pt') or not os.path.exists(f'{self.root}/FEMNIST/processed/femnist_train.pt'):
            if self.download:
                self.dataset_download()
            else:
                raise RuntimeError('Dataset not found, set parameter download=True to download')

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
                os.makedirs(path, exist_ok=True)

        # download files
        filename = self.download_link.split('/')[-1]
        utils.download_and_extract_archive(self.download_link, download_root=f'{self.root}/FEMNIST/raw/', filename=filename, md5=self.file_md5)

        files = ['femnist_train.pt', 'femnist_test.pt']
        for file in files:
            # move to processed dir
            shutil.move(os.path.join(f'{self.root}/FEMNIST/raw/', file), f'{self.root}/FEMNIST/processed/')

class FEMNISTDATASET():
    """
    Down the FEMNIST datasets

    Args:
    iid: iid or non-iid data seperate
    """
    def __init__(self, iid=True):
        self.trainset = None
        self.testset = None
        self.iid = iid

        data_path = f"{sys.path[0]}/data/FEMNIST/"

        transform_data = transforms.Compose(
            [
                # transforms.CenterCrop((96, 96)),
                # transforms.Grayscale(num_output_channels=1),
                # transforms.Resize((28, 28)),
                # transforms.ColorJitter(contrast=3),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))]
        )

        self.trainset = FEMNIST(train=True, transform=transform_data)
        self.testset = FEMNIST(train=False, transform=transform_data)

        if not self.iid:
            # if non-iid, sort the dataset
            self.trainset = self.sort_dataset(self.trainset)
            self.testset = self.sort_dataset(self.testset)

    def sort_dataset(self, dataset):
        sorted_indexes = dataset.targets.sort()[1]
        dataset.targets = (dataset.targets[sorted_indexes])
        dataset.data = dataset.data[sorted_indexes]
        return dataset


#######################################
#    FEMNISTDataModule for FEMNIST    #
#######################################


# class FEMNISTDataModule(LightningDataModule):
#     """
#     LightningDataModule of partitioned FEMNIST.

#     This dataset is derived from the Leaf repository
#     (https://github.com/TalwalkarLab/leaf) pre-processing of the Extended MNIST
#     dataset, grouping examples by writer. Details about Leaf were published in
#     "LEAF: A Benchmark for Federated Settings" https://arxiv.org/abs/1812.01097.

#     The FEMNIST dataset is naturally non-iid

#     IMPORTANT: The data is generated using ./preprocess.sh -s niid --sf 0.05 -k 0 -t sample (small-sized dataset)

#     Args:

#     """

#     # Singleton
#     femnist_train = None
#     femnist_val = None

#     def __init__(
#             self,
#             sub_id=0,
#             number_sub=1,
#             batch_size=32,
#             num_workers=8,
#             val_percent=0.1,
#             iid=True,
#             root_dir=None,
#             label_flipping=False,
#             data_poisoning=False,
#             poisoned_persent=0,
#             poisoned_ratio=0,
#             targeted=False,
#             target_label=0,
#             target_changed_label=0,
#             noise_type="salt",
#             indices_dir=None
#     ):
#         super().__init__()
#         self.sub_id = sub_id
#         self.number_sub = number_sub
#         self.batch_size = batch_size
#         self.num_workers = num_workers
#         self.val_percent = val_percent
#         self.idd = iid
#         self.root_dir = root_dir,
#         self.label_flipping = label_flipping
#         self.data_poisoning = data_poisoning
#         self.poisoned_persent = poisoned_persent
#         self.poisoned_ratio = poisoned_ratio
#         self.targeted = targeted
#         self.target_label = target_label
#         self.target_changed_label = target_changed_label
#         self.noise_type = noise_type,
#         self.indices_dir = indices_dir


#         transform_data = transforms.Compose(
#             [
#                 # transforms.CenterCrop((96, 96)),
#                 # transforms.Grayscale(num_output_channels=1),
#                 # transforms.Resize((28, 28)),
#                 # transforms.ColorJitter(contrast=3),
#                 transforms.ToTensor(),
#                 transforms.Normalize((0.5,), (0.5,))]
#         )

#         self.train = FEMNIST(sub_id=self.sub_id, number_sub=self.number_sub, root_dir=root_dir, train=True, transform=transform_data, target_transform=None, download=True)
#         self.test = FEMNIST(sub_id=self.sub_id, number_sub=self.number_sub, root_dir=root_dir, train=False, transform=transform_data, target_transform=None, download=True)


#         if iid:
#             train_len = len(self.train.targets)
#             train_index = np.arange(train_len)
#             np.random.shuffle(train_index)

#             test_len = len(self.test.targets)
#             test_index = np.arange(test_len)
#             np.random.shuffle(test_index)

#             self.train.data=self.train.data[train_index]
#             self.train.targets=self.train.targets[train_index]

#             self.test.data=self.test.data[test_index]
#             self.test.targets=self.test.targets[test_index]
            
#         if not iid:
#             train_sorted_index = self.train.targets.sort()[1]

#             test_sorted_index = self.test.targets.sort()[1]

#             self.train.data=self.train.data[train_sorted_index]
#             self.train.targets=self.train.targets[train_sorted_index]

#             self.test.data=self.test.data[test_sorted_index]
#             self.test.targets=self.test.targets[test_sorted_index]


#         if len(self.test) < self.number_sub:
#             raise ("Too much partitions")

#         # Training / validation set
#         trainset = self.train
#         rows_by_sub = floor(len(trainset) / self.number_sub)

#         tr_subset = ChangeableSubset(
#             trainset, range(self.sub_id * rows_by_sub, (self.sub_id + 1) * rows_by_sub), \
#             label_flipping=self.label_flipping, data_poisoning=self.data_poisoning, poisoned_persent=self.poisoned_persent, \
#             poisoned_ratio=self.poisoned_ratio, targeted=self.targeted, target_label=self.target_label, \
#             target_changed_label=self.target_changed_label, noise_type=self.noise_type
#         )

#         femnist_train, femnist_val = random_split(
#             tr_subset,
#             [
#                 round(len(tr_subset) * (1 - self.val_percent)),
#                 round(len(tr_subset) * self.val_percent),
#             ],
#         )

#         # Test set
#         testset = self.test
#         rows_by_sub = floor(len(testset) / self.number_sub)
#         te_subset = ChangeableSubset(
#             testset, range(self.sub_id * rows_by_sub, (self.sub_id + 1) * rows_by_sub)
#         )


#         # DataLoaders
#         self.train_loader = DataLoader(
#             femnist_train,
#             batch_size=self.batch_size,
#             shuffle=True,
#             num_workers=self.num_workers,
#         )
#         self.val_loader = DataLoader(
#             femnist_val,
#             batch_size=self.batch_size,
#             shuffle=False,
#             num_workers=self.num_workers,
#         )
#         self.test_loader = DataLoader(
#             te_subset,
#             batch_size=self.batch_size,
#             shuffle=False,
#             num_workers=self.num_workers,
#         )
#         print(
#             "Train: {} Val:{} Test:{}".format(
#                 len(femnist_train), len(femnist_val), len(testset)
#             )
#         )

#     def train_dataloader(self):
#         """ """
#         return self.train_loader

#     def val_dataloader(self):
#         """ """
#         return self.val_loader

#     def test_dataloader(self):
#         """ """
#         return self.test_loader
