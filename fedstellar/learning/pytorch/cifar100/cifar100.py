# 
# This file is part of the fedstellar framework (see https://github.com/enriquetomasmb/fedstellar).
# Copyright (c) 2023 Chao Feng.
#
import os
import sys
from math import floor

# To Avoid Crashes with a lot of nodes
import torch.multiprocessing
from torch import tensor
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import transforms
from torchvision.datasets import CIFAR100
from fedstellar.learning.pytorch.changeablesubset import ChangeableSubset
torch.multiprocessing.set_sharing_strategy("file_system")
import pickle as pk

#######################################
#  FederatedDataModule for CIFAR100  #
#######################################


class CIFAR100DATASET():
    """
    Down the CIFAR10 datasets from torchversion.

    Args:
    iid: iid or non-iid data seperate
    """
    def __init__(self, iid=True):
        self.trainset = None
        self.testset = None
        self.iid = iid

        # Singletons of CIFAR100 train and test datasets
        if not os.path.exists(f"{sys.path[0]}/data"):
            os.makedirs(f"{sys.path[0]}/data", exist_ok=True)
        
        self.trainset = CIFAR100(
                f"{sys.path[0]}/data", train=True, download=True, transform=transforms.ToTensor()
                )
        self.testset = CIFAR100(
                f"{sys.path[0]}/data", train=False, download=True, transform=transforms.ToTensor()
                )

        self.trainset.targets = tensor(self.trainset.targets)
        self.testset.targets = tensor(self.testset.targets)

        if not self.iid:
            # if non-iid, sort the dataset
            self.trainset = self.sort_dataset(self.trainset)
            self.testset = self.sort_dataset(self.testset)

    def sort_dataset(self, dataset):
        sorted_indexes = dataset.targets.sort()[1]
        dataset.targets = (dataset.targets[sorted_indexes])
        dataset.data = dataset.data[sorted_indexes]
        return dataset
