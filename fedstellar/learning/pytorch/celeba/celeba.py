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
from torchvision.datasets import CelebA
from fedstellar.learning.pytorch.changeablesubset import ChangeableSubset
torch.multiprocessing.set_sharing_strategy("file_system")
import pickle as pk

#######################################
#  FederatedDataModule for CelebA  #
#######################################


class CelebADATASET():
    """
    Down the CelebA datasets from torchversion.

    Args:
    iid: iid or non-iid data seperate
    """
    def __init__(self, iid=True):
        self.trainset = BasicDataset()
        self.testset = BasicDataset()
        self.iid = iid

        # Singletons of CelebA train and test datasets
        if not os.path.exists(f"{sys.path[0]}/data"):
            os.makedirs(f"{sys.path[0]}/data", exist_ok=True)
        
        trainset = CelebA(
                f"{sys.path[0]}/data", split='train', download=True, transform=transforms.ToTensor()
                )
        valid =  CelebA(
                f"{sys.path[0]}/data", split='valid', download=True, transform=transforms.ToTensor()
                )
        testset =  CelebA(
                f"{sys.path[0]}/data", split='test', download=True, transform=transforms.ToTensor()
                )
        self.trainset.data = [i[0] for i in trainset]
        self.trainset.targets = [i[1] for i in trainset]

        self.valid.data = [i[0] for i in valid]
        self.valid.targets = [i[1] for i in valid]

        self.trainset.data = self.trainset.data+self.valid.data
        self.trainset.targets = self.trainset.targets+self.valid.targets

        self.trainset.data = tensor(self.trainset.data)
        self.trainset.targets = tensor(self.trainset.targets)

        self.testset.data = [i[0] for i in testset]
        self.testset.targets = [i[1] for i in testset]

        self.testset.data = tensor(self.testset.data)
        self.testset.targets = tensor(self.testset.targets)

        if not self.iid:
            # if non-iid, sort the dataset
            self.trainset = self.sort_dataset(self.trainset)
            self.testset = self.sort_dataset(self.testset)

    def sort_dataset(self, dataset):
        dataset.data = tensor(dataset.targets)
        sorted_indexes = dataset.targets.sort()[1]
        dataset.targets = (dataset.targets[sorted_indexes])
        dataset.data = dataset.data[sorted_indexes]
        return dataset
    

class BasicDataset():
    def __init__(self, data=None, targets=None):
        self.data = data
        self.targets = targets
