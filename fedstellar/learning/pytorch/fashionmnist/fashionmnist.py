# 
# This file is part of the fedstellar framework (see https://github.com/enriquetomasmb/fedstellar).
# Copyright (c) 2023 Chao Feng.
#
import os
import sys
from math import floor

# To Avoid Crashes with a lot of nodes
import torch.multiprocessing
from torchvision import transforms
from torchvision.datasets import FashionMNIST
from fedstellar.learning.pytorch.changeablesubset import ChangeableSubset
torch.multiprocessing.set_sharing_strategy("file_system")
import pickle as pk

#######################################
#  FederatedDataModule for FashionMNIST  #
#######################################


class FASHIONMNISTDATASET():
    """
    Down the FashionMNIST datasets from torchversion.

    Args:
    iid: iid or non-iid data seperate
    """
    def __init__(self, iid=True):
        self.trainset = None
        self.testset = None
        self.iid = iid

        # Singletons of FashionMNIST train and test datasets
        if not os.path.exists(f"{sys.path[0]}/data"):
            os.makedirs(f"{sys.path[0]}/data", exist_ok=True)
        
        self.trainset = FashionMNIST(
                f"{sys.path[0]}/data", train=True, download=True, transform=transforms.ToTensor()
                )
        self.testset = FashionMNIST(
                f"{sys.path[0]}/data", train=False, download=True, transform=transforms.ToTensor()
                )

        if not self.iid:
            # if non-iid, sort the dataset
            self.trainset = self.sort_dataset(self.trainset)
            self.testset = self.sort_dataset(self.testset)

    def sort_dataset(self, dataset):
        sorted_indexes = dataset.targets.sort()[1]
        dataset.targets = (dataset.targets[sorted_indexes])
        dataset.data = dataset.data[sorted_indexes]
        return dataset
