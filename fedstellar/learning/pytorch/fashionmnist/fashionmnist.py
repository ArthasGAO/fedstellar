# 
# This file is part of the fedstellar framework (see https://github.com/enriquetomasmb/fedstellar).
# Copyright (c) 2023 Enrique Tomás Martínez Beltrán.
#
import os
import sys

# To Avoid Crashes with a lot of nodes
import torch.multiprocessing
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import FashionMNIST


torch.multiprocessing.set_sharing_strategy("file_system")


#######################################
#    FederatedDataModule for FashionMNIST    #
#######################################


class FashionMNISTDataset(Dataset):
    """
    LightningDataModule of partitioned MNIST.

    Args:
        sub_id: Subset id of partition. (0 <= sub_id < number_sub)
        number_sub: Number of subsets.
        batch_size: The batch size of the data.
        num_workers: The number of workers of the data.
        val_percent: The percentage of the validation set.
    """

    # Singleton
    fmnist_train = None
    fmnist_val = None

    def __init__(
            self,
            sub_id=0,
            number_sub=1,
            batch_size=32,
            num_workers=4,
            val_percent=0.1,
            iid=True,
    ):
        super().__init__()
        self.train_set = None
        self.test_set = None
        self.sub_id = sub_id
        self.number_sub = number_sub
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_percent = val_percent
        self.iid = iid

        # Singletons of MNIST train and test datasets
        if not os.path.exists(f"{sys.path[0]}/data"):
            os.makedirs(f"{sys.path[0]}/data")

        if FashionMNISTDataset.fmnist_train is None:
            FashionMNISTDataset.fmnist_train = FashionMNIST(
                f"{sys.path[0]}/data", train=True, download=True, transform=transforms.ToTensor()
            )
            if not iid:
                sorted_indexes = FashionMNISTDataset.fmnist_train.targets.sort()[1]
                FashionMNISTDataset.fmnist_train.targets = (
                    FashionMNISTDataset.fmnist_train.targets[sorted_indexes]
                )
                FashionMNISTDataset.fmnist_train.data = FashionMNISTDataset.fmnist_train.data[
                    sorted_indexes
                ]
        if FashionMNISTDataset.fmnist_val is None:
            FashionMNISTDataset.fmnist_val = FashionMNIST(
                f"{sys.path[0]}/data", train=False, download=True, transform=transforms.ToTensor()
            )
            if not iid:
                sorted_indexes = FashionMNISTDataset.fmnist_val.targets.sort()[1]
                FashionMNISTDataset.fmnist_val.targets = FashionMNISTDataset.fmnist_val.targets[
                    sorted_indexes
                ]
                FashionMNISTDataset.fmnist_val.data = FashionMNISTDataset.fmnist_val.data[
                    sorted_indexes
                ]
        if self.sub_id + 1 > self.number_sub:
            raise ("Not exist the subset {}".format(self.sub_id))

        self.train_set = FashionMNISTDataset.fmnist_train
        self.test_set = FashionMNISTDataset.fmnist_val

        if not self.iid:
            # if non-iid, sort the dataset
            self.train_set = self.sort_dataset(self.train_set)
            self.test_set = self.sort_dataset(self.test_set)

    def sort_dataset(self, dataset):
        sorted_indexes = dataset.targets.sort()[1]
        dataset.targets = (dataset.targets[sorted_indexes])
        dataset.data = dataset.data[sorted_indexes]
        return dataset