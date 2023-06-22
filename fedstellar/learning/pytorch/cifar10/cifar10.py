# 
# This file is part of the fedstellar framework (see https://github.com/enriquetomasmb/fedstellar).
# Copyright (c) 2023 Enrique Tomás Martínez Beltrán.
#
import os
import sys

# To Avoid Crashes with a lot of nodes
import torch.multiprocessing
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from torchvision.datasets import CIFAR10

torch.multiprocessing.set_sharing_strategy("file_system")


class CIFAR10Dataset(Dataset):
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
    cifar10_train = None
    cifar10_val = None

    def __init__(
            self,
            normalization="cifar10",
            loading="torchvision",
            sub_id=0,
            number_sub=1,
            num_workers=2,
            batch_size=32,
            iid=True,
    ):
        super().__init__()
        self.train_set = None
        self.test_set = None
        self.sub_id = sub_id
        self.number_sub = number_sub
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.iid = iid
        self.loading = loading
        self.normalization = normalization
        self.mean = self.set_normalization(normalization)["mean"]
        self.std = self.set_normalization(normalization)["std"]

        # Singletons of MNIST train and test datasets
        if not os.path.exists(f"{sys.path[0]}/data"):
            os.makedirs(f"{sys.path[0]}/data")

        if CIFAR10Dataset.cifar10_train is None:
            CIFAR10Dataset.cifar10_train = CIFAR10(
                f"{sys.path[0]}/data", train=True, download=True, transform=T.Compose(
                    [
                        T.RandomCrop(32, padding=4),
                        T.RandomHorizontalFlip(),
                        T.ToTensor(),
                        T.Normalize(self.mean, self.std),
                    ])
            )
            if not iid:
                sorted_indexes = CIFAR10Dataset.cifar10_train.targets.sort()[1]
                CIFAR10Dataset.cifar10_train.targets = (
                    CIFAR10Dataset.cifar10_train.targets[sorted_indexes]
                )
                CIFAR10Dataset.cifar10_train.data = CIFAR10Dataset.cifar10_train.data[
                    sorted_indexes
                ]
        if CIFAR10Dataset.cifar10_val is None:
            CIFAR10Dataset.cifar10_val = CIFAR10(
                f"{sys.path[0]}/data", train=False, download=True, transform=T.Compose(
                    [
                        T.ToTensor(),
                        T.Normalize(self.mean, self.std),
                    ])
            )
            if not iid:
                sorted_indexes = CIFAR10Dataset.cifar10_val.targets.sort()[1]
                CIFAR10Dataset.cifar10_val.targets = CIFAR10Dataset.cifar10_val.targets[
                    sorted_indexes
                ]
                CIFAR10Dataset.cifar10_val.data = CIFAR10Dataset.cifar10_val.data[
                    sorted_indexes
                ]
        if self.sub_id + 1 > self.number_sub:
            raise ("Not exist the subset {}".format(self.sub_id))

        self.train_set = CIFAR10Dataset.cifar10_train
        self.test_set = CIFAR10Dataset.cifar10_val

        if not self.iid:
            # if non-iid, sort the dataset
            self.train_set = self.sort_dataset(self.train_set)
            self.test_set = self.sort_dataset(self.test_set)

    def sort_dataset(self, dataset):
        sorted_indexes = dataset.targets.sort()[1]
        dataset.targets = (dataset.targets[sorted_indexes])
        dataset.data = dataset.data[sorted_indexes]
        return dataset

    def set_normalization(self, normalization):
        # Image classification on the CIFAR10 dataset
        # Albumentations Documentation https://albumentations.ai/docs/autoalbument/examples/cifar10/
        if normalization == "cifar10":
            mean = (0.4914, 0.4822, 0.4465)
            std = (0.2471, 0.2435, 0.2616)
        elif normalization == "imagenet":
            # ImageNet - torchbench Docs https://paperswithcode.github.io/torchbench/imagenet/
            mean = (0.485, 0.456, 0.406)
            std = (0.229, 0.224, 0.225)
        else:
            raise NotImplementedError
        return {"mean": mean, "std": std}
