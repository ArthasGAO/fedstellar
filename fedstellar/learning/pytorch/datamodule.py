#
# This file is part of the fedstellar framework (see https://github.com/enriquetomasmb/fedstellar).
# Copyright (c) 2023 Chao Feng.
#
import os
import sys
from math import floor

# To Avoid Crashes with a lot of nodes
import torch.multiprocessing
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import transforms
from torchvision.datasets import FashionMNIST
from fedstellar.learning.pytorch.changeablesubset import ChangeableSubset

torch.multiprocessing.set_sharing_strategy("file_system")
import pickle as pk


class DataModule(LightningDataModule):
    """
    LightningDataModule

    Args:
        sub_id: Subset id of partition. (0 <= sub_id < number_sub)
        number_sub: Number of subsets.
        batch_size: The batch size of the data.
        num_workers: The number of workers of the data.
        val_percent: The percentage of the validation set.
    """

    def __init__(
            self,
            train_set,
            test_set,
            sub_id=0,
            number_sub=1,
            batch_size=32,
            num_workers=4,
            val_percent=0.1,
            label_flipping=False,
            data_poisoning=False,
            poisoned_persent=0,
            poisoned_ratio=0,
            targeted=False,
            target_label=0,
            target_changed_label=0,
            noise_type="salt",
            indices_dir=None

    ):
        super().__init__()

        self.train_set = train_set
        self.test_set = test_set
        self.sub_id = sub_id
        self.number_sub = number_sub
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_percent = val_percent
        self.label_flipping = label_flipping
        self.data_poisoning = data_poisoning
        self.poisoned_percent = poisoned_persent
        self.poisoned_ratio = poisoned_ratio
        self.targeted = targeted
        self.target_label = target_label
        self.target_changed_label = target_changed_label
        self.noise_type = noise_type,
        self.indices_dir = indices_dir

        if self.sub_id + 1 > self.number_sub:
            raise ("Not exist the subset {}".format(self.sub_id))

        # Training / validation set
        rows_by_sub = floor(len(train_set) / self.number_sub)
        tr_subset = ChangeableSubset(
            train_set, range(self.sub_id * rows_by_sub, (self.sub_id + 1) * rows_by_sub), label_flipping=self.label_flipping, data_poisoning=self.data_poisoning, poisoned_persent=self.poisoned_percent, poisoned_ratio=self.poisoned_ratio, targeted=self.targeted, target_label=self.target_label,
            target_changed_label=self.target_changed_label, noise_type=self.noise_type
        )
        data_train, data_val = random_split(
            tr_subset,
            [
                round(len(tr_subset) * (1 - self.val_percent)),
                round(len(tr_subset) * self.val_percent),
            ],
        )

        # Test set
        rows_by_sub = floor(len(test_set) / self.number_sub)
        te_subset = ChangeableSubset(
            test_set, range(self.sub_id * rows_by_sub, (self.sub_id + 1) * rows_by_sub)
        )

        if len(test_set) < self.number_sub:
            raise "Too much partitions"

        # Save indices to local files
        train_indices_filename = f"{self.indices_dir}/participant_{self.sub_id}_train_indices.pk"
        valid_indices_filename = f"{self.indices_dir}/participant_{self.sub_id}_valid_indices.pk"
        test_indices_filename = f"{self.indices_dir}/participant_{self.sub_id}_test_indices.pk"

        with open(train_indices_filename, 'wb') as f:
            pk.dump(data_train.indices, f)
            f.close()
        with open(valid_indices_filename, 'wb') as f:
            pk.dump(data_val.indices, f)
            f.close()
        with open(test_indices_filename, 'wb') as f:
            pk.dump(te_subset.indices, f)
            f.close()

        # DataLoaders
        self.train_loader = DataLoader(
            data_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=False,
        )
        self.val_loader = DataLoader(
            data_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=False,
        )
        self.test_loader = DataLoader(
            te_subset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=False,
        )
        print(
            "Train: {} Val:{} Test:{}".format(
                len(data_train), len(data_val), len(te_subset)
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
