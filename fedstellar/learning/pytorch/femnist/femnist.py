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