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
from datasets import load_dataset
torch.multiprocessing.set_sharing_strategy("file_system")
import pickle as pk
from torchvision.datasets import MNIST, utils
from sklearn.model_selection import train_test_split
from torchtext import vocab
import pandas as pd
from torch.nn.functional import pad
from nltk.corpus import stopwords
from string import punctuation
import random
import numpy as np
import ast
import shutil
import zipfile


#######################################
#  FederatedDataModule for syscall  #
#######################################

class SYSCALL(MNIST):
    def __init__(self, train=True):
        self.root = f"{sys.path[0]}/data"
        self.download = True
        self.train = train
        super(MNIST, self).__init__(self.root)
        self.training_file = f'{self.root}/syscall/processed/syscall_train.pt'
        self.test_file = f'{self.root}/syscall/processed/syscall_test.pt'

        if not os.path.exists(f'{self.root}/syscall/processed/syscall_test.pt') or not os.path.exists(f'{self.root}/syscall/processed/syscall_train.pt'):
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
        # self.class_to_idx = data_and_targets[2]
        # self.classes = data_and_targets[3]

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

        # download data
        data_link = 'https://files.ifi.uzh.ch/CSG/research/fl/data/syscall.zip'
        filename = data_link.split('/')[-1]

        utils.download_and_extract_archive(data_link, download_root=f'{self.root}/syscall/raw/', filename=filename)

        with zipfile.ZipFile(f'{self.root}/syscall/raw/{filename}', 'r') as zip_ref:
            zip_ref.extractall(f'{self.root}/syscall/raw/')


        df = pd.DataFrame()
        files = os.listdir(f'{self.root}/syscall/raw/')
        cols = ["ids", "maltype", "system calls frequency_1gram-scaled"]
        for f in files: 
            if '.csv' in f:
                feature_name = 'system calls frequency_1gram-scaled'
                fi_path= f'{self.root}/syscall/raw/{f}'
                csv_df = pd.read_csv(fi_path, sep='\t')
                feature = [ast.literal_eval(i) for i in csv_df[feature_name]]
                csv_df[feature_name] = feature
                df = pd.concat([df, csv_df])
        df['maltype'] = df['maltype'].replace(to_replace='normalv2',value='normal')
        classes_to_targets = {}
        t = 0
        for i in set(df['maltype']):
            classes_to_targets[i] = t 
            t += 1
        classes = list(classes_to_targets.keys())

        for c in classes_to_targets:
            df['maltype'] = df['maltype'].replace(to_replace=c,value=classes_to_targets[c])
        
        all_targes = torch.tensor(df['maltype'].tolist())
        all_data = torch.tensor(df[feature_name].tolist())

        X_train, X_test, y_train, y_test = train_test_split(all_data, all_targes, test_size=0.15, random_state=42)
        train = [X_train, y_train, classes_to_targets, classes]
        test = [X_test, y_test, classes_to_targets, classes ]

        # save to files
        train_file = f'{self.root}/syscall/processed/syscall_train.pt'
        test_file = f'{self.root}/syscall/processed/syscall_test.pt'

        # save to processed dir            
        if not os.path.exists(train_file):
            torch.save(train, train_file)
        if not os.path.exists(test_file):
            torch.save(test, test_file)


class SYSCALLDATASET():
    """
    Down the syscall datasets

    Args:
    iid: iid or non-iid data seperate
    """
    def __init__(self, iid=True):
        self.trainset = None
        self.testset = None
        self.iid = iid

        data_path = f"{sys.path[0]}/data/syscall/"

        self.trainset = SYSCALL(train=True)
        self.testset = SYSCALL(train=False)

        if not self.iid:
            # if non-iid, sort the dataset
            self.trainset = self.sort_dataset(self.trainset)
            self.testset = self.sort_dataset(self.testset)

    def sort_dataset(self, dataset):
        sorted_indexes = dataset.targets.sort()[1]
        dataset.targets = (dataset.targets[sorted_indexes])
        dataset.data = dataset.data[sorted_indexes]
        return dataset
