import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.use('Agg')
import dataframe_image as dfi
from attacks import helper
import os
from pathlib import Path
pd.set_option('display.max_columns', None)

fmnist_b = pd.read_csv('data/wandb_export_2023-07-12_baseline_fashionmnist_loss.csv')
mnist_b = pd.read_csv('data/wandb_export_2023-07-12_baseline_mnist_loss.csv')
cifar10_b = pd.read_csv('data/wandb_export_2023-07-12_baseline_cifar10_loss.csv')

N_ROUNDS = 11

aggregation_list = ["Krum", "FedAvg", "TrimmedMean", "FlTrust", "Sentinel", "SentinelGlobal"]
aggregation_list.sort()
dataset_list = ["MNIST", "FMNIST", "CIFAR10"]
print(aggregation_list)
print(dataset_list)
overview = pd.DataFrame(data=None, index=aggregation_list, columns=dataset_list, dtype=None, copy=None)

fmnist_loss = fmnist_b[fmnist_b.columns.drop(list(fmnist_b.filter(regex='__MAX')))]
fmnist_loss = fmnist_loss[fmnist_loss.columns.drop(list(fmnist_loss.filter(regex='__MIN')))]

tmp = pd.DataFrame(index=range(0,N_ROUNDS), columns=aggregation_list)
for aggregator in aggregation_list:
    vals = fmnist_b.filter(regex=(f'{aggregator}_')).dropna()
    vals = vals.reset_index().iloc[:, 1]
    tmp[aggregator] = vals
    assert len(vals) == N_ROUNDS
tmp = tmp.reindex(sorted(tmp.columns), axis=1)
tmp = tmp.transpose()
overview['FMNIST'] = tmp[10]

mnist_loss = mnist_b[mnist_b.columns.drop(list(mnist_b.filter(regex='__MAX')))]
mnist_loss = mnist_loss[mnist_loss.columns.drop(list(mnist_loss.filter(regex='__MIN')))]

tmp = pd.DataFrame(index=range(0,N_ROUNDS), columns=aggregation_list)
for aggregator in aggregation_list:
    vals = mnist_loss.filter(regex=(f'{aggregator}_')).dropna()
    vals = vals.reset_index().iloc[:, 1]
    assert len(vals) == N_ROUNDS
    tmp[aggregator] = vals
tmp = tmp.reindex(sorted(tmp.columns), axis=1)
tmp = tmp.transpose()
overview['MNIST'] = tmp[10]

cifar10_loss = cifar10_b[cifar10_b.columns.drop(list(cifar10_b.filter(regex='__MAX')))]
cifar10_loss = cifar10_loss[cifar10_loss.columns.drop(list(cifar10_loss.filter(regex='__MIN')))]

tmp = pd.DataFrame(index=range(0,N_ROUNDS), columns=aggregation_list)
for aggregator in aggregation_list:
    vals = cifar10_loss.filter(regex=(f'{aggregator}_')).dropna()
    vals = vals.reset_index().iloc[:, 1]
    assert len(vals) == N_ROUNDS
    tmp[aggregator] = vals
tmp = tmp.reindex(sorted(tmp.columns), axis=1)
tmp = tmp.transpose()
overview['CIFAR10'] = tmp[10]

overview = overview.sort_index() # alphabetically

styles = [
    dict(selector="table", props=[
        ("font-family" , 'Arial'),
        ("margin" , "25px 100px"),
        ("border-collapse" , "collapse"),
        ("border" , "1px solid #eee"),
        ("border-bottom" , "2px solid #00cccc"),
    ]),
]

overview_style = overview.style.set_properties(**{'margin': '100px'})
overview_style.format(precision=3)
overview_style.highlight_min(props='font-weight: bold')
overview_style.set_table_styles(styles)
file = "figures/baseline_overview_loss.png"
dfi.export(overview_style, file, dpi=300, fontsize=12)
print("Saved tabel to: " + file)