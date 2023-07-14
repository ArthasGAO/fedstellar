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

fmnist_b = pd.read_csv('data/wandb_export_2023-07-12_baseline_fashionmnist_f1.csv')
mnist_b = pd.read_csv('data/wandb_export_2023-07-12_baseline_mnist_f1.csv')
cifar10_b = pd.read_csv('data/wandb_export_2023-07-12_baseline_cifar10_f1.csv')

datasets = [mnist_b, fmnist_b, cifar10_b]

N_ROUNDS = 11

aggregation_list = ["Krum", "FedAvg", "TrimmedMean", "FlTrust", "Sentinel", "SentinelGlobal"]
aggregation_list.sort()
dataset_list = ["MNIST", "FMNIST", "CIFAR10"]
print(aggregation_list)
print(dataset_list)
overview_error = pd.DataFrame(data=None, index=aggregation_list, columns=dataset_list, dtype=None, copy=None)

for aggregator in aggregation_list:
    mnist_b_agg = mnist_b.filter(regex=(f'{aggregator}_'))
    mnist_max_agg_name = mnist_b_agg.filter(regex='__MAX').columns[0]
    mnist_min_agg_name = mnist_b_agg.filter(regex='__MIN').columns[0]
    mnist_b[f'{aggregator}_ERROR'] = mnist_b_agg[mnist_max_agg_name] - mnist_b_agg[mnist_min_agg_name]

for aggregator in aggregation_list:
    fmnist_b_agg = fmnist_b.filter(regex=(f'{aggregator}_'))
    fmnist_max_agg_name = fmnist_b_agg.filter(regex='__MAX').columns[0]
    fmnist_min_agg_name = fmnist_b_agg.filter(regex='__MIN').columns[0]
    fmnist_b[f'{aggregator}_ERROR'] = fmnist_b[fmnist_max_agg_name] - fmnist_b[fmnist_min_agg_name]

for aggregator in aggregation_list:
    cifar10_b_agg = cifar10_b.filter(regex=(f'{aggregator}_'))
    cifar10_max_agg_name = cifar10_b_agg.filter(regex='__MAX').columns[0]
    cifar10_min_agg_name = cifar10_b_agg.filter(regex='__MIN').columns[0]
    cifar10_b[f'{aggregator}_ERROR'] = cifar10_b[cifar10_max_agg_name] - cifar10_b[cifar10_min_agg_name]

mnist_f1 = mnist_b[mnist_b.columns.drop(list(mnist_b.filter(regex='__MAX')))]
mnist_f1 = mnist_f1[mnist_f1.columns.drop(list(mnist_f1.filter(regex='__MIN')))]

tmp = pd.DataFrame(index=range(0,N_ROUNDS), columns=aggregation_list)
for aggregator in aggregation_list:
    vals = mnist_f1.filter(regex=(f'{aggregator}_ERROR')).dropna()
    vals = vals.reset_index().iloc[:, 1]
    assert len(vals) == N_ROUNDS
    tmp[aggregator] = vals
tmp = tmp.reindex(sorted(tmp.columns), axis=1)
tmp = tmp.transpose()
overview_error['MNIST'] = tmp[10]

fmnist_f1 = fmnist_b[fmnist_b.columns.drop(list(fmnist_b.filter(regex='__MAX')))]
fmnist_f1 = fmnist_b[fmnist_b.columns.drop(list(fmnist_b.filter(regex='__MIN')))]

tmp = pd.DataFrame(index=range(0,N_ROUNDS), columns=aggregation_list)
for aggregator in aggregation_list:
    vals = fmnist_b.filter(regex=(f'{aggregator}_ERROR')).dropna()
    vals = vals.reset_index().iloc[:, 1]
    tmp[aggregator] = vals
    assert len(vals) == N_ROUNDS
tmp = tmp.reindex(sorted(tmp.columns), axis=1)
tmp = tmp.transpose()
overview_error['FMNIST'] = tmp[10]

cifar10_f1 = cifar10_b[cifar10_b.columns.drop(list(cifar10_b.filter(regex='__MAX')))]
cifar10_f1 = cifar10_f1[cifar10_f1.columns.drop(list(cifar10_f1.filter(regex='__MIN')))]

tmp = pd.DataFrame(index=range(0,N_ROUNDS), columns=aggregation_list)
for aggregator in aggregation_list:
    vals = cifar10_f1.filter(regex=(f'{aggregator}_ERROR')).dropna()
    vals = vals.reset_index().iloc[:, 1]
    assert len(vals) == N_ROUNDS
    tmp[aggregator] = vals
tmp = tmp.reindex(sorted(tmp.columns), axis=1)
tmp = tmp.transpose()
overview_error['CIFAR10'] = tmp[10]

styles = [
    dict(selector="table", props=[
        ("font-family" , 'Arial'),
        ("margin" , "25px 100px"),
        ("border-collapse" , "collapse"),
        ("border" , "1px solid #eee"),
        ("border-bottom" , "2px solid #00cccc"),
    ]),
]

overview_style = overview_error.style.set_properties(**{'margin': '100px'})
overview_style.format(precision=3)
overview_style.highlight_min(props='font-weight: bold')
overview_style.set_table_styles(styles)
file = "figures/baseline_overview_f1_sem.png"
dfi.export(overview_style, file, dpi=300, fontsize=12)
print("Saved tabel to: " + file)
