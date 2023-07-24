import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.use('Agg')
import dataframe_image as dfi
from attacks import helper
import os
from pathlib import Path
from scripts import css_helper
pd.set_option('display.max_columns', None)

N_ROUNDS = 11

fmnist_b = pd.read_csv('data/wandb_export_2023-07-12_baseline_fashionmnist_loss.csv')
mnist_b = pd.read_csv('data/wandb_export_2023-07-12_baseline_mnist_loss.csv')
cifar10_b = pd.read_csv('data/wandb_export_2023-07-12_baseline_cifar10_loss.csv')
datasets = [mnist_b, fmnist_b, cifar10_b]
dataset_names = ["MNIST", "FMNIST", "CIFAR10"]
aggregation_list = ["Krum", "FedAvg", "TrimmedMean", "FlTrust", "Sentinel", "SentinelGlobal"]
aggregation_list.sort()
print(aggregation_list)
print(dataset_names)

overview_loss_mean = pd.DataFrame(data=None, index=aggregation_list, columns=dataset_names, dtype=None, copy=None)
for i, dataset in enumerate(datasets):
    ds_noerr = dataset[dataset.columns.drop(list(dataset.filter(regex='__MAX')))]
    ds_noerr = ds_noerr[ds_noerr.columns.drop(list(ds_noerr.filter(regex='__MIN')))]
    tmp = pd.DataFrame(index=range(0,N_ROUNDS), columns=aggregation_list)
    for aggregator in aggregation_list:
        vals = ds_noerr.filter(regex=(f'{aggregator}_')).dropna()
        vals = vals.reset_index().iloc[:, 1]
        tmp[aggregator] = vals
        assert len(vals) == N_ROUNDS
    tmp = tmp.reindex(sorted(tmp.columns), axis=1)
    tmp = tmp.transpose()
    overview_loss_mean[dataset_names[i]] = tmp[10] # 10th round

overview_loss_error = pd.DataFrame(data=None, index=aggregation_list, columns=dataset_names, dtype=None, copy=None)
for i, dataset in enumerate(datasets):
    for aggregator in aggregation_list:
        ds_agg = dataset.filter(regex=(f'{aggregator}_'))
        ds_max_agg_name = ds_agg.filter(regex='__MAX').columns[0]
        ds_min_agg_name = ds_agg.filter(regex='__MIN').columns[0]
        dataset[f'{aggregator}_ERROR'] = (ds_agg[ds_max_agg_name] - ds_agg[ds_min_agg_name])/2

    ds_noerr = dataset[dataset.columns.drop(list(dataset.filter(regex='__MAX')))]
    ds_noerr = ds_noerr[ds_noerr.columns.drop(list(ds_noerr.filter(regex='__MIN')))]
    tmp = pd.DataFrame(index=range(0,N_ROUNDS), columns=aggregation_list)
    for aggregator in aggregation_list:
        vals = ds_noerr.filter(regex=(f'{aggregator}_ERROR')).dropna()
        vals = vals.reset_index().iloc[:, 1]
        tmp[aggregator] = vals
        assert len(vals) == N_ROUNDS
    tmp = tmp.reindex(sorted(tmp.columns), axis=1)
    tmp = tmp.transpose()
    overview_loss_error[dataset_names[i]] = tmp[10] # 10th round

overview_loss_mean.style.format(precision=3)
overview_loss_error.style.format(precision=3)
overview_loss_mean.index.name = None
overview_loss_error.index.name = None
overview_loss = overview_loss_mean.copy()
for c in overview_loss.columns:
    overview_loss[c] = overview_loss[c].apply(lambda x: "{:.3f}".format(x) + "\u00B1")
    overview_loss[c] = overview_loss[c] + overview_loss_error[c].apply(lambda x: "{:.3f}".format(x))
print(overview_loss)

overview = overview_loss.sort_index() # alphabetically
styles = css_helper.get_table_styles()
overview_loss_style = overview_loss.style.set_table_styles(styles)
overview_loss_style = overview_loss_style.set_caption(f'Loss')
file = "figures/baseline_overview_loss.png"
dfi.export(overview_loss_style, file, dpi=600, fontsize=12)
print("Saved tabel to: " + file)