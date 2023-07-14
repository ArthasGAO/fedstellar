import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
matplotlib.use('Agg')
import dataframe_image as dfi
import os
from pathlib import Path
pd.set_option('display.max_columns', None)
FONT_SIZE = 12

def plot_baseline(dataset, metric):
    ROOT_DIR = Path.cwd()
    DATA_DIR = ROOT_DIR.joinpath('data')
    print(DATA_DIR)
    DATA_FILE = list(DATA_DIR.glob(f'*_{dataset}_{metric}.csv'))[0]
    print('Reading file: ')
    print(DATA_FILE)
    #DATA_DIR = ROOT_DIR.joinpath(dataset, attack_name, data_file)
    SAVE_PATH = ROOT_DIR.joinpath("figures")

    data = pd.read_csv(DATA_FILE)

    data = data[data.columns.drop(list(data.filter(regex='__MAX')))]
    data = data[data.columns.drop(list(data.filter(regex='__MIN')))]

    aggregation_list = ["Krum", "FedAvg", "TrimmedMean", "FlTrust", "Sentinel", "SentinelGlobal"]
    aggregation_list.sort()
    clean = pd.DataFrame(data=None, index=range(0,11), columns=aggregation_list, dtype=None, copy=None)

    round = range(0,11)
    clean['round'] = round


    for aggregator in aggregation_list:
        vals = data.filter(regex=(f'{aggregator}_')).dropna()
        vals = vals.reset_index().iloc[:, 1]
        clean[aggregator] = vals
    clean = clean.reindex(sorted(clean.columns), axis=1)

    marker_styles = ['+', 'x', 'o']
    line_styles = ['-', '--', '-.', ':', '-']
    fig = plt.figure()
    ax = None
    for i, aggregator in enumerate(aggregation_list):
        ax = clean.plot(x='round',
                        y=aggregator,
                        #yerr=error,
                        marker=marker_styles[i % len(marker_styles)],
                        markersize=3,
                        linestyle=line_styles[i % len(line_styles)],
                        linewidth=1,
                        label=aggregator,
                        ax=ax)
    plt.ylim(bottom=0)
    if metric == 'f1':
        plt.ylim(0.0, 1.01)
    plt.xlim(-0.5,10.5)
    # Set legend
    plt.legend(fontsize=FONT_SIZE)
    # Set grid
    plt.grid(True, linestyle='--', alpha=0.5)
    #ax.minorticks_on()
    #ax.grid(which='major', linestyle='-', linewidth='0.5', color='black')
    # Increase tick font sizes
    plt.xticks(fontsize=FONT_SIZE)
    plt.yticks(fontsize=FONT_SIZE)
    plt.xlabel("Round", fontsize=FONT_SIZE)
    plt.ylabel("Loss", fontsize=FONT_SIZE)
    if metric == 'f1':
        plt.ylabel("F1-Score", fontsize=FONT_SIZE)
    # plt.title(f'{dataset.upper()} Baseline, {metric.capitalize()}', fontsize=FONT_SIZE)
    file_name = f'baseline-{dataset}-{metric}.png'
    plt.savefig(os.path.join(SAVE_PATH, file_name), dpi=300, bbox_inches=0)
    plt.close('all')

def main():
    datasets = ["mnist", "fashionmnist", "cifar10"]
    metrics = ['f1','loss']  # lowercase
    for dataset in datasets:
        for metric in metrics:
            plot_baseline(dataset=dataset, metric=metric)

if __name__ == "__main__":
    main()



