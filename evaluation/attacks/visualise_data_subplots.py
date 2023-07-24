import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
# matplotlib.use('Agg')
import numpy as np
import dataframe_image as dfi
from attacks import helper
import os
import imgkit
from pathlib import Path
from html2image import Html2Image
from scripts import css_helper

hti = Html2Image()

FONT_SIZE = 16
pd.set_option('display.max_columns', None)
pd.set_option("display.precision", 3)


def generate_plot(_dataset: str, _attack: str, _agg_error):
    if _attack == 'label_flipping_targeted':
        metric = 'Test/ASR-targeted'
    elif _attack == 'sample_poison':
        metric = 'Test/ASR-backdoor'
    else:
        metric = 'Test/F1Score'

    metric_save = metric.replace("/", "_").replace("-", "_")
    prefix = f'{_dataset}_{_attack}_{metric_save}'

    ROOT_DIR = Path.cwd()
    DATA_DIR = ROOT_DIR.joinpath('data', _dataset, _attack)
    print(DATA_DIR)
    DATA_FILE = list(DATA_DIR.glob('*.csv'))[0]
    print('Reading file: ')
    print(DATA_FILE)
    # DATA_DIR = ROOT_DIR.joinpath(dataset, attack_name, data_file)
    #SAVE_PATH = DATA_DIR.joinpath("figures")

    SAVE_PATH = ROOT_DIR.joinpath("figures", _attack)

    print("Writing figures to: ")
    print(SAVE_PATH)
    isExist = os.path.exists(SAVE_PATH)
    if not isExist:
        os.makedirs(SAVE_PATH)

    raw_data = pd.read_csv(DATA_FILE)

    raw_data['adversarial_args.attack_env.poisoned_sample_percent'] = raw_data['Group'].str.extract(r'-S(.*)_R')
    raw_data['adversarial_args.attack_env.poisoned_sample_percent'].unique()
    fixed_data = raw_data.astype({'adversarial_args.attack_env.poisoned_sample_percent': 'int64'})
    # Sanity check
    print(fixed_data["adversarial_args.attack_env.poisoned_sample_percent"].unique())
    print(fixed_data["adversarial_args.attack_env.poisoned_node_percent"].unique())

    benign = fixed_data[fixed_data['adversarial_args.attacks'] == 'No Attack']
    df_unfinished = fixed_data[fixed_data['Round'] != 10]

    if len(df_unfinished) > 0:
        print("Not sentinel runs finished")
        print(df_unfinished)
        exit(0)

    mean_benign = pd.pivot_table(benign, index=["aggregator_args.algorithm",
                                                "adversarial_args.attack_env.attack",
                                                "adversarial_args.attack_env.poisoned_node_percent",
                                                "adversarial_args.attack_env.poisoned_sample_percent", ],
                                 values=["Test/Accuracy", "Test/F1Score", "Test/ASR-targeted", "Test/ASR-backdoor"],
                                 aggfunc=['mean', _agg_error], dropna=False)
    mean_benign = mean_benign.reset_index()
    num_attack_configs = len(mean_benign.index)

    print(num_attack_configs)

    # Generate Overview Tables
    columns = ["PNR", "PSR"]
    if _attack == 'model_poison':
        columns = ["PNR"]

    if _attack == 'label_flipping_targeted':
        metric_title = 'ASR-LF'
    elif _attack == 'sample_poison':
        metric_title = 'BA'
    else:
        metric_title = 'F1-Score'

    mean_benign_copy = mean_benign.rename(columns={"adversarial_args.attack_env.poisoned_node_percent": "PNR",
                                                   "adversarial_args.attack_env.poisoned_sample_percent": "PSR",
                                                   "aggregator_args.algorithm": "Aggregator"})
    overview_mean = mean_benign_copy.pivot(columns=columns, index=["Aggregator"], values=('mean', metric))
    overview_error = mean_benign_copy.pivot(columns=columns, index=["Aggregator"], values=(_agg_error, metric))

    overview_mean.style.format(precision=3)
    overview_error.style.format(precision=3)

    overview_mean.index.name = None
    overview_error.index.name = None

    # This is one of the hackiest solutions ever hacked, but it's research, and it works, so...
    overview_final = overview_mean.copy()
    print(overview_final)
    for c in overview_final.columns:
        overview_final[c] = overview_final[c].apply(lambda x: "{:.3f}".format(x) + "\u00B1")
        overview_final[c] = overview_final[c] + overview_error[c].apply(lambda x: "{:.3f}".format(x / 2))
    print(overview_final.head())

    overview_final_style = overview_final.style.format(precision=3)
    overview_final_style = overview_final_style.set_caption(f'{metric_title}')
    if metric in ["Test/Accuracy", "Test/F1Score"]:
        overview_final_style.highlight_max(props='font-weight: bold')
    else:
        overview_final_style.highlight_min(props='font-weight: bold')
    styles = css_helper.get_table_styles()
    overview_final_style = overview_final_style.set_table_styles(styles)
    overview_file_name = f'{prefix}_overview.png'
    dfi.export(overview_final_style, os.path.join(SAVE_PATH, overview_file_name), dpi=600, fontsize=12)

    # Generate Multiline Plots
    if _attack != 'model_poison':
        fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(21, 7), sharex="all", sharey="all")
    else:
        fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(7, 7), sharex="all", sharey="all")
    sample_percents = mean_benign['adversarial_args.attack_env.poisoned_sample_percent'].unique()
    for i, percent in enumerate(sample_percents):
        data_by_percent = mean_benign[mean_benign['adversarial_args.attack_env.poisoned_sample_percent'] == percent]
        ax_percent = helper.get_plot_by_node_percent(data=data_by_percent, y_col=('mean', metric),
                                                     y_err=(_agg_error, metric),
                                                     ax=axs[i] if _attack != 'model_poison' else axs,
                                                     font_size=FONT_SIZE)
        if _attack != 'model_poison':
            axs[i].set_title(f'PSR: {percent}', fontsize=FONT_SIZE)
            handles, labels = axs[0].get_legend_handles_labels()
        else:
            axs.set_title(f'Noise Type: salt', fontsize=FONT_SIZE)
            handles, labels = axs.get_legend_handles_labels()

    fig.legend(handles=handles, labels=labels, loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=3, fancybox=True,
               shadow=False, fontsize=FONT_SIZE)
    fig.subplots_adjust(wspace=0.05, hspace=0)
    file_name = f'{prefix}.pdf'
    fig.savefig(os.path.join(SAVE_PATH, file_name), dpi=600, bbox_inches='tight')
    # plt.close('sentinel')


def main():
    datasets = ["fmnist", "mnist", "cifar10"]
    # datasets = ["fmnist"]
    attack_names = ["label_flipping_targeted", "label_flipping_untargeted", "sample_poison", "model_poison"]
    # attack_names = ["sample_poison", "model_poison"]
    # attack_names = ["model_poison"]
    for dataset in datasets:
        for attack_name in attack_names:
            generate_plot(_dataset=dataset, _attack=attack_name, _agg_error='sem')


if __name__ == "__main__":
    main()
