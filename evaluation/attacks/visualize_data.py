import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import dataframe_image as dfi
import numpy as np
from scripts import helper
import os
from pathlib import Path
pd.set_option('display.max_columns', None)


def generate_plot(_dataset: str, _attack: str, _agg_error):
    if _attack == 'label_flipping_targeted':
        metric = 'Test/ASR-targeted'
    elif _attack == 'sample_poison':
        metric = 'Test/ASR-backdoor'
    else:
        metric = 'Test/F1Score'

    metric_save = metric.replace("/", "-")
    prefix = f'{_dataset}_{_attack}_{metric_save}'

    ROOT_DIR = Path.cwd()
    DATA_DIR = ROOT_DIR.joinpath(_dataset, _attack)
    print(DATA_DIR)
    DATA_FILE = list(DATA_DIR.glob('*.csv'))[0]
    print('Reading file: ')
    print(DATA_FILE)
    #DATA_DIR = ROOT_DIR.joinpath(dataset, attack_name, data_file)
    SAVE_PATH = DATA_DIR.joinpath("figures")

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
    """
    if len(df_unfinished) > 0:
        print(df_unfinished)
        exit(0)
    """
    mean_benign = pd.pivot_table(benign, index=["aggregator_args.algorithm",
                                                "adversarial_args.attack_env.attack",
                                                "adversarial_args.attack_env.poisoned_node_percent",
                                                "adversarial_args.attack_env.poisoned_sample_percent", ],
                                 values=["Test/Accuracy", "Test/F1Score", "Test/ASR-targeted", "Test/ASR-backdoor"],
                                 aggfunc=['mean', _agg_error], dropna=False)
    mean_benign = mean_benign.reset_index()
    num_attack_configs = len(mean_benign.index)

    """
    if attack_name == 'model_poison':
        assert num_attack_configs == 15
    else:
        assert num_attack_configs == 45
    """

    # Generate Overview Tables
    mean_benign_copy = mean_benign.rename(columns={"adversarial_args.attack_env.poisoned_node_percent": "PNR",
                                                   "adversarial_args.attack_env.poisoned_sample_percent": "PSR",
                                                   "aggregator_args.algorithm": "Aggregator"})
    overview_mean = mean_benign_copy.pivot(columns=["PNR", "PSR"], index=["Aggregator"], values=('mean', metric))
    overview_error = mean_benign_copy.pivot(columns=["PNR", "PSR"], index=["Aggregator"], values=(_agg_error, metric))
    overview_mean_style = overview_mean.style.set_caption(f'Mean {metric}\n')
    if metric in ["Test/Accuracy", "Test/F1Score"]:
        overview_mean_style.highlight_max(props='font-weight: bold')
    else:
        overview_mean_style.highlight_min(props='font-weight: bold')
    overview_error_style = overview_error.style.set_caption(f'{_agg_error.upper()} {metric}\n')
    overview_error_style.highlight_max(props='font-weight: bold')
    mean_file_name = f'{prefix}-mean_overview.png'
    error_file_name = f'{prefix}-error_overview.png'
    dfi.export(overview_mean_style, os.path.join(SAVE_PATH, mean_file_name))
    dfi.export(overview_error_style, os.path.join(SAVE_PATH, error_file_name))

    # Generate Multiline Plots
    sample_percents = mean_benign['adversarial_args.attack_env.poisoned_sample_percent'].unique()
    for percent in sample_percents:
        data_by_percent = mean_benign[mean_benign['adversarial_args.attack_env.poisoned_sample_percent'] == percent]
        plt_title = "Poisoned Sample Percent: " + str(percent)
        percent_plt = helper.get_plot_by_node_percent(data=data_by_percent, y_col=('mean', metric),
                                                      y_err=(_agg_error, metric),
                                                      plt_title=plt_title)
        if _attack != 'model_poison':
            plt.title(f'{_dataset.upper()}, PSR: {percent}', fontsize=10)
        else:
            plt.title(f'{_dataset.upper()}, Noise Type: salt', fontsize=10)
        file_name = f'{prefix}-{percent}.png'
        plt.savefig(os.path.join(SAVE_PATH, file_name))
    plt.close('sentinel')

def main():
    datasets = ["mnist", "fmnist", "cifar10"]
    #attack_names = ["label_flipping_targeted", "label_flipping_untargeted", "sample_poison", "model_poison"]
    attack_names = ["sample_poison"]
    agg_error = 'sem'  # lowercase
    for dataset in datasets:
        for attack_name in attack_names:
            generate_plot(_dataset=dataset, _attack=attack_name, _agg_error='sem')

if __name__ == "__main__":
    main()
