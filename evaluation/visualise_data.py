import pandas as pd
from scripts import helper
import matplotlib.pyplot as plt
import dataframe_image as dfi
import numpy as np
import os
from pathlib import Path

pd.set_option('display.max_columns', None)

data_file = "wandb_export_2023-06-14_lf_untargeted.csv"
dataset = "mnist"
attack_name = "label_flipping_untargeted"

prefix = f'{dataset}_{attack_name}'

ROOT_DIR = Path.cwd()
print(ROOT_DIR)
DATA_DIR = ROOT_DIR.joinpath(dataset, attack_name, data_file)
save_path = ROOT_DIR.joinpath(dataset, attack_name, "figures")

print("Writing figures to: ")
print(save_path)

raw_data = pd.read_csv(DATA_DIR)

raw_data['adversarial_args.attack_env.poisoned_sample_percent'] = raw_data['Group'].str.extract(r'-S(.*)_R')
raw_data['adversarial_args.attack_env.poisoned_sample_percent'].unique()
fixed_data = raw_data.astype({'adversarial_args.attack_env.poisoned_sample_percent': 'int64'})
# Sanity check
print(fixed_data["adversarial_args.attack_env.poisoned_sample_percent"].unique())
print(fixed_data["adversarial_args.attack_env.poisoned_node_percent"].unique())

benign = fixed_data[fixed_data['adversarial_args.attacks'] == 'No Attack']

mean_benign = pd.pivot_table(benign, index=["aggregator_args.algorithm",
                                            "adversarial_args.attack_env.attack",
                                            "adversarial_args.attack_env.poisoned_node_percent",
                                            "adversarial_args.attack_env.poisoned_sample_percent",],
                             values=["Test/Accuracy","Test/F1Score"], aggfunc = 'mean', dropna=False)
mean_benign = mean_benign.reset_index()
num_attack_configs = len(mean_benign.index)
assert num_attack_configs == 45

overview_f1 = mean_benign.pivot(columns=["adversarial_args.attack_env.poisoned_node_percent", "adversarial_args.attack_env.poisoned_sample_percent"],
                                index=["aggregator_args.algorithm"],
                                values="Test/F1Score")

overview_f1 = overview_f1.rename(columns={'adversarial_args.attack_env.poisoned_node_percent': 'poisoned_node_percent'})
overview_f1_style = overview_f1.style.set_caption("Label Flipping Untargeted: F1 Score")
overview_f1_style.highlight_max()
file_name = f'{prefix}-_overview_f1.png'
dfi.export(overview_f1_style, os.path.join(save_path, file_name))

sample_percents = mean_benign['adversarial_args.attack_env.poisoned_sample_percent'].unique()

for percent in sample_percents:
    data_by_percent = mean_benign[mean_benign['adversarial_args.attack_env.poisoned_sample_percent'] == percent]
    plt_title = "Poisoned Sample Percent: " + str(percent)

    # percent_f1 = helper.get_plot_by_node_percent(data=data_by_percent, y_col='Test/F1Score', plt_title=str(percent))
    percent_acc = helper.get_plot_by_node_percent(data=data_by_percent, y_col='Test/F1Score', plt_title=plt_title)
    file_name = f'{prefix}-{percent}.png'
    plt.savefig(os.path.join(save_path, file_name))