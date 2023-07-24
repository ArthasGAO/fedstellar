import random

import wandb
import pandas as pd
import math
import matplotlib.pyplot as plt
import os
from matplotlib.lines import Line2D

N_ROUNDS = 10
FONT_SIZE = 12
FILE_TYPE = ".pdf"

def get_metrics_plot(_entity, _project, _filter, _attack_name, _metric_name, _metric_label):

    api = wandb.Api(timeout=20)
    SAVE_PATH = os.path.join(os.getcwd(), "figures_" + FILE_TYPE.replace(".", ""), _attack_name, _metric_name,)
    SAVE_PATH_CSV = os.path.join(os.getcwd(), "csv", _project)
    print("Writing figure to: " + SAVE_PATH)
    file_name = _metric_name + "_" + _project + "_" + ("_").join(str(v).replace(" ", "_") for v in _filter.values())
    print(file_name)

    if not os.path.exists(SAVE_PATH): os.makedirs(SAVE_PATH)
    if not os.path.exists(SAVE_PATH_CSV): os.makedirs(SAVE_PATH_CSV)

    print(_filter)

    # find p0 with the lowest port
    runs = api.runs(f'{_entity}/{_project}', filters=_filter)
    print(len(runs))
    addresses = []
    benign = []
    malicious = []
    for run in runs:
        print(run.group)
        ip = run.config["network_args"]
        attack = run.config["adversarial_args"]["attacks"]
        addresses.append(ip)
        if attack == 'No Attack':
            benign.append(ip)
        else:
            malicious.append(ip)

    addresses.sort(key=lambda x: x.split(':')[1])
    p0_ip = addresses[0]

    _filter['config.network_args'] = p0_ip
    p0_run = api.runs(f'{_entity}/{_project}', filters=_filter)[0]
    neis = addresses.copy()
    neis.remove(p0_ip)
    df_metric = pd.DataFrame(columns=neis, index=range(0, N_ROUNDS))

    for ip in neis:
        nei_metric = []
        keys = [f'{_metric_name}_{ip}']
        print(keys)
        for metric_dict in p0_run.scan_history(keys=keys):
            metric = metric_dict.get(f'{_metric_name}_{ip}', float(-1))
            # if metric is None: metric = float(-1)
            if metric == 'Infinity': metric = float(-1)
            if not math.isnan(metric): nei_metric.append(metric)
        if not nei_metric:
            nei_metric = [-1] * N_ROUNDS
        if len(nei_metric) > N_ROUNDS:  # fix for redundant logging
            nei_metric_fix = []
            num_redundant = int((len(nei_metric) / N_ROUNDS))
            for i in range(0, len(nei_metric), num_redundant):
                fix = nei_metric[i]
                nei_metric_fix.append(fix)
            nei_metric = nei_metric_fix
        df_metric[ip] = nei_metric

    df_metric.to_csv(os.path.join(SAVE_PATH_CSV, file_name + '.csv'))

    benign_neis = benign.copy()
    benign_neis.remove(p0_ip)
    df_metric_benign = pd.DataFrame(columns=benign_neis, index=range(0, N_ROUNDS))
    df_metric_malicious = pd.DataFrame(columns=malicious, index=range(0, N_ROUNDS))

    for b in benign_neis:
        df_metric_benign[b] = df_metric[b]
    for m in malicious:
        df_metric_malicious[m] = df_metric[m]

    ax = None
    plt.subplots(figsize=(5, 3))

    benign_style = Line2D([0], [0], marker='o', color='green', lw=1, label='Benign', linestyle='solid')
    malicious_style = Line2D([0], [0], marker='x', color='red', lw=1, label='Malicious', linestyle='dashed')
    for nei in neis:
        df_metric_nei = df_metric[nei]
        is_mal = True if nei in malicious else False
        style = malicious_style if is_mal else benign_style
        ax = df_metric_nei.plot(
            y=nei,
            # yerr=error,
            marker=style.get_marker(),
            markersize=random.randint(0,5),
            linestyle=style.get_linestyle(),
            linewidth=1,
            color=style.get_color(),
            # capsize=4,
            # label='malicious' if malicious else 'benign',
            ax=ax,
            legend=None)

    legend_elements = [benign_style, malicious_style]
    if 'loss' not in _metric_name: ax.set_ylim(-0.05, 1.05)
    # ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(-0.1, 9.1)
    # Set legend
    ax.legend(handles=legend_elements, fontsize=FONT_SIZE)
    # Set grid
    ax.grid(True, linestyle='--', alpha=0.5)
    # Increase tick font sizes
    plt.yticks(fontsize=FONT_SIZE)
    plt.xticks(fontsize=FONT_SIZE)
    ax.xaxis.set_tick_params(labelsize=FONT_SIZE)
    ax.yaxis.set_tick_params(labelsize=FONT_SIZE)
    ax.set_xlabel("Round", fontsize=FONT_SIZE)
    ax.set_ylabel(_metric_label, fontsize=FONT_SIZE)
    # plt.figure().set_figheight(3)
    plt.savefig(os.path.join(SAVE_PATH, file_name + FILE_TYPE), dpi=600, bbox_inches='tight')

    ax = None
    plt.close('all')


def run(project, attack, metric):
    attack_name = attack[0]
    targeted = attack[1]
    psrs = [30, 50, 100]
    t_project = project
    t_metric_name = metric[0]  # cos, avg_loss, mapped_loss, agg_weight
    t_metric_label = metric[1]  # Cosine Similarity, Bootstrap Loss, Mapped Loss, Aggregation Weight,
    t_entity = "janousy"
    t_attack_name = attack_name.replace(" ", "_").lower()
    is_targeted = "_targeted" if targeted else "_untargeted"
    t_attack_name += is_targeted
    for psr in psrs:
        run_filter_attack = {
            'config.attack_env.poisoned_node_percent': 50,
            'config.attack_env.poisoned_sample_percent': psr,
            'config.aggregator_args.algorithm': 'Sentinel',
            'config.attack_env.attack': attack_name,
            'config.attack_env.targeted': targeted
        }
        run_filter_attack_mp = {
            'config.attack_env.poisoned_node_percent': 50,
            'config.aggregator_args.algorithm': 'Sentinel',
            'config.attack_env.attack': attack_name
        }
        run_filter = run_filter_attack_mp if attack_name == 'Model Poisoning' else run_filter_attack
        get_metrics_plot(t_entity, t_project, run_filter, t_attack_name, t_metric_name, t_metric_label)

if __name__ == '__main__':

    projects = [
        "metrics_fmnist",
        "metrics_mnist",
        "metrics_cifar10"
    ]

    metrics = [
        # ("mapped_loss", "Mapped Loss"),
        ("cos", "Cosine Similarity"),
        ("agg_weight", "Agg. Weight"),
        ("avg_loss", "Avg. Loss"),
    ]

    attacks = [
        ("Sample Poisoning", True),
        ("Model Poisoning", False),
        ("Label Flipping", True),
        ("Label Flipping", False),
    ]

    for project in projects:
        for attack in attacks:
            for metric in metrics:
                run(project, attack, metric)
