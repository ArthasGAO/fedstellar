import random

import wandb
import pandas as pd
import math
import matplotlib.pyplot as plt
import os
from matplotlib.lines import Line2D

N_ROUNDS = 10
FONT_SIZE = 12


def get_metrics_plot(_entity, _project, _filter, _metric_name, _metric_label):
    api = wandb.Api()
    attack_name = f'{_filter["config.attack_env.attack"]}_{_filter["config.attack_env.targeted"]}'
    SAVE_PATH = os.path.join(os.getcwd(), "figures", _project, attack_name)
    print("Writing figure to: " + SAVE_PATH)
    file_name = _metric_name.replace(" ", "") + "_" + _project + \
                "_" + ("_").join(str(v).replace(" ", "_") for v in _filter.values())
    print(file_name)

    isExist = os.path.exists(SAVE_PATH)
    if not isExist:
        os.makedirs(SAVE_PATH)

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

    N_ROUNDS = 10
    ACTIVE_ROUND = p0_run.config["aggregator_args"]["sentinelglobal_active_round"]
    N_METRICS = N_ROUNDS - ACTIVE_ROUND
    print(N_METRICS)

    df_metric = pd.DataFrame(columns=neis, index=range(ACTIVE_ROUND, N_ROUNDS))

    metric = [("Global Trust ", "Global Trust"), ][0]
    _metric_name = metric[0]  # cos, avg_loss, mapped_loss, agg_weight
    _metric_label = metric[1]

    for ip in neis:
        nei_metric = []
        keys = [f'{_metric_name}{ip}']
        print(keys)
        for metric_dict in p0_run.scan_history(keys=keys):
            metric = metric_dict.get(f'{_metric_name}{ip}', float(-1))
            # if metric is None: metric = float(-1)
            if metric == 'Infinity': metric = float(-1)
            if not math.isnan(metric): nei_metric.append(metric)
        print(nei_metric)
        df_metric[ip] = nei_metric

    benign_neis = benign.copy()
    benign_neis.remove(p0_ip)
    df_metric_benign = pd.DataFrame(columns=benign_neis, index=range(0, N_ROUNDS))
    df_metric_malicious = pd.DataFrame(columns=malicious, index=range(0, N_ROUNDS))

    for b in benign_neis:
        df_metric_benign[b] = df_metric[b]
    for m in malicious:
        df_metric_malicious[m] = df_metric[m]

    ax = None

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
            markersize=random.randint(0, 5),
            linestyle=style.get_linestyle(),
            linewidth=1,
            color=style.get_color(),
            # capsize=4,
            # label='malicious' if malicious else 'benign',
            ax=ax,
            legend=None)

    legend_elements = [benign_style, malicious_style]
    # ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(ACTIVE_ROUND - 0.1, 9.1)
    # Set legend
    ax.legend(handles=legend_elements, fontsize=12)
    # Set grid
    ax.grid(True, linestyle='--', alpha=0.5)
    # Increase tick font sizes
    plt.yticks(fontsize=FONT_SIZE)
    plt.xticks(fontsize=FONT_SIZE)
    ax.xaxis.set_tick_params(labelsize=FONT_SIZE - 2)
    ax.yaxis.set_tick_params(labelsize=FONT_SIZE - 2)
    ax.set_xlabel("Round", fontsize=FONT_SIZE)
    ax.set_ylabel(_metric_label, fontsize=FONT_SIZE)
    plt.savefig(os.path.join(SAVE_PATH, file_name), dpi=600, bbox_inches='tight')

    ax = None
    plt.close('all')


def run(project, attack, metric, pnr, psr):
    t_project = project

    run_filter_attack = {
        'config.attack_env.poisoned_node_percent': pnr,
        'config.attack_env.poisoned_sample_percent': psr,
        'config.aggregator_args.algorithm': 'SentinelGlobal',
        'config.attack_env.attack': attack[0],
        'config.attack_env.targeted': attack[1]
    }

    run_filter_attack_mp = {
        'config.attack_env.poisoned_node_percent': pnr,
        'config.aggregator_args.algorithm': 'SentinelGlobal',
        'config.attack_env.attack': attack[0],
        'config.attack_env.targeted': attack[1]
    }

    run_filter = run_filter_attack_mp if attack[0] == 'Model Poisoning' else run_filter_attack
    print(run_filter)

    t_metric_name = metric[0]  # cos, avg_loss, mapped_loss, agg_weight
    t_metric_label = metric[1]  # Cosine Similarity, Bootstrap Loss, Mapped Loss, Aggregation Weight,
    t_entity = "janousy"

    get_metrics_plot(t_entity, t_project, run_filter, t_metric_name, t_metric_label)


if __name__ == '__main__':

    projects = [
        "fedstellar-fmnist-sentinelglobal-eval",
    ]

    metrics = [
        ("Global Trust", "Global Trust"),
    ]
    attacks = [
        ("Label Flipping", True),
        ("Label Flipping", False),
        ("Model Poisoning", False),
        ("Sample Poisoning", True),
    ]

    pnrs = [10, 50, 80]
    psrs = [30, 50, 100]

    # pnrs = [10]
    # psrs = [30]

    for project in projects:
        for attack in attacks:
            for pnr in pnrs:
                for psr in psrs:
                    for metric in metrics:
                        run(project, attack, metric, pnr, psr)
