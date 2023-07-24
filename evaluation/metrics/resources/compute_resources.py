import numpy as np
import wandb
import pandas as pd
import math
import matplotlib.pyplot as plt
import os

from PIL import Image
from matplotlib.lines import Line2D
import numpy as np
import dataframe_image as dfi

from scripts import css_helper

api = wandb.Api(timeout=20)

FONT_SIZE = 12

def filter_system_metrics(system_metrics, gpu=False):
    if gpu:
        gpu = system_metrics[(list(system_metrics.filter(regex='system.gpu.0.gpu$')))]
        gpu = gpu.dropna() # remove system errors
        mean_gpu = gpu.mean().values.tolist()[0]
        sem_gpu = gpu.sem().values.tolist()[0] / 2 # +,-
        mean_ram = system_metrics['system.gpu.0.memoryAllocated'].mean() / 100 * 10000 # % of GPU 0 (10000 MB)
        sem_ram = system_metrics['system.gpu.0.memoryAllocated'].sem() / 100 * 10000
        sem_ram = sem_ram / 2
        bytes_sent_mb = system_metrics['system.network.sent'].mean() / pow(1000,2) # Bytes to MB
        agg_metrics = [mean_gpu, mean_ram, bytes_sent_mb]
        agg_range = [sem_gpu, sem_ram, 0]
    else:
        mean_all_cpus = system_metrics[(list(system_metrics.filter(regex='system.cpu.*.cpu_percent$')))].mean()
        mean_cpu = mean_all_cpus.mean()
        sem_cpu = mean_all_cpus.sem() / 2
        mean_ram = system_metrics['system.proc.memory.rssMB'].mean()
        sem_ram = system_metrics['system.proc.memory.rssMB'].sem() / 2
        bytes_sent_mb = system_metrics['system.network.sent'].mean() / pow(1000,2) # Bytes to MB
        agg_metrics = [mean_cpu, mean_ram, bytes_sent_mb]
        agg_range = [sem_cpu, sem_ram, 0]
    return agg_metrics, agg_range


def plot_compute_metrics(entity, project, psr, pnr, SAVE_PATH, file_name):
    print(project)
    aggregation_list = ["Krum", "FedAvg", "TrimmedMean", "FlTrust", "Sentinel", "SentinelGlobal"]
    aggregation_list = aggregation_list.sort(reverse=True)
    metrics_list_cpu = ['CPU (%)', 'Memory (MB)', 'Traffic (MB)']
    metrics_list_gpu = ['GPU (%)', 'Memory (MB)', 'Traffic (MB)']
    metrics_list = metrics_list_gpu if 'cifar10' in project else metrics_list_cpu
    gpu = True if 'cifar10' in project else False
    baseline = True if 'baseline' in project else False
    print("Computing GPU metrics: " + str(gpu))
    final_metrics = pd.DataFrame(data=None, index=aggregation_list, columns=metrics_list, dtype=None, copy=None)
    final_metrics_range = pd.DataFrame(data=None, index=aggregation_list, columns=metrics_list, dtype=None, copy=None)

    # filter by attack
    runs = api.runs(f'{entity}/{project}', filters={'config.attack_env.poisoned_node_percent': pnr,
                                                    'config.attack_env.poisoned_sample_percent': psr,
                                                    'description': 'participant_0',
                                                    })
    print(len(runs))
    assert len(runs) == 6
    for run in runs:
        # print(run.description)
        # print(run.group)
        system_metrics = run.history(stream="events")
        agg_metrics, agg_error = filter_system_metrics(system_metrics, gpu)
        agg = run.config['aggregator_args']['algorithm']
        final_metrics.loc[agg] = agg_metrics
        final_metrics_range.loc[agg] = agg_error

    final_metrics = final_metrics.sort_index()

    """
    cpu_bound = final_metrics[metrics_list[0]].min()
    cpu_bound = cpu_bound - cpu_bound/10
    ram_bound = final_metrics[metrics_list[1]].min()
    ram_bound = ram_bound - ram_bound/10
    traffic_bound = final_metrics[metrics_list[2]].min()
    traffic_bound = traffic_bound - traffic_bound/10
    """

    # if "fmnist" in project:
    proc_lower = 34
    proc_upper = 42
    ram_lower = 950
    ram_upper = 1050
    traffic_lower = 40
    traffic_upper = 50
    if "cifar10" in project:
        proc_lower = 50
        proc_upper = 80
        ram_lower = 3000
        ram_upper = 5000
        traffic_lower = 20
        traffic_upper = 35

    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(15, 3), sharey="all")
    colors = ['c', 'm', 'y']
    for i, col in enumerate(final_metrics.columns):
        x_err = final_metrics_range[col].tolist()
        final_metrics.plot.barh(y=col,
                                xerr=x_err,
                                rot=0,
                                legend=None,
                                fontsize=FONT_SIZE,
                                ax=axs[i],
                                width=0.2,
                                color=colors[i])
        axs[i].set_xlabel(col, fontsize=FONT_SIZE)
        axs[i].xaxis.grid(True, alpha=0.5)
        axs[i].yaxis.grid(False)
        axs[i].xaxis.set_major_locator(plt.MaxNLocator(3))
        if 'Memory' in col:
            axs[i].set_xlim(ram_lower, ram_upper)
        if 'CPU' in col or 'GPU' in col:
            axs[i].set_xlim(proc_lower, proc_upper)
        if 'Traffic' in col:
            axs[i].set_xlim(traffic_lower, traffic_upper)
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.1, hspace=0)
    fig.savefig(os.path.join(SAVE_PATH, file_name), dpi=600, bbox_inches='tight')

    columns_with_var = metrics_list[0:2]
    for c in columns_with_var:
        final_metrics[c] = final_metrics[c].apply(lambda x: "{:.3f}".format(x) + "\u00B1")
        final_metrics[c] = final_metrics[c] + final_metrics_range[c].apply(lambda x: "{:.3f}".format(x))
    print(final_metrics)
    final_metrics.to_csv(os.path.join(SAVE_PATH, 'csv', f'final_metrics_{project}.csv'))
    styles = css_helper.get_table_styles()
    final_metrics_style = final_metrics.style.set_table_styles(styles)
    final_metrics_style.format(precision=3)
    final_metrics_style.set_properties(subset=[metrics_list[1]], **{'width': '120px'})
    # final_metrics_style = final_metrics_style.highlight_min()
    # final_metrics_style = final_metrics_style.highlight_max(props='font-weight: bold')
    # print(final_metrics.to_html())
    dfi.export(final_metrics_style, os.path.join(PNG_PATH, f'final_metrics_{project}.png'), dpi=600, fontsize=12)
    image_1 = Image.open(os.path.join(PNG_PATH, f'final_metrics_{project}.png'))
    image_1.convert('RGB').save(os.path.join(SAVE_PATH, f'final_metrics_{project}.pdf'))


if __name__ == '__main__':

    projects = ["fedstellar-mnist-lf-untargeted",
                "fedstellar-mnist-lf-targeted",
                "fedstellar-mnist-mp", # (N80 not set)
                "fedstellar-mnist-sp-targeted",
                "fedstellar-fashionmnist-lf-untargeted",
                "fedstellar-fashionmnist-lf-targeted",
                "fedstellar-fashionmnist-mp",
                "fedstellar-fashionmnist-sp-targeted",
                "fedstellar-cifar10-lf-untargeted",
                "fedstellar-cifar10-lf-targeted",
                "fedstellar-cifar10-mp",
                "fedstellar-cifar10-sp-targeted",
                "fedstellar-mnist-baseline",
                "fedstellar-fasionmnist-baseline",
                "fedstellar-cifar10-baseline"]

    """
    projects = ["fedstellar-fashionmnist-lf-untargeted",
                "fedstellar-fashionmnist-lf-targeted",
                "fedstellar-fashionmnist-mp",
                "fedstellar-fashionmnist-sp-targeted",
                "fedstellar-cifar10-lf-untargeted",
                "fedstellar-cifar10-lf-targeted",
                "fedstellar-cifar10-mp",
                "fedstellar-cifar10-sp-targeted",
                "fedstellar-mnist-baseline",
                "fedstellar-fashionmnist-baseline",
                "fedstellar-cifar10-baseline"]
    """

    # projects = ["fedstellar-cifar10-mp", "fedstellar-fashionmnist-mp"]

    for project in projects:
        entity = "janousy"
        project = project
        SAVE_PATH = os.path.join(os.getcwd(), "figures")
        CSV_SAVE_PATH = os.path.join(SAVE_PATH, "csv")
        PNG_PATH = os.path.join(SAVE_PATH, "png")
        print("Writing figures to: " + SAVE_PATH)
        file_name = project + ".pdf"
        if not os.path.exists(SAVE_PATH): os.makedirs(SAVE_PATH)
        if not os.path.exists(CSV_SAVE_PATH): os.makedirs(CSV_SAVE_PATH)
        if not os.path.exists(PNG_PATH): os.makedirs(PNG_PATH)
        psr = 100 if "mp" not in project else 0
        pnr = 80
        if 'baseline' in project:
            psr = 0
            pnr = 0
        plot_compute_metrics(entity=entity, project=project, psr=psr, pnr=pnr, SAVE_PATH=SAVE_PATH, file_name=file_name)
#%