import pandas as pd
import wandb
import os
import matplotlib.pyplot as plt

def download_threshold_stats(_project, _attack_list, _thresholds):
    benign_filter = {
        'config.adversarial_args.attacks': 'No Attack',
    }
    api = wandb.Api(timeout=19)
    runs = api.runs(f'janousy/{_project}', filters=benign_filter)
    keys = ["Threshold", "PSR", "Attack", "Targeted", "ASR-backdoor", "ASR-targeted", "F1Score"]
    data = pd.DataFrame(columns=keys)

    for run in runs:
        # .summary contains the output keys/values for metrics like accuracy.
        #  We call ._json_dict to omit large files
        row = dict.fromkeys(keys)
        row["Threshold"] = run.config["aggregator_args"]["sentinel_distance_threshold"]
        row["Attack"] = run.config["attack_env"]["attack"]
        row["Targeted"] = run.config["attack_env"]["targeted"]
        row["PSR"] = run.config["attack_env"]["poisoned_sample_percent"]
        row["ASR-backdoor"] = run.summary["Test/ASR-backdoor"]
        row["ASR-targeted"] = run.summary["Test/ASR-targeted"]
        row["F1Score"] = run.summary["Test/F1Score"]

        df_dict = pd.DataFrame.from_dict([row])
        data = pd.concat([data, df_dict])

    attack_configs = [("Label Flipping", False, "F1Score", "Label Flipping Untargeted"),
                      ("Label Flipping", True, "ASR-targeted", "Label Flipping Targeted"),
                      ("Model Poisoning", False, "F1Score", "Model Poisoning"),
                      ("Sample Poisoning", True, "ASR-backdoor", "Sample Poisoning")]

    df_final = pd.DataFrame(columns=_thresholds, index=_attack_list)

    for config in attack_configs:
        filtered = data[(data["Attack"] == config[0]) & (data["Targeted"] == config[1])]
        filtered = filtered[["Threshold", config[2]]]
        filtered = filtered.groupby(['Threshold']).mean().reset_index()

        scores = filtered[config[2]].tolist()
        df_final.loc[config[3]] = scores
    df_final.index.name = "Attack"
    print(df_final)
    df_final.to_csv(_project + ".csv")


def plot_attack_threshold(attack, projects, thresholds, save_path):
    attack_name = attack[0]
    attack_metric = attack[1]
    datasets = []
    for project in projects:
        datasets.append(project.split("-")[1].upper())
    df_attack = pd.DataFrame(columns=thresholds, index=datasets)

    for i, project in enumerate(projects):
        df_dataset = pd.read_csv(project + ".csv", index_col='Attack')
        scores = df_dataset.loc[attack_name].tolist()
        df_attack.loc[datasets[i]] = scores

    colors = ['c', 'm', 'y']
    marker_styles = ['+', 'x', 'o']
    line_styles = ['-', '--', '-.', ':', '-']
    for i, dataset in enumerate(datasets):
        df_dataset = df_attack.loc[dataset]
        ax = df_dataset.plot(
            y=dataset,
            marker=marker_styles[i],
            markersize=5,
            linestyle=line_styles[i],
            linewidth=1,
            color=colors[i],
            label=dataset,
            legend=True)
    FONT_SIZE = 12
    plt.ylim(-0.05, 1.05)
    # Set legend
    plt.legend(fontsize=FONT_SIZE)
    # Set grid
    plt.grid(True, linestyle='--', alpha=0.5)
    # ax.minorticks_on()
    # ax.grid(which='major', linestyle='-', linewidth='0.5', color='black')
    plt.xticks([0, 0.25, 0.5, 0.75, 1])
    # Increase tick font sizes
    plt.xticks(fontsize=FONT_SIZE)
    plt.yticks(fontsize=FONT_SIZE)
    plt.xlabel("Sentinel Distance Threshold", fontsize=FONT_SIZE)
    plt.ylabel(attack_metric, fontsize=FONT_SIZE)
    # plt.title(f'{dataset.upper()} Baseline, {metric.capitalize()}', fontsize=FONT_SIZE)
    file_name = f'sentinel_threshold_{attack_name.lower().replace(" ", "_")}.pdf'
    plt.savefig(os.path.join("figures", file_name), dpi=300, bbox_inches=0)
    plt.close('all')


if __name__ == '__main__':
    projects = ["fedstellar-mnist-threshold-eval", "fedstellar-fmnist-threshold-eval", "fedstellar-cifar10-threshold-eval"]
    attack_list = [("Label Flipping Untargeted", "F1-Score"),
                   ("Label Flipping Targeted", "ASR-LF"),
                   ("Model Poisoning", "F1-Score"),
                   ("Sample Poisoning", "BA")]
    thresholds = [0.01, 0.1, 0.25, 0.5, 0.75, 1.0]
    SAVE_PATH = os.path.join(os.getcwd(), "figures")
    if not os.path.exists(SAVE_PATH): os.makedirs(SAVE_PATH)

    # Use if not yet downloaded from wandb

    for project in projects:
        download_threshold_stats(project, attack_list, thresholds)

    for attack in attack_list:
        plot_attack_threshold(attack, projects, thresholds, save_path=SAVE_PATH)
    print("Figures saved to: " + SAVE_PATH)
