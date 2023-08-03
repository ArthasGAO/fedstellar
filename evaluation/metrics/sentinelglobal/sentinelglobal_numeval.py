import pandas as pd
import wandb
import os
import matplotlib.pyplot as plt
from scripts import css_helper
import dataframe_image as dfi

# Use the commented code to download again
"""
_project = "fedstellar-fmnist-sentinelglobal-eval"

benign_filter = {
    'config.adversarial_args.attacks': 'No Attack',
    'description': 'participant_0'
}
api = wandb.Api(timeout=60)
runs = api.runs(f'janousy/{_project}', filters=benign_filter)
keys = ["PNR", "PSR", "Attack", "Evaluations"]
data = pd.DataFrame(columns=keys)

for run in runs:
    # .summary contains the output keys/values for metrics like accuracy.
    #  We call ._json_dict to omit large files
    row = dict.fromkeys(keys)
    attack = run.config["attack_env"]["attack"]
    targeted = run.config["attack_env"]["targeted"]
    if attack == 'Label Flipping':
        if targeted: attack = 'Label Flipping Targeted'
        else: attack = 'Label Flipping Untargeted'
    elif attack == 'Sample Poisoning':
        attack = 'Backdoor'
    row["Attack"] = attack
    row["PSR"] = run.config["attack_env"]["poisoned_sample_percent"]
    row["PNR"] = run.config["attack_env"]["poisoned_node_percent"]
    row["Evaluations"] = run.summary["num_evals"]

    df_dict = pd.DataFrame.from_dict([row])
    data = pd.concat([data, df_dict])

data.to_csv("csv/sentinelglobal_numeval.csv")
"""

data = pd.read_csv("csv/sentinelglobal_numeval.csv")

# df_nomp, df_mp = [x for _, x in data.groupby(data['Attack'] == 'Model Poisoning')]
pv = data.pivot_table(index=["Attack","PSR", "PNR"], columns=[], values='Evaluations', aggfunc='mean')
# pv_mp = df_mp.pivot_table(index=["Attack","PNR"], columns=[], values='Evaluations', aggfunc='mean')

pv_unstkd = pv.unstack().unstack()
pv_unstkd = pv_unstkd.fillna('-')
pv_unstkd.index.name = None
pv_unstkd_lv = pv_unstkd.droplevel(0, axis=1)
pv_unstkd_lv_style= pv_unstkd_lv.style.format(precision=0)
styles = css_helper.get_table_styles()
pv_unstkd_lv_style = pv_unstkd_lv_style.set_table_styles(styles)
file = "figures/sentinelglobal_numeval.png"
dfi.export(pv_unstkd_lv_style, file, dpi=600, fontsize=12)
print("Saved figure to: " + file)

