## Evaluation
This directory contains the processed raw data and the required 
scripts to reproduce the figures and tables of the associated thesis.

All experiments were logged to [Weights & Biases](https://wandb.ai/janousy/projects). Each project
is publicly available and further data can be retrieved using the UI or the API provided by WandB.

The raw data used for the thesis is available in each corresponding directory in `csv` format. The directory `attack` 
includes the main results for each attack on all datasets. The directory `baseline` holds the performance
metrics without the presence of adversaries. The directory `metrics` encompasses the data of additional
discussions. Each directory holds the necessary scripts to visualize or summarize data. Some also offer
the option to refresh the data from WandB.

### Requirements
Recommended python version: 3.10.x

```pip install -r requirements.txt```

To retrieve the data from the WandB API, an account is recommended. The API key can be found in the settings of the account.
Alternatively, any random key of length 46 can be used to retrieve data from the public projects. Further information
can be retrieved from the [Documentation](https://docs.wandb.ai/)