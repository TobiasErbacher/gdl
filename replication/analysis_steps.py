import pandas as pd
import wandb
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

from io import StringIO
from typing import Dict, List

from replication.constants import Dataset, Model, datasetToColorString, MATPLOTLIBPARAMS
from replication.replicate import get_hyperparameters

api = wandb.Api()
mpl.rcParams.update(MATPLOTLIBPARAMS)

# Choose the datasets and model for which the distribution of the number of steps is plotted
DATASETS = [Dataset.CITESEER, Dataset.CORAML, Dataset.PUBMED, Dataset.ACOMPUTER]
MODEL = Model.SPINELLI

# Not necessary to change this
PROJECT_NAME = "AP-GCN/AP-GCN"


def get_name_in_tags(tags, dataset_names):
    for tag in tags:
        if tag in dataset_names:
            return tag
    return None


def download_steps(dataset_names: [Dataset], model_name: str):
    steps = {}

    for run in api.runs(PROJECT_NAME):
        # Only consider runs with the given dataset, model, and the best propagation penalty
        dataset_name = get_name_in_tags(dataset_names, run.tags)

        if dataset_name is None:  # Only consider runs with a dataset from the list
            continue

        best_prop_penalty = str(get_hyperparameters(dataset_name)["prop_penalty"])

        if model_name in run.tags and best_prop_penalty in run.tags:
            if "test_steps" in run.summary:
                df_string = run.summary.get('test_steps')
                df = pd.read_json(StringIO(df_string), orient="split")
                list = df['step'].tolist()
                if dataset_name not in steps:
                    steps[dataset_name] = [list]
                else:
                    steps[dataset_name].append(list)
            else:
                raise RuntimeError(f"Run {run} is missing test_steps summary statistic")

    return steps


def plot_step_distribution_per_dataset(steps_per_dataset: Dict[str, List[List[int]]], model_name: str):
    # Over nodes
    step_distribution_per_dataset = {}

    for dataset_name, steps_list in steps_per_dataset.items():
        steps_list = np.array(steps_list)
        step_distribution_per_dataset[dataset_name] = np.average(steps_list, axis=0)

    fig, ax = plt.subplots()
    for dataset_name, avg_steps in step_distribution_per_dataset.items():
        sns.kdeplot(avg_steps,
                    fill=True,
                    ax=ax,
                    label=dataset_name,
                    color=datasetToColorString(Dataset(dataset_name))
        )

    plt.xlabel("Number of Steps")
    plt.ylabel("Density")
    ax.legend()
    ax.grid(axis="y")
    plt.tight_layout()
    plt.savefig(f"{model_name}_step_distribution.svg")


dataset_names = [dataset.value for dataset in DATASETS]
model_name = MODEL.value

steps_per_dataset = download_steps(dataset_names, model_name)
plot_step_distribution_per_dataset(steps_per_dataset, model_name)
