import pandas as pd
import wandb
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

from io import StringIO
from typing import Dict, List

from replication.constants import Model, MATPLOTLIBPARAMS, ANALYSIS_FIGURE_OUTPUT_PATH
from replication.dataset import Dataset
from replication.model_classes.model_spinelli import get_spinelli_configuration

api = wandb.Api()
mpl.rcParams.update(MATPLOTLIBPARAMS)

# Choose the datasets and model for which the distribution of the number of steps is plotted
DATASETS = [Dataset.CITESEER, Dataset.CORAML, Dataset.PUBMED, Dataset.MSACADEMIC, Dataset.ACOMPUTER, Dataset.APHOTO]
# Does not work with Model.Cooperative_AP_GCN
MODEL = Model.RL_AP_GCN

# Not necessary to change this
PROJECT_NAME = f"{MODEL.label}"


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

        _, _, _, _, model_args = get_spinelli_configuration(None, dataset_name)
        best_prop_penalty = str(model_args.prop_penalty)

        # Only consider runs with the given model
        if model_name in run.tags:
            # Analyze only runs with the best propagation penalty
            if model_name == Model.SPINELLI.label and best_prop_penalty not in run.tags:
                continue
            if "test_steps" in run.summary:
                df_string = run.summary.get("test_steps")
                df = pd.read_json(StringIO(df_string), orient="split")
                step_list = df["step"].tolist()
                if dataset_name not in steps:
                    steps[dataset_name] = [step_list]
                else:
                    steps[dataset_name].append(step_list)
            else:
                raise RuntimeError(f"Run {run} is missing test_steps summary statistic")

    return steps


def plot_step_distribution_per_dataset(steps_per_dataset: Dict[str, List[List[int]]], model_name: str):
    # Over nodes
    step_distribution_per_dataset = {}

    for dataset_name, steps_list in steps_per_dataset.items():
        steps_list = np.array(steps_list)
        step_distribution_per_dataset[dataset_name] = np.average(steps_list, axis=0)

    # Have the legend in a consistent order
    if len(step_distribution_per_dataset) == 6:
        desired_order = [Dataset.CITESEER.label,
                         Dataset.CORAML.label,
                         Dataset.PUBMED.label,
                         Dataset.MSACADEMIC.label,
                         Dataset.ACOMPUTER.label,
                         Dataset.APHOTO.label]

        step_distribution_per_dataset = {key: step_distribution_per_dataset[key] for key in desired_order}

    fig, ax = plt.subplots()

    for dataset_name, avg_steps in step_distribution_per_dataset.items():
        sns.kdeplot(avg_steps,
                    fill=True,
                    linewidth=2,
                    ax=ax,
                    label=dataset_name,
                    color=Dataset.from_label(dataset_name).plot_color
        )

    # We want the same x-axis for all plots
    ax.set_xlim(0, 10)
    ax.set_xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    # We want the same y-axis for all plots
    y_max = 4.0
    ax.set_ylim(0, y_max)
    ax.set_yticks(np.arange(0, y_max, 0.5))

    plt.xlabel("Number of Steps")
    plt.ylabel("Density")
    ax.legend()
    ax.grid(axis="y")
    plt.tight_layout()
    plt.savefig(ANALYSIS_FIGURE_OUTPUT_PATH + f"{model_name}_steps_distribution.pdf")


dataset_names = [dataset.label for dataset in DATASETS]
model_name = MODEL.label

steps_per_dataset = download_steps(dataset_names, model_name)
plot_step_distribution_per_dataset(steps_per_dataset, model_name)
