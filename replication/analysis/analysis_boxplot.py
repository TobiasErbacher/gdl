from io import StringIO
from typing import Dict, List

import numpy as np
import pandas as pd
import wandb
from matplotlib import pyplot as plt
import matplotlib as mpl
import matplotlib.colors as mcolors

from replication.constants import Model, ANALYSIS_FIGURE_OUTPUT_PATH, MATPLOTLIBPARAMS
from replication.dataset import Dataset
from replication.model_classes.model_spinelli import get_spinelli_configuration

api = wandb.Api()
mpl.rcParams.update(MATPLOTLIBPARAMS)

# Choose the datasets and model for which the boxplots are created
DATASETS = [Dataset.CITESEER, Dataset.CORAML, Dataset.PUBMED, Dataset.MSACADEMIC, Dataset.ACOMPUTER, Dataset.APHOTO]
# Does not work with Model.Cooperative_AP_GCN
MODEL = Model.Gumbel_AP_GCN

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


def plot_boxplot(steps_per_dataset: Dict[str, List[List[int]]], model_name: str):
    standard_deviations_per_dataset = {}

    for dataset_name, steps in steps_per_dataset.items():
        steps_array = np.array(steps)
        standard_deviations_per_node = steps_array.std(axis=0)
        standard_deviations_per_dataset[dataset_name] = standard_deviations_per_node

    # If we create a plot for all datasets we want a consistent order
    if len(standard_deviations_per_dataset) == 6:
        desired_order = [Dataset.CITESEER.label,
                         Dataset.CORAML.label,
                         Dataset.PUBMED.label,
                         Dataset.MSACADEMIC.label,
                         Dataset.ACOMPUTER.label,
                         Dataset.APHOTO.label]

        # Rebuild the dictionary in the desired order
        standard_deviations_per_dataset = {key: standard_deviations_per_dataset[key] for key in desired_order}

    boxplot = plt.boxplot(standard_deviations_per_dataset.values(),
                          tick_labels=standard_deviations_per_dataset.keys(),
                          patch_artist=True,
                          medianprops=dict(color="red")
                          )

    for patch, tick in zip(boxplot["boxes"], plt.gca().get_xticklabels()):
        tick_label = tick.get_text()
        color_value = Dataset.from_label(tick_label).plot_color
        color_rgba = mcolors.to_rgba(color_value, alpha=0.5)
        patch.set_facecolor(color_rgba)

    # We want a consistent y-axis
    y_max = 4.5
    plt.ylim(0, y_max)
    plt.yticks(np.arange(0, y_max, 0.5))

    plt.ylabel("Standard Deviation per Node")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(ANALYSIS_FIGURE_OUTPUT_PATH + f"{model_name}_std_steps_per_node_boxplot.pdf")


dataset_names = [dataset.label for dataset in DATASETS]
model_name = MODEL.label

steps_per_dataset = download_steps(dataset_names, model_name)
plot_boxplot(steps_per_dataset, model_name)
