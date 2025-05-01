import os
from io import StringIO
from typing import List, Dict

import numpy as np
import pandas as pd
import wandb
from matplotlib import pyplot as plt
import matplotlib as mpl
import matplotlib.colors as mcolors

from replication.constants import Model, ANALYSIS_FIGURE_OUTPUT_PATH, MATPLOTLIBPARAMS
from replication.data import get_dataset, set_train_val_test_split
from replication.dataset import Dataset
from replication.model_classes.model_spinelli import get_spinelli_configuration

api = wandb.Api()
mpl.rcParams.update(MATPLOTLIBPARAMS)

# Choose the datasets and model for which the boxplots per class are created
DATASETS = [Dataset.CITESEER, Dataset.CORAML, Dataset.PUBMED, Dataset.MSACADEMIC, Dataset.ACOMPUTER, Dataset.APHOTO]
MODEL = Model.SPINELLI

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


def get_test_true_classes(dataset_name):
    dataset = get_dataset(dataset_name)
    dataset.data = set_train_val_test_split(
        0,  # The seed does only affect train and val sets, not the test set
        dataset.data,
        num_development=Dataset.from_label(dataset_name).num_development,
        num_per_class=20
    )
    return dataset.data.y[dataset.data.test_mask].numpy()


def plot_boxplot_per_class(steps_per_dataset: Dict[str, List[List[int]]], model_name: str):
    for dataset_name, steps in steps_per_dataset.items():
        true_classes = get_test_true_classes(dataset_name)

        steps_array = np.array(steps)
        mean_steps_per_node = steps_array.mean(axis=0)

        means_per_class = {}
        for class_index in true_classes:
            means_per_class[class_index] = []

        for index, mean in enumerate(mean_steps_per_node):
            means_per_class[true_classes[index]].append(mean)

        means_per_class_sorted = dict(sorted(means_per_class.items()))

        plt.clf()
        boxplot = plt.boxplot(means_per_class_sorted.values(),
                              tick_labels=means_per_class_sorted.keys(),
                              patch_artist=True,
                              medianprops=dict(color="red")
                              )

        for patch in boxplot["boxes"]:
            patch.set_facecolor("lightblue")

        plt.ylabel("Halting Step")
        plt.xlabel("Class")
        plt.grid(True)
        plt.tight_layout()
        folder_path = os.path.join(ANALYSIS_FIGURE_OUTPUT_PATH, f"halting_step_per_class_{model_name}")
        os.makedirs(folder_path, exist_ok=True)
        save_path = os.path.join(folder_path, f"{model_name}_{dataset_name}_halting_steps_per_class.pdf")
        plt.savefig(save_path)


dataset_names = [dataset.label for dataset in DATASETS]
model_name = MODEL.label

steps_per_dataset = download_steps(dataset_names, model_name)
plot_boxplot_per_class(steps_per_dataset, model_name)
