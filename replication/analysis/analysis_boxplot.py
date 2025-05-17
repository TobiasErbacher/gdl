from io import StringIO

import pandas as pd
import wandb
from matplotlib import pyplot as plt
import matplotlib as mpl
import matplotlib.colors as mcolors
from matplotlib.patches import Patch

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


def plot_boxplots(state_distributions_per_dataset):
    # Consistent order
    if len(state_distributions_per_dataset) == 6:
        desired_order = [
            Dataset.CITESEER.label,
            Dataset.CORAML.label,
            Dataset.PUBMED.label,
            Dataset.MSACADEMIC.label,
            Dataset.ACOMPUTER.label,
            Dataset.APHOTO.label
        ]
        state_distributions_per_dataset = {
            key: state_distributions_per_dataset[key] for key in desired_order
        }

    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(12, 10), sharey=True)
    axes = axes.flatten()

    # Define colors for the 4 states
    state_colors = [
        mcolors.CSS4_COLORS["royalblue"],
        mcolors.CSS4_COLORS["seagreen"],
        mcolors.CSS4_COLORS["plum"],
        mcolors.CSS4_COLORS["darkorange"]
    ]

    for dataset_index, (dataset_name, steps_dict) in enumerate(state_distributions_per_dataset.items()):
        axis = axes[dataset_index]
        axis.set_title(dataset_name)
        axis.tick_params(labelleft=True)  # Show y tick labels despite shared y-axis

        all_box_data = []  # List of all state values grouped by step
        xtick_positions = []  # X positions for labeling steps
        xtick_labels = []  # Step numbers
        state_color_indices = []  # Used to track state index per box

        for step_idx, runs_for_step in sorted(steps_dict.items()):
            transposed = list(zip(*runs_for_step))  # Each is a list of values for one state
            all_box_data.extend(transposed)  # Add 4 boxplots (one per state)
            state_color_indices.extend([0, 1, 2, 3])  # Track which state each box corresponds to

            # Center x position of the group of 4 boxes
            center_position = len(all_box_data) - 1.5
            xtick_positions.append(center_position)
            xtick_labels.append(str(step_idx))

        # Alternating gray background
        step_width = 4  # since 4 boxes per step
        for i in range(0, len(xtick_positions)):
            start = i * step_width + 0.5
            end = start + step_width
            axis.axvspan(start, end, facecolor='lightgrey' if i % 2 == 0 else 'white', alpha=0.3)

        # Create the boxplot
        bp = axis.boxplot(all_box_data, patch_artist=True, boxprops={"linewidth":1}, medianprops=dict(color="red"))

        # Color each box according to its state index
        for patch, state_idx in zip(bp["boxes"], state_color_indices):
            patch.set_facecolor(state_colors[state_idx])
            patch.set_alpha(0.7)

        index_to_state = {
            0: "Standard",
            1: "Broadcast",
            2: "Listen",
            3: "Isolate"
        }

        legend_elements = [
            Patch(facecolor=state_colors[i], edgecolor="black", label=f"{index_to_state[i]}", alpha=0.7)
            for i in range(4)
        ]
        fig.legend(handles=legend_elements, loc="upper center", ncol=2)

        axis.set_xticks(xtick_positions)
        axis.set_xticklabels(xtick_labels)
        axis.set_xlabel("Step")
        axis.set_ylabel("Proportion of Nodes")

    plt.tight_layout()
    plt.savefig(ANALYSIS_FIGURE_OUTPUT_PATH + f"state_distribution_per_step.pdf")


dataset_names = [dataset.label for dataset in DATASETS]
model_name = MODEL.label

steps_per_dataset = download_steps(dataset_names, model_name)
plot_boxplot(steps_per_dataset, model_name)
