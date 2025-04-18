import wandb
import numpy as np

from replication.constants import Model
from replication.dataset import Dataset
from replication.model_classes.model_spinelli import get_spinelli_configuration

api = wandb.Api()

# Choose the datasets and model for which the average accuracy is calculated
DATASETS = [Dataset.CITESEER, Dataset.CORAML, Dataset.PUBMED, Dataset.MSACADEMIC, Dataset.ACOMPUTER, Dataset.APHOTO]
MODEL = Model.SPINELLI

# Not necessary to change this
PROJECT_NAME = f"{MODEL.label}"


class TrainingTimeResult():
    def __init__(self, dataset_name, avg_training_time_epoch, total_training_time):
        self.dataset_name = dataset_name
        self.avg_training_time_epoch = avg_training_time_epoch
        self.total_training_time = total_training_time


def get_name_in_tags(tags, dataset_names):
    for tag in tags:
        if tag in dataset_names:
            return tag
    return None


def calculate_training_times(dataset_names, model_name) -> [TrainingTimeResult]:
    run_durations = {}
    for name in dataset_names:
        run_durations[name] = []

    avg_training_times_per_epoch = {}
    for name in dataset_names:
        avg_training_times_per_epoch[name] = []

    for run in api.runs(PROJECT_NAME):
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

            run_duration = run.summary['_runtime']
            run_durations[dataset_name].append(run_duration)

            avg_training_times_per_epoch[dataset_name].append(np.average(run.history(keys=["training_time_epoch"])["training_time_epoch"]))

    result_list = []
    for name in dataset_names:
        result_list.append(TrainingTimeResult(name, np.average(avg_training_times_per_epoch[name]), np.sum(run_durations[name])))

    return result_list


dataset_names = [name.label for name in DATASETS]
model_name = MODEL.label
results = calculate_training_times(dataset_names, model_name)

for result in results:
    print(f"Dataset: {result.dataset_name}, avg training time per epoch: {result.avg_training_time_epoch:.1f} [ms], sum of run-times: {result.total_training_time:.0f} [s]")
