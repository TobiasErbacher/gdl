import wandb
import numpy as np

from replication.constants import Dataset, Model
from replication.replicate import get_hyperparameters

api = wandb.Api()

# Choose the datasets and model for which the average accuracy is calculated
DATASETS = [Dataset.CITESEER, Dataset.CORAML, Dataset.PUBMED, Dataset.ACOMPUTER]
MODEL = Model.SPINELLI

# Not necessary to change this
PROJECT_NAME = "AP-GCN/AP-GCN"
NUMBER_SAMPLES = 1_000
CONFIDENCE = 0.95


class AvgAccuracyResult():
    def __init__(self, dataset_name, avg_accuracy, uncertainty):
        self.dataset_name = dataset_name
        self.avg_accuracy = avg_accuracy
        self.uncertainty = uncertainty


def bootstrap_confidence_interval(data, confidence, number_samples):
    if confidence < 0 or confidence > 1:
        raise RuntimeError("Invalid confidence value")

    averages = []
    for i in range(number_samples):
        sample = np.random.choice(data, size=len(data), replace=True)
        averages.append(np.average(sample))

    delta = (1 - confidence) / 2
    lower_percentile = np.percentile(averages, delta * 100)  # Convert to percentage
    mean = np.average(averages)
    upper_percentile = np.percentile(averages, (confidence + delta) * 100)  # Convert to percentage

    uncertainty = np.max(np.abs(np.array([lower_percentile - mean, upper_percentile - mean])))
    return mean, uncertainty


def get_name_in_tags(tags, dataset_names):
    for tag in tags:
        if tag in dataset_names:
            return tag
    return None


def calculate_average_accuracies(dataset_names, model_name) -> [AvgAccuracyResult]:
    test_accuracies = {}
    for name in dataset_names:
        test_accuracies[name] = []

    for run in api.runs(PROJECT_NAME):
        dataset_name = get_name_in_tags(dataset_names, run.tags)

        if dataset_name is None:  # Only consider runs with a dataset from the list
            continue

        best_prop_penalty = str(get_hyperparameters(dataset_name)["prop_penalty"])

        # Only consider runs with the given dataset, model, and the best propagation penalty
        if model_name in run.tags and best_prop_penalty in run.tags:
            if "test_accuracy" in run.summary:
                test_accuracies[dataset_name].append(run.summary["test_accuracy"])
            else:
                raise RuntimeError(f"Run {run} is missing test_accuracy summary statistic")

    result_list = []
    for (name, accuracies_list) in test_accuracies.items():
        mean, confidence = bootstrap_confidence_interval(accuracies_list, CONFIDENCE, NUMBER_SAMPLES)
        result_list.append(AvgAccuracyResult(name, mean, confidence))

    return result_list


dataset_names = [name.value for name in DATASETS]
model_name = MODEL.value
results = calculate_average_accuracies(dataset_names, model_name)

for result in results:
    print(f"Dataset: {result.dataset_name}, mean: {result.avg_accuracy*100:.2f} [%], uncertainty: {result.uncertainty*100:.2f} [%]")
