import wandb
import numpy as np

from replication.constants import Dataset

api = wandb.Api()

DATASET = Dataset.CITESEER

PROJECT_NAME = "AP-GCN/AP-GCN"
NUMBER_SAMPLES = 1_000
CONFIDENCE = 0.95


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

    return lower_percentile, mean, upper_percentile


def calculate_average_accuracy(dataset_name):
    tag_to_filter = dataset_name
    test_accuracies = []

    for run in api.runs(PROJECT_NAME):
        if tag_to_filter in run.tags:
            if "test_accuracy" in run.summary:
                test_accuracies.append(run.summary["test_accuracy"])
            else:
                raise RuntimeError(f"Run {run} is missing test_accuracy summary statistic")

    return bootstrap_confidence_interval(test_accuracies, CONFIDENCE, NUMBER_SAMPLES)


dataset_name = DATASET.value
lower, mean, upper = calculate_average_accuracy(dataset_name)
plus_minus = ((mean - lower) + (upper - mean)) / 2
print(f"Dataset: {dataset_name}, lower: {lower}, mean: {mean}, upper: {upper}, plus_minus: {plus_minus}")
