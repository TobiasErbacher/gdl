import pandas as pd

import wandb
import numpy as np

from replication.constants import Dataset, Model

api = wandb.Api()

# Choose the dataset and model for which the accuracy is calculated
DATASET = Dataset.CITESEER
MODEL = Model.SPINELLI

# Not necessary to change this
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


def calculate_average_accuracy_and_steps(dataset_name, model_name):
    test_accuracies = []
    steps = []

    for run in api.runs(PROJECT_NAME):
        # Only consider runs with the given dataset and model
        if dataset_name in run.tags and model_name in run.tags:
            if "test_accuracy" in run.summary:
                test_accuracies.append(run.summary["test_accuracy"])
            else:
                raise RuntimeError(f"Run {run} is missing test_accuracy summary statistic")
            if "test_steps" in run.summary:
                df_string = run.summary.get('test_steps')
                df = pd.read_json(df_string, orient="split")
                list = df['step'].tolist()
                steps.append(list)
            else:
                raise RuntimeError(f"Run {run} is missing test_steps summary statistic")

    return bootstrap_confidence_interval(test_accuracies, CONFIDENCE, NUMBER_SAMPLES), np.array(steps)


dataset_name = DATASET.value
model_name = MODEL.value
(lower, mean, upper), steps = calculate_average_accuracy_and_steps(dataset_name, model_name)
plus_minus = max(lower, upper)  # Like the authors do it in the AP-GCN_demo.ipynb
print(f"Dataset: {dataset_name}, lower: {lower}, mean: {mean}, upper: {upper}, plus_minus: {plus_minus}")
print(f"Steps per node: {steps}")
