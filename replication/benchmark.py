# Run using python replicate --dataset=<name> --model=<name> --prop-penalties <prop-penalties>
# <name>=Citeseer|Cora-ML|PubMed|MS-Academic|A.Computer|A.Photo
# <model>=SPINELLI|RL-AP-GCN|Ponder-AP-GCN|Gumbel-AP-GCN|CO-AP-GCN
# <prop-penalties>=list of propagation penalties to run. E.g.: 0.1 0.05 0.01 0.005 0.001 0.0005 0.0001
# See constants.py
# Note: you must be logged in wandb
# Trains 5 models for 20 seeds for the provided dataset and stores result in wandb for each propagation penalty
import numpy as np
import torch
import wandb
import pandas as pd
from tqdm import tqdm

from datetime import datetime
import argparse

from replication.constants import Model
from replication.dataset import Dataset
from replication.data_loading.data import get_dataset, set_train_val_test_split
from replication.model_classes.interfaces import Integrator, TrainArgs, EvalArgs, ModelArgs
from replication.data_loading.seeds import gen_seeds, test_seeds

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def train_model(dataset, epochs, best_model_path, patience, ModelClass, integrator: Integrator, train_args: TrainArgs,
                eval_args: EvalArgs, model_args: ModelArgs):
    model = ModelClass(**vars(model_args)).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=train_args.learning_rate
    )

    # variables
    best_val_acc = 0
    counter = 0

    data = dataset.data.to(device)

    print("Dataset statistics:")
    print(data)

    for epoch in range(1, epochs + 1):
        start_time = datetime.now()

        train_loss, train_step = integrator.train_epoch(model, data, optimizer, epoch, epochs, train_args)

        end_time = datetime.now()

        # validation
        val_acc, val_step = integrator.evaluate(model, data, data.val_mask, eval_args)

        # early stopping :
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            counter = 0
            # Save the best model
            torch.save(model.state_dict(), best_model_path)
        else:
            counter += 1

        if counter >= patience:
            print(f"Early stopping after {epoch} epochs.")
            break

        elapsed_time_ms = (end_time - start_time).total_seconds() * 1000

        train_avg_steps = train_step.mean()
        train_min_steps = train_step.min()
        train_max_steps = train_step.max()

        val_avg_steps = val_step.mean()
        val_min_steps = val_step.min()
        val_max_steps = val_step.max()

        if epoch % 10 == 0:
            print(
                f"Epoch: {epoch:03d}, Loss: {train_loss:.4f}, Train Avg Steps: {train_avg_steps:.2f}, Val Acc: {val_acc:.4f}")

        wandb.log({
            "train_loss": train_loss,
            "val_acc": val_acc,
            "train_avg_steps": train_avg_steps,
            "train_min_steps": train_min_steps,
            "train_max_steps": train_max_steps,
            "val_avg_steps": val_avg_steps,
            "val_min_steps": val_min_steps,
            "val_max_steps": val_max_steps,
            "training_time_epoch": elapsed_time_ms
        }, epoch)

    # Testing
    model.load_state_dict(torch.load(best_model_path))
    test_acc, test_step = integrator.evaluate(model, data, data.test_mask, eval_args)
    print(f"Test Accuracy: {test_acc:.4f}, Avg Test Steps: {test_step.mean():.2f}")

    # Log list of steps per node for the test evaluation
    df = pd.DataFrame(test_step, columns=["step"])
    df_json = df.to_json(orient="split")

    wandb.log({
        "test_accuracy": test_acc,
        "test_steps": df_json,
    })

    return model


def validate_input(dataset_name, model_name):
    is_good_dataset = False
    for dataset in Dataset:
        if dataset_name == dataset.label:
            is_good_dataset = True
            break

    if not is_good_dataset:
        raise RuntimeError("Invalid dataset name provided")

    is_good_model = False
    for model in Model:
        if model_name == model.label:
            is_good_model = True
            break

    if not is_good_model:
        raise RuntimeError("Invalid model name provided")


def main():
    np.random.seed(11)  # Required for reproducibility of torch_seed = gen_seeds()

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--dataset", type=str, help="The dataset name")
    parser.add_argument("--model", default=Model.SPINELLI.label, type=str, help="The model name")
    parser.add_argument("--prop-penalties", type=float, nargs="+",
                        help=f"The list of propagation penalties (alphas) to run (only if model={Model.SPINELLI.label})")
    parser.add_argument("--store-models", default=False, type=bool, help="Whether models should be stored in wandb")

    args = parser.parse_args()

    dataset_name = args.dataset
    model_name = args.model
    prop_penalties = args.prop_penalties
    store_models = args.store_models

    validate_input(dataset_name, model_name)

    dataset = get_dataset(dataset_name)

    _, _, _, _, model_args = Model.from_label(model_name).get_config(dataset, dataset_name)

    # If the model is Spinelli and no propagation penalty is provided just run with the best propagation penalty
    if prop_penalties is None and model_name == Model.SPINELLI.label:
        prop_penalties = [model_args.prop_penalty]

    # For any non-Spinelli model we do not care about the propagation penalty
    if model_name != Model.SPINELLI.label:
        prop_penalties = ["x"]  # Placeholder

    seeds = test_seeds

    print(f"Using device {device}")

    if model_name == Model.SPINELLI.label:
        print(f"Running for propagation penalties: {prop_penalties}")

    for prop_penalty in prop_penalties:
        if model_name == Model.SPINELLI.label:
            model_args.prop_penalty = prop_penalty

        for index, seed in tqdm(enumerate(seeds)):
            for i in range(5):
                if model_name == Model.SPINELLI.label:
                    run_suffix = f"{model_name}_{dataset_name}_{prop_penalty}_{seed}_{i}"
                else:
                    run_suffix = f"{model_name}_{dataset_name}_{seed}_{i}"

                date_time = datetime.now().strftime("%H-%M-%S_%d-%m-%Y")
                run_name = f"run_{run_suffix}_{date_time}"

                tags = [dataset_name, str(seed), model_name]
                if model_name == Model.SPINELLI.label:
                    tags.append(str(prop_penalty))

                run = wandb.init(project=model_name, name=run_name, tags=tags, reinit=True)

                torch_seed = gen_seeds()
                torch.manual_seed(seed=torch_seed)

                # Do not move this: different train-val-test split for each seed
                dataset.data = set_train_val_test_split(
                    seed,
                    dataset.data,
                    num_development=Dataset.from_label(dataset_name).num_development,
                    num_per_class=20
                )

                best_model_path = f"./saved_models/{run_suffix}.pt"

                # Since train_args or eval_args can be stateful, need to fetch fresh instance
                # for every training run
                ModelClass, integrator, train_args, eval_args, model_args = Model.from_label(model_name).get_config(
                    dataset, dataset_name)

                train_model(dataset, 10_000, best_model_path, 100, ModelClass, integrator, train_args, eval_args,
                            model_args)

                if store_models:
                    run.log_model(path=best_model_path, name=run_name)

                if model_name == Model.SPINELLI.label:
                    print(
                        f"Run for prop-penalty {prop_penalty}, seed index {index}/{len(seeds) - 1}, trial {i}/4 finished")
                else:
                    print(f"Run for seed index {index}/{len(seeds) - 1} and trial {i}/4 finished")

                run.finish()


if __name__ == "__main__":
    main()
