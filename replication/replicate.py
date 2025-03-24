# Run using python replicate --dataset=<name> --model=<name>
# <name>=Citeseer|Cora-ML|PubMed|MS-Academic|A.Computer|A.Photo
# <model>=spinelli|mapping
# See constants.py
# Note: you must be logged in wandb
# Trains 5 models for 20 seeds for the provided dataset and stores result in wandb

import torch
import torch.nn.functional as F
import wandb
import pandas as pd
from tqdm import tqdm

from datetime import datetime
import argparse

from replication.constants import Model, Dataset
from replication.data import get_dataset, set_train_val_test_split
from replication.seeds import gen_seeds, test_seeds

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def get_hyperparameters(dataset_name):
    """hyperparameters from the paper"""
    # default
    params = {
        'hidden': [64],
        'dropout': 0.5,
        'niter': 10,  # maximum number of MP iterations (T)
        'learning_rate': 0.01,
        'weight_decay': 0.008
    }

    # Dataset-specific propagation penalty (α)
    if dataset_name == Dataset.CORAML.value:
        params['prop_penalty'] = 0.005
    elif dataset_name == Dataset.CITESEER.value:
        params['prop_penalty'] = 0.001
    elif dataset_name == Dataset.PUBMED.value:
        params['prop_penalty'] = 0.001
    elif dataset_name == Dataset.MSACADEMIC:
        params['prop_penalty'] = 0.05
    elif dataset_name == Dataset.APHOTO.value or dataset_name == Dataset.ACOMPUTER.value:
        # amazon datasets use weight_decay=0 according to the author's paper and code
        params['weight_decay'] = 0
        params['prop_penalty'] = 0.05
    else:
        raise RuntimeError("Invalid dataset name")

    return params


def train(model, data, optimizer, train_halt=True, weight_decay=0.008):
    """
    model: The AP-GCN model
    data: data object
    optimizer
    train_halt: if to train the halting unit in this epoch
    weight_decay: L2 regularization strength
    Returns: Training loss, average propagation steps
    """
    model.train()

    # the authors do this by optimizing each 5 epochs.
    # TODO: WHY?
    for param in model.prop.parameters():
        param.requires_grad = train_halt

    optimizer.zero_grad()
    logits, steps, reminders = model(data)
    loss = F.nll_loss(logits[data.train_mask], data.y[data.train_mask])  # task loss
    l2_reg = sum((torch.sum(param ** 2) for param in model.reg_params))  # the l2 reg as they did

    # FINAL LOSS EXPRESSION
    loss += weight_decay / 2 * l2_reg + model.prop_penalty * (
            steps[data.train_mask] + reminders[data.train_mask]).mean()

    # backprop
    loss.backward()
    optimizer.step()

    return loss.item(), steps.cpu().numpy()


def evaluate(model, data, mask=None):
    model.eval()
    with torch.no_grad():
        logits, steps, reminders = model(data)

        # just in case we don't have we assign all to evaluate but all these datasets brings their own masks
        if mask is None:
            mask = torch.ones(data.num_nodes, dtype=torch.bool, device=data.x.device)

        pred = logits[mask].max(1)[1]

        # accuracy
        correct = pred.eq(data.y[mask]).sum().item()
        total = mask.sum().item()
        acc = correct / total if total > 0 else 0

    return acc, steps.cpu().numpy()


def train_model(dataset, hyperparams, best_model_path, ModelClass, epochs=10000, patience=100, halting_step=5):
    """Train the AP-GCN model with early stopping"""
    model = ModelClass(
        dataset=dataset,
        niter=hyperparams['niter'],
        prop_penalty=hyperparams['prop_penalty'],
        hidden=hyperparams['hidden'],
        dropout=hyperparams['dropout']
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=hyperparams['learning_rate']
    )

    # variables
    best_val_acc = 0
    counter = 0
    train_losses = []
    train_steps = []
    val_accs = []
    val_steps = []

    # ponder time evolution checkup for my plots.
    ponder_time_evolution = []
    epoch_to_track = [1, 10, 25, 50, 100, 200, 500]

    data = dataset.data.to(device)
    print(data.train_mask.sum().item())

    print("Dataset statistics:")
    print(data)

    print(f"Training AP-GCN with hyperparameters:")
    for key, value in hyperparams.items():
        print(f"  - {key}: {value}")

    for epoch in range(1, epochs + 1):
        start_time = datetime.now()

        # following the authors approach of training the halting unit every 5 epochs
        train_loss, train_step = train(model, data, optimizer, epoch % halting_step == 0, hyperparams['weight_decay'])

        end_time = datetime.now()

        train_losses.append(train_loss)
        train_steps.append(train_step)

        # validation eval
        val_acc, val_step = evaluate(model, data, data.val_mask)
        val_accs.append(val_acc)
        val_steps.append(val_step)

        # time pondering
        if epoch in epoch_to_track:
            with torch.no_grad():
                _, steps, remainders = model(data)
                prop_cost = steps + remainders
                ponder_time_evolution.append((epoch, prop_cost.cpu().numpy()))

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

        train_avg_steps = steps.mean()
        train_min_steps = steps.min()
        train_max_steps = steps.max()

        val_avg_steps = val_step.mean()
        val_min_steps = val_step.min()
        val_max_steps = val_step.max()

        if epoch % 10 == 0:
            print(f"Epoch: {epoch:03d}, Loss: {train_loss:.4f}, Steps: {train_avg_steps:.2f}, Val Acc: {val_acc:.4f}")

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
    test_acc, test_step = evaluate(model, data, data.test_mask)
    print(f"Test Accuracy: {test_acc:.4f}, Avg Steps: {test_step.mean():.2f}")

    # Log entire list of steps per node for the test evaluation
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
        if dataset_name == dataset.value:
            is_good_dataset = True
            break

    if not is_good_dataset:
        raise RuntimeError("Invalid dataset name provided")

    is_good_model = False
    for model in Model:
        if model_name == model.value:
            is_good_model = True
            break

    if not is_good_model:
        raise RuntimeError("Invalid model name provided")


def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--dataset", type=str, help="The dataset name")
    parser.add_argument("--model", default=Model.SPINELLI.value, type=str, help="The model name")

    args = parser.parse_args()

    dataset_name = args.dataset
    model_name = args.model

    validate_input(dataset_name, model_name)

    hyperparameters = get_hyperparameters(dataset_name)
    dataset = get_dataset(dataset_name)
    ModelClass = Model.get_model_class(model_name)

    seeds = test_seeds

    print(f"Using device {device}")

    for seed in tqdm(seeds):
        for i in range(5):
            run_suffix = f"{model_name}_{dataset_name}_{seed}_{i}"
            date_time = datetime.now().strftime('%H-%M-%S_%d-%m-%Y')
            run_name = f"run_{run_suffix}_{date_time}"
            run = wandb.init(project="AP-GCN", name=run_name, tags=[dataset_name, str(seed), model_name],
                             reinit=True)

            torch_seed = gen_seeds()
            torch.manual_seed(seed=torch_seed)

            # Do not move this: different train-val-test split for each seed
            dataset.data = set_train_val_test_split(
                seed,
                dataset.data,
                num_development=1500,
                num_per_class=20
            )

            best_model_path = f'./models/{run_suffix}.pt'

            train_model(dataset, hyperparameters, best_model_path, ModelClass)

            run.log_model(path=best_model_path, name=run_name)

            run.finish()


main()
