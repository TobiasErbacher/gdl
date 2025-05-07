import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Dropout, Linear
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.utils import dropout_edge

from replication.dataset import Dataset
from replication.model_classes.interfaces import TrainArgs, Integrator, EvalArgs, ModelArgs


# From Jonathan RL_GUMBEL_GUMBELSAMPLING.ipynb


class Gumbel_AP_GCN(nn.Module):
    def __init__(self, dataset, niter, dropout, beta, lambda_p, hidden):
        super(Gumbel_AP_GCN, self).__init__()

        self.niter = niter

        in_features = dataset.data.x.shape[1]
        hidden_dim = hidden[0]
        out_features = dataset.num_classes

        # Local prediction network
        self.fc1 = Linear(in_features, hidden_dim)
        self.fc2 = Linear(hidden_dim, out_features)

        self.halt = Linear(out_features, 1)

        self.dropout = Dropout(p=dropout)

        self.beta = beta
        self.lambda_p = lambda_p

        self.halt.bias.data.fill_(-1.0)
        self.training_steps = []

    def forward(self, data, edge_dropout):
        x, edge_index = data.x, data.edge_index

        # Initial embedding
        h = F.relu(self.fc1(self.dropout(x)))
        h = self.fc2(self.dropout(h))

        predictions = [h]  # predictions at each step

        # prop steps
        current_h = h
        halting_logits = []
        for _ in range(self.niter):
            if self.training and edge_dropout > 0:
                drop_edge_index, _ = dropout_edge(edge_index, p=edge_dropout)
            else:
                drop_edge_index = edge_index

            edge_index_norm, norm = gcn_norm(
                drop_edge_index, None,
                x.size(0), False, True, "source_to_target", x.dtype
            )

            # mp
            row, col = edge_index_norm
            x_j = current_h[row] * norm.view(-1, 1)
            new_h = torch.zeros_like(current_h)
            new_h.scatter_add_(0, col.unsqueeze(1).expand(-1, current_h.size(1)), x_j)

            current_h = self.dropout(new_h)
            predictions.append(current_h)

            # halting logit
            logit = self.halt(current_h)
            halting_logits.append(logit)

        halting_logits = torch.cat(halting_logits, dim=1)
        lambda_vals = torch.sigmoid(halting_logits)  # halting probabilities

        # halting distribution (PonderNet-style)
        p_list = []
        remaining_prob = torch.ones(lambda_vals.size(0), 1, device=lambda_vals.device)
        for n in range(self.niter):
            p_n = lambda_vals[:, n:n + 1] * remaining_prob
            p_list.append(p_n)
            remaining_prob = remaining_prob * (1 - lambda_vals[:, n:n + 1])

        p_list[-1] = p_list[-1] + remaining_prob
        p = torch.cat(p_list, dim=1)

        return predictions, p

    # TODO (note to myself): Large diff to Ponder variant how loss is calculated
    def compute_loss(self, predictions, p, target, tau):
        """Compute loss using Gumbel-softmax sample during training"""
        # sample a halting step using Gumbel-softmax
        log_p = torch.log(p + 1e-10)
        sampled = F.gumbel_softmax(log_p, tau=tau, hard=True)
        halt_steps = torch.argmax(sampled, dim=1)

        if self.training:
            self.training_steps.append(halt_steps.float().mean().item())

        # predictions at sampled steps
        selected_preds = torch.zeros_like(predictions[0])
        for n in range(self.niter):
            mask = (halt_steps == n)
            if mask.any():
                selected_preds[mask] = predictions[n + 1][mask]

        # Cross-entropy loss on sampled predictions
        rec_loss = F.cross_entropy(selected_preds, target)

        # KL divergence with geometric prior
        n_steps = torch.arange(1, self.niter + 1, device=p.device, dtype=torch.float)
        # Geometric distribution
        prior = self.lambda_p * (1 - self.lambda_p) ** (n_steps - 1)
        prior = prior / prior.sum()
        prior = prior.unsqueeze(0).expand(p.size(0), -1)

        # KL divergence
        kl_loss = torch.sum(p * (torch.log(p + 1e-10) - torch.log(prior + 1e-10)), dim=1).mean()
        total_loss = rec_loss + self.beta * kl_loss

        return total_loss, rec_loss, kl_loss, halt_steps + 1

    def inference(self, data, tau=0.01):
        """Inference with controlled Gumbel sampling"""
        self.eval()
        with torch.no_grad():
            predictions, p = self.forward(data, 0)

            # small tau for near-deterministic sampling
            log_p = torch.log(p + 1e-10)
            sampled = F.gumbel_softmax(log_p, tau=tau, hard=True)
            halt_steps = torch.argmax(sampled, dim=1)

            final_pred = torch.zeros_like(predictions[0])
            for n in range(self.niter):
                mask = (halt_steps == n)
                if mask.any():
                    final_pred[mask] = predictions[n + 1][mask]

            return final_pred, halt_steps + 1, p

    def get_tau(self, epoch, tau_warmup, tau_decay, tau_initial, tau_final):
        """tau value for current epoch"""
        warmup = tau_warmup
        decay = tau_decay
        tau_initial = tau_initial
        tau_final = tau_final

        if epoch <= warmup:
            return tau_initial

        if epoch >= warmup + decay:
            return tau_final

        # linear decay
        progress = (epoch - warmup) / decay
        tau = tau_initial - progress * (tau_initial - tau_final)

        return tau


class Gumbel_AP_GCN_TrainArgs(TrainArgs):
    def __init__(self, learning_rate, weight_decay, edge_dropout, tau_warmup, tau_decay, tau_initial, tau_final):
        super().__init__(learning_rate)
        self.weight_decay = weight_decay
        self.edge_dropout = edge_dropout
        self.tau_warmup = tau_warmup
        self.tau_decay = tau_decay
        self.tau_initial = tau_initial
        self.tau_final = tau_final


class Gumbel_AP_GCN_EvalArgs(EvalArgs):
    def __init__(self, tau):
        self.tau = tau


class Gumbel_AP_GCN_ModelArgs(ModelArgs):
    def __init__(self, dataset, niter, dropout, beta, lambda_p, hidden):
        self.dataset = dataset
        self.niter = niter
        self.dropout = dropout
        self.beta = beta
        self.lambda_p = lambda_p
        self.hidden = hidden


class Gumbel_AP_GCN_Integrator(Integrator):
    def train_epoch(self, model, data, optimizer, epoch: int, total_epochs: int,
                    train_args: Gumbel_AP_GCN_TrainArgs) -> (
            float, np.array):
        model.train()

        # current temperature
        tau = model.get_tau(epoch, train_args.tau_warmup, train_args.tau_decay, train_args.tau_initial, train_args.tau_final)

        optimizer.zero_grad()
        predictions, p = model(data, train_args.edge_dropout)

        train_mask = data.train_mask
        masked_preds = [pred[train_mask] for pred in predictions]
        masked_p = p[train_mask]
        masked_y = data.y[train_mask]

        # loss with Gumbel sampling
        loss, rec_loss, kl_loss, halt_steps = model.compute_loss(
            masked_preds, masked_p, masked_y, tau
        )

        # L2 regularization
        l2_reg = sum(torch.sum(param ** 2) for param in model.fc1.parameters())
        loss = loss + (train_args.weight_decay / 2) * l2_reg

        loss.backward()
        optimizer.step()

        return loss.item(), halt_steps.cpu().numpy()

    def evaluate(self, model, data, mask, eval_args: Gumbel_AP_GCN_EvalArgs = None) -> (float, np.array):
        model.eval()

        with torch.no_grad():
            final_pred, halt_steps, _ = model.inference(data, tau=eval_args.tau)

            # Calculate accuracy
            pred = final_pred[mask].max(1)[1]
            correct = pred.eq(data.y[mask]).sum().item()
            total = mask.sum().item()
            accuracy = correct / total if total > 0 else 0

        return accuracy, halt_steps.cpu().numpy()[mask.cpu().numpy()]


def get_Gumbel_AP_GCN_configuration(dataset, dataset_name):
    weight_decay = 0.008

    if dataset_name == Dataset.APHOTO.label or dataset_name == Dataset.ACOMPUTER.label:
        weight_decay = 0

    integrator = Gumbel_AP_GCN_Integrator()
    train_args = Gumbel_AP_GCN_TrainArgs(0.01, weight_decay, 0.3,20, 50, 10.0, 1.0)
    eval_args = Gumbel_AP_GCN_EvalArgs(0.01)
    # Even though the beta is 0 and the lambda_p does not do anything do not set it to 0 because this results in
    # division by 0 and the loss will be nan
    model_args = Gumbel_AP_GCN_ModelArgs(dataset, 10, 0.5, 0.0, 0.2, [64])

    return Gumbel_AP_GCN, integrator, train_args, eval_args, model_args
