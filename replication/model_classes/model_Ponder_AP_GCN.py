import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import ModuleList, Dropout, ReLU, Linear
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.utils import dropout_edge

from replication.dataset import Dataset
from replication.model_classes.interfaces import Integrator, EvalArgs, TrainArgs, ModelArgs


class Ponder_AP_GCN(nn.Module):
    """
    Args:
        niter: maximum number of propagation steps
        dropout: percentage of dropout
        beta: factor before the regularization loss (KL divergence to geometric prior)
        lampda_p: defines the geometric prior distribution p_G(lambda_p) on the halting policy
        halt_bias_init: Set the bias for the layer predicting whether to halt or not. Setting it to 0 makes the halting unit more likely to predict no halt in the first iterations. Set it to None to not set it explicitly.
        hidden: size of the hidden layer
    """
    def __init__(self, dataset, niter, dropout, beta, lambda_p, halt_bias_init, hidden):
        """
        AP-GCN with PonderNet-style halting
        """
        super(Ponder_AP_GCN, self).__init__()

        self.niter = niter

        # Local prediction network (MLP)
        num_features = [dataset.data.x.shape[1]] + hidden + [dataset.num_classes]
        # we pass the number of features of node + dimensions of the mlp + output dim
        self.layers = ModuleList()

        for i in range(len(num_features) - 1):
            self.layers.append(Linear(num_features[i], num_features[i + 1]))

        self.halt = Linear(dataset.num_classes, 1)  # halting unit

        if halt_bias_init is not None:
            self.halt.bias.data.fill_(halt_bias_init)

        self.reg_params = list(
            self.layers[0].parameters())  # L2 regularization parameters on first layer like authors of apgcn

        self.dropout = Dropout(p=dropout)
        self.act_fn = ReLU()

        # KL regularization settings
        self.beta = beta
        self.lambda_p = lambda_p

    def forward(self, data, edge_dropout):
        x, edge_index = data.x, data.edge_index

        # local prediction network
        h = x
        for i, layer in enumerate(self.layers):
            h = layer(self.dropout(h))
            if i < len(self.layers) - 1:
                h = self.act_fn(h)

        # storage of those outputs
        outputs = [h]

        # Propagation loop
        prop_h = h
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
            prop_h = self.propagate(edge_index_norm, x=prop_h, norm=norm)
            outputs.append(prop_h)

            # halting logits
            logit = self.halt(prop_h)
            logit = torch.clamp(logit, min=-10, max=10)
            halting_logits.append(logit)

        halting_logits = torch.cat(halting_logits, dim=1)
        lambda_vals = torch.sigmoid(halting_logits)

        # Halting distribution (p_n from PonderNet paper)
        p_list = []
        cumulative_remain_prob = torch.ones(lambda_vals.size(0), 1, device=lambda_vals.device)
        for n in range(self.niter):
            p_n = lambda_vals[:, n:n + 1] * cumulative_remain_prob
            p_list.append(p_n)
            cumulative_remain_prob = cumulative_remain_prob * (1 - lambda_vals[:, n:n + 1])

        # remaining probability
        p_list[-1] = p_list[-1] + cumulative_remain_prob

        # halting probabilities
        p = torch.cat(p_list, dim=1)

        return outputs, p, halting_logits

    def propagate(self, edge_index, x, norm):
        """
        GCN propagation
        """
        x_j = x[edge_index[0]]
        x_j = norm.view(-1, 1) * x_j

        out = torch.zeros_like(x)
        out.scatter_add_(0, edge_index[1].unsqueeze(1).expand(-1, x.size(1)), x_j)

        return out

    def inference(self, data, temperature, deterministic=False):
        self.eval()
        with torch.no_grad():
            outputs, p, _ = self.forward(data, edge_dropout=0)

            if deterministic:
                halt_step = torch.argmax(p, dim=1)
            else:
                # temperature scaling for controlled stochasticity
                p_temp = p ** (1 / max(temperature, 1e-10))
                p_temp = p_temp / p_temp.sum(dim=1, keepdim=True)

                # sampling from the distribution.
                halt_step = torch.multinomial(p_temp, 1).squeeze(1)

            # Final prediction
            final_pred = torch.zeros(p.size(0), outputs[0].size(1), device=p.device)
            for n in range(self.niter):
                mask = (halt_step == n)
                if mask.any():
                    final_pred[mask] = outputs[n + 1][mask]

            # +1 because always at least one step of MP
            return final_pred, halt_step + 1, p

    def compute_loss(self, outputs, p, target, mask=None):
        if mask is not None:
            masked_outputs = [out[mask] for out in outputs]
            masked_p = p[mask]
            masked_target = target[mask]

            num_nodes, niter = masked_p.size()
            device = masked_p.device

            rec_loss = 0.0
            for n in range(niter):
                loss_n = F.cross_entropy(masked_outputs[n + 1], masked_target, reduction='none')
                rec_loss += (masked_p[:, n] * loss_n).mean()  # expectation

            # geometric prior
            steps = torch.arange(1, niter + 1, device=device).float()
            p_g = self.lambda_p * (1 - self.lambda_p) ** (steps - 1)
            p_g[-1] = 1 - p_g[:-1].sum()  # reminder
            p_g = p_g.unsqueeze(0).expand(num_nodes, -1)

            # KL divergence
            kl_loss = torch.sum(masked_p * (torch.log(masked_p + 1e-10) - torch.log(p_g + 1e-10)), dim=1).mean()

            total_loss = rec_loss + self.beta * kl_loss

            return total_loss, rec_loss, kl_loss
        else:
            num_nodes, niter = p.size()
            device = p.device

            rec_loss = 0.0
            for n in range(niter):
                loss_n = F.cross_entropy(outputs[n + 1], target, reduction='none')
                rec_loss += (p[:, n] * loss_n).mean()

            steps = torch.arange(1, niter + 1, device=device).float()
            p_g = self.lambda_p * (1 - self.lambda_p) ** (steps - 1)
            p_g[-1] = 1 - p_g[:-1].sum()  # Ensure sums to 1
            p_g = p_g.unsqueeze(0).expand(num_nodes, -1)

            kl_loss = torch.sum(p * (torch.log(p + 1e-10) - torch.log(p_g + 1e-10)), dim=1).mean()

            total_loss = rec_loss + self.beta * kl_loss
            return total_loss, rec_loss, kl_loss


class Ponder_AP_GCN_TrainArgs(TrainArgs):
    def __init__(self, learning_rate, weight_decay: float, edge_dropout: float):
        super().__init__(learning_rate)
        self.weight_decay = weight_decay
        self.edge_dropout = edge_dropout


class Ponder_AP_GCN_EvalArgs(EvalArgs):
    def __init__(self, temperature: float, deterministic: bool):
        self.temperature = temperature
        self.deterministic = deterministic


class Ponder_AP_GCN_ModelArgs(ModelArgs):
    def __init__(self, dataset, niter, dropout, beta, lambda_p, halt_bias_init, hidden):
        self.dataset = dataset
        self.niter = niter
        self.dropout = dropout
        self.beta = beta
        self.lambda_p = lambda_p
        self.halt_bias_init = halt_bias_init
        self.hidden = hidden


class Ponder_AP_GCN_Integrator(Integrator):
    def train_epoch(self, model, data, optimizer, epoch: int, total_epochs: int,
                    train_args: Ponder_AP_GCN_TrainArgs) -> (
            float, np.array):
        model.train()
        optimizer.zero_grad()

        train_mask = data.train_mask

        outputs, p, halting_logits = model(data, train_args.edge_dropout)
        loss, rec_loss, kl_loss = model.compute_loss(outputs, p, data.y, train_mask)
        l2_reg = sum(torch.sum(param ** 2) for param in model.reg_params)
        loss = loss + train_args.weight_decay / 2 * l2_reg

        loss.backward()
        optimizer.step()

        halt_steps = np.argmax(p[train_mask.cpu().numpy()].cpu().detach().numpy(), axis=1) + 1

        return loss.item(), halt_steps

    def evaluate(self, model, data, mask, eval_args: Ponder_AP_GCN_EvalArgs = None) -> (float, np.array):
        model.eval()

        with torch.no_grad():
            final_pred, halt_step, p = model.inference(
                data,
                temperature=eval_args.temperature,
                deterministic=eval_args.deterministic
            )

            # accuracy
            pred = final_pred[mask].max(1)[1]
            correct = pred.eq(data.y[mask]).sum().item()
            accuracy = correct / mask.sum().item()

        return accuracy, halt_step.cpu().numpy()[mask.cpu().numpy()]


def get_Ponder_AP_GCN_configuration(dataset, dataset_name):
    weight_decay = 0.008

    if dataset_name == Dataset.APHOTO.label or dataset_name == Dataset.ACOMPUTER.label:
        weight_decay = 0

    integrator = Ponder_AP_GCN_Integrator()
    train_args = Ponder_AP_GCN_TrainArgs(0.01, weight_decay, 0.3)
    eval_args = Ponder_AP_GCN_EvalArgs(5.0, False)
    model_args = Ponder_AP_GCN_ModelArgs(dataset, 10, 0.5, 0.01, 0.2, 0.0, [64])

    return Ponder_AP_GCN, integrator, train_args, eval_args, model_args
