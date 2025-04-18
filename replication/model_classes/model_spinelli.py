# Mostly from Jonathan/APGCN_WITH_PLOTS with some changes
import math
import numpy as np

import torch
from torch.nn import Dropout, Linear, ReLU, ModuleList
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.utils import dropout_edge
import torch.nn.functional as F

from replication.dataset import Dataset
from replication.model_classes.interfaces import Integrator, EvalArgs, TrainArgs, ModelArgs


class AdaptivePropagation(MessagePassing):
    def __init__(self, niter: int, h_size: int, bias=True, **kwargs):
        """
        Adaptive propagation layer.

        niter: max number of propagation steps (T in the paper)
        h_size: size of the node embeddings
        bias: if to add a bias in the halting unit
        """
        super(AdaptivePropagation, self).__init__(aggr='add', **kwargs)

        self.niter = niter
        self.halt = Linear(h_size, 1)  # halting unit (Q and q in equation 6)

        self.reg_params = list(self.halt.parameters())  # halting params
        self.dropout = Dropout()

        # normalization params for the GCN layer norm they do in their code, needed to adapt for the new version.
        self.improved = False
        self.add_self_loops = True

        # init params
        self.reset_parameters()

    def reset_parameters(self):
        """
        bias around 1/n+1 -> check my paper comments. it is easy to show that after passing
        through the sigmoid, we get that the probability takes a value around 1/n+1
        """
        self.halt.reset_parameters()

        x = (self.niter + 1) // 1
        b = math.log((1 / x) / (1 - (1 / x)))
        self.halt.bias.data.fill_(b)

    def forward(self, local_preds: torch.FloatTensor, edge_index):
        """
        local_preds: node embeddings from local prediction network
        edge_index: graph connectivity in COO format

        returns:
            Updated node embeddings, number of steps, and remainders
        """
        sz = local_preds.size(0)  # num of nodes.

        steps = torch.ones(sz).to(local_preds.device)  # steps for each node (K_i)
        sum_h = torch.zeros(sz).to(local_preds.device)  # accum halting probs
        continue_mask = torch.ones(sz, dtype=torch.bool).to(local_preds.device)  # active nodes
        x = torch.zeros_like(local_preds).to(local_preds.device)  # embeddings

        # dropout of embedding.
        prop = self.dropout(local_preds)

        # propagation loop
        for i in range(self.niter):
            old_prop = prop  # h^(t-1)

            continue_fmask = continue_mask.float().to(local_preds.device)
            drop_edge_index, _ = dropout_edge(edge_index, p=0.5, training=self.training)  # default is 0.5 as they did.

            # GCN normalization using the util that is now available.
            # -> https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/conv/gcn_conv.html#GCNConv
            edge_index_norm, norm = gcn_norm(
                drop_edge_index,  # Adjacency matrix with some edges dropped
                None,  # We consider unweighted graphs
                sz,
                self.improved,
                self.add_self_loops,  # Whether we want to add self-loops
                self.flow,  # Message passing direction
                local_preds.dtype
            )

            prop = self.propagate(edge_index_norm, x=prop, norm=norm)
            h = torch.sigmoid(self.halt(prop)).t().squeeze()  # h^k_i = non-linearity(Qz^k_i + q)

            # here we do the soft update based on equation 7
            # K_i = min{k : sum(j=1 to k) h^j_i >= 1 - eps}
            # 0.99 is equivalent to (1 - eps) where eps = 0.01
            prob_mask = (((sum_h + h) < 0.99) & continue_mask).squeeze()
            prob_fmask = prob_mask.float().to(local_preds.device)

            # If we get to the last iteration we must not increase the number of steps
            # This is different from the authors' code
            if i == self.niter - 1:
                last_iteration_mask = torch.zeros(sz).to(local_preds.device)
            else:
                last_iteration_mask = torch.ones(sz).to(local_preds.device)

            # we add another step for those nodes that continue and that the accum prob is lower than threshold.
            steps = steps + prob_fmask * last_iteration_mask
            sum_h = sum_h + prob_fmask * h  # and update the accumulation for those nodes that continue  (otherwise the prob mask takes 0 so no update. )

            final_iter = steps < self.niter

            # prob_mask = 1 iff sum_h + h < 0.99, but we want to halt if it is greater
            # final_iter = 1 iff steps < self.niter, but we want to halt if it is greater
            # we want to return (1 - sum_h) iff prob_mask = 0 or final_iter = 0
            # this is equivalent to returning (1 - sum_h) iff (prob_mask AND final_iter) = 0
            condition = prob_mask & final_iter
            p = torch.where(condition, sum_h, 1 - sum_h)  # p^k_i according to equation 8

            # this is something they did in the code too
            # Randomly set continuation mask to 0 for some nodes (i.e., force them to halt)
            # Could be useful to not rely too heavily on specific nodes
            to_update = self.dropout(continue_fmask).unsqueeze(1)

            # equation 9 -> soft-update
            # z'_i = (1/K_i) * sum(k=1 to K_i) p^k_i * z^k_i + (1-p^k_i) * z^(k-1)_i

            x = x + (p.unsqueeze(1) * prop + (1 - p).unsqueeze(1) * old_prop) * to_update
            continue_mask = continue_mask & prob_mask

            # if all nodes halted, then stop.
            if (~continue_mask).all():
                break

        # continuation of the equation 9 (1/K_i)
        x = x / steps.unsqueeze(1)

        # updated embeddings, steps, and  R_i
        return x, steps, (1 - sum_h)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j


class APGCN(torch.nn.Module):
    """
    Adaptive Propagation Graph Convolutional Network.
    """

    def __init__(self,
                 dataset,
                 niter=10,
                 prop_penalty=0.005,
                 hidden=[64],
                 dropout=0.5):
        """
        dataset: The graph dataset
        niter: Maximum number of propagation steps
        prop_penalty: Propagation penalty α in equation 11
        hidden: List of hidden layer sizes
        dropout: Dropout rate
        """
        super(APGCN, self).__init__()

        num_features = [dataset.data.x.shape[1]] + hidden + [dataset.num_classes]  # layer sizes.

        # as authors did, we create the mlp before prop.
        layers = []
        for in_features, out_features in zip(num_features[:-1], num_features[1:]):
            layers.append(Linear(in_features, out_features))

        # we do the propagation with the previous format.
        self.prop = AdaptivePropagation(niter, dataset.num_classes)

        self.prop_penalty = prop_penalty  # alpha

        self.layers = ModuleList(layers)  # mlp

        # we separate parameters into regularized and non-regularized groups -> they did this in their code.
        self.reg_params = list(layers[0].parameters())
        self.non_reg_params = list([p for l in layers[1:] for p in l.parameters()])
        self.dropout = Dropout(p=dropout)
        self.act_fn = ReLU()

        self.reset_parameters()

    def reset_parameters(self):
        self.prop.reset_parameters()
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, data, return_propagation_cost=False):
        """
        data: PyG data object containing x and edge_index
        return_propagation_cost: Whether to return the propagation cost

        returns:
            Log probabilities, number of steps, and remainders
        """
        x, edge_index = data.x, data.edge_index

        # MLP
        for i, layer in enumerate(self.layers):
            x = layer(self.dropout(x))

            # no non-linearity in the last layer.
            if i == len(self.layers) - 1:
                break

            x = self.act_fn(x)

        # the adaptive propagation.
        x, steps, reminders = self.prop(x, edge_index)

        # log probabilities, steps, and remainders
        if return_propagation_cost:
            return torch.nn.functional.log_softmax(x, dim=1), steps, reminders
        return torch.nn.functional.log_softmax(x, dim=1), steps, reminders


class SpinelliTrainArgs(TrainArgs):
    def __init__(self, learning_rate, halting_step, weight_decay):
        super().__init__(learning_rate)
        self.halting_step = halting_step
        self.weight_decay = weight_decay


class SpinelliModelArgs(ModelArgs):
    def __init__(self, dataset, niter, prop_penalty, hidden, dropout):
        self.dataset = dataset
        self.niter = niter
        self.prop_penalty = prop_penalty
        self.hidden = hidden
        self.dropout = dropout


class SpinelliIntegrator(Integrator):
    def train_epoch(self, model, data, optimizer, epoch: int, total_epochs: int, train_args: SpinelliTrainArgs) -> (float, np.array):
        model.train()

        # the authors do this by optimizing each 5 epochs.
        # TODO: WHY?
        for param in model.prop.parameters():
            param.requires_grad = epoch % train_args.halting_step == 0

        optimizer.zero_grad()
        logits, steps, reminders = model(data)
        loss = F.nll_loss(logits[data.train_mask], data.y[data.train_mask])  # task loss
        l2_reg = sum((torch.sum(param ** 2) for param in model.reg_params))  # the l2 reg as they did

        # FINAL LOSS EXPRESSION
        loss += train_args.weight_decay / 2 * l2_reg + model.prop_penalty * (
                steps[data.train_mask] + reminders[data.train_mask]).mean()

        # backprop
        loss.backward()
        optimizer.step()

        return loss.item(), steps.cpu().numpy()[data.train_mask.cpu().numpy()]

    def evaluate(self, model, data, mask, eval_args: EvalArgs = None) -> (float, np.array):
        model.eval()
        with torch.no_grad():
            logits, steps, reminders = model(data)

            pred = logits[mask].max(1)[1]

            # accuracy
            correct = pred.eq(data.y[mask]).sum().item()
            total = mask.sum().item()
            acc = correct / total if total > 0 else 0

        return acc, steps.cpu().numpy()[mask.cpu().numpy()]


def get_spinelli_configuration(dataset, dataset_name):
    prop_penalty = None
    weight_decay = 0.008

    if dataset_name == Dataset.CORAML.label:
        prop_penalty = 0.005
    elif dataset_name == Dataset.CITESEER.label:
        prop_penalty = 0.001
    elif dataset_name == Dataset.PUBMED.label:
        prop_penalty = 0.001
    elif dataset_name == Dataset.MSACADEMIC.label:
        prop_penalty = 0.05
    elif dataset_name == Dataset.APHOTO.label or dataset_name == Dataset.ACOMPUTER.label:
        # amazon datasets use weight_decay=0 according to the author's paper and code
        weight_decay = 0
        prop_penalty = 0.05
    else:
        raise RuntimeError("Invalid dataset name")

    integrator = SpinelliIntegrator()
    train_args = SpinelliTrainArgs( 0.01, 5, weight_decay)
    model_args = SpinelliModelArgs(dataset, 10, prop_penalty, [64], 0.5)

    return APGCN, integrator, train_args, None, model_args
