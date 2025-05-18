from collections import deque

import math
import numpy as np
import torch
from torch.nn import ModuleList, Dropout, ReLU, Linear
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.utils import dropout_edge
import torch.nn.functional as F

from replication.dataset import Dataset
from replication.model_classes.interfaces import TrainArgs, ModelArgs, Integrator, EvalArgs


# From Jonathan RL_GUMBEL_GUMBELSAMPLING.ipynb

class RewardNormalizer:
    def __init__(self, size=100):
        self.returns = deque(maxlen=size)

    def normalize(self, reward):
        self.returns.append(reward)
        if len(self.returns) > 1:
            mean = np.mean(self.returns)
            std = np.std(self.returns) + 1e-8
            return (reward - mean) / std
        return reward


class RLAdaptiveProp(MessagePassing):
    def __init__(self,
                 niter: int,
                 h_size: int,
                 computation_penalty=0,
                 exploration_factor=0.1,
                 use_scheduled_penalty=True,
                 **kwargs):
        """
        Args:
          niter: maximum number of propagation steps
          h_size: size of the node embeddings for policy & value networks
          computation_penalty: base penalty for each propagation step
          exploration_factor: controls exploration noise
          use_scheduled_penalty: if we want to phase in the penalty over epochs -> we actually dont use it but was a first appraoch to add it.
        """
        super(RLAdaptiveProp, self).__init__(aggr='add', **kwargs)
        self.niter = niter
        self.base_computation_penalty = computation_penalty
        self.computation_penalty = computation_penalty
        self.exploration_factor = exploration_factor
        self.use_scheduled_penalty = use_scheduled_penalty

        self.current_epoch = 0

        # POLICY NETWORK HERE: MANAGE THE HALTING DECISION.
        # The max is necessary because you can have datasets with 3 classes (3//4=0)
        self.policy_hidden1 = Linear(h_size, h_size // 2)
        self.policy_hidden2 = Linear(h_size // 2, max(1, h_size // 4))
        self.policy = Linear(max(1, h_size // 4), 1)

        # VALUE NETWORK FOR THE BASELINE.
        self.value_hidden1 = Linear(h_size, h_size // 2)
        self.value_hidden2 = Linear(h_size // 2, max(1, h_size // 4))
        self.value = Linear(max(1, h_size // 4), 1)

        self.reg_params = list(self.policy_hidden1.parameters()) + \
                          list(self.policy_hidden2.parameters()) + \
                          list(self.policy.parameters())
        self.dropout = Dropout()

        self.improved = False
        self.add_self_loops = True

        self.reset_parameters()

    def set_epoch(self, current, total):
        self.current_epoch = current
        self.total_epochs = total
        if self.use_scheduled_penalty:
            self.update_computation_penalty()

    def update_computation_penalty(self):
        """
          my first idea was to think of the task as 3 phases.  a first part we allow to go to more layers by lower penalty
          and slowly goes up such that enforce lower layers later on , once the layers are more or less trained. anyway,
          since the computation penalty is set to 0, we could just remove it.
        """
        progress = self.current_epoch / max(self.total_epochs, 1)
        if progress < 0.3:
            self.computation_penalty = self.base_computation_penalty * 0.1
        elif progress < 0.7:
            phase_progress = (progress - 0.3) / 0.4
            self.computation_penalty = self.base_computation_penalty * (0.1 + 0.9 * phase_progress)
        else:
            self.computation_penalty = self.base_computation_penalty

    def reset_parameters(self):
        self.policy_hidden1.reset_parameters()
        self.policy_hidden2.reset_parameters()
        self.policy.reset_parameters()
        self.value_hidden1.reset_parameters()
        self.value_hidden2.reset_parameters()
        self.value.reset_parameters()
        # policy bias to -2.5 . lower sets prob of halting lower  which encourage deeper propagation -> can estimate by sigmoid introducing
        # WE COULD THINK THIS AS AN HYPERPARAMETER THOUGH! ALTHOUGH THE APGCN AUTHORS DO SOMETHING SIMILAR WITH 1/N
        if hasattr(self.policy, 'bias') and self.policy.bias is not None:
            self.policy.bias.data.fill_(-2.5)

    def forward(self, local_preds: torch.FloatTensor, edge_index):
        if self.total_epochs is None:
            raise RuntimeError("Must set call set_epoch() before calling forward()")

        sz = local_preds.size(0)
        x = local_preds.clone()
        steps = torch.ones(sz, device=local_preds.device)
        active = torch.ones(sz, dtype=torch.bool, device=local_preds.device)  # all need to start active -> 1

        # record per node:  the log probability, value and entropy at halting
        halting_log_prob = torch.zeros(sz, device=local_preds.device)
        halting_value = torch.zeros(sz, device=local_preds.device)
        halting_entropy = torch.zeros(sz, device=local_preds.device)

        prop = self.dropout(local_preds)

        # exploration noise (decays over epochs)
        current_exploration = self.exploration_factor
        if self.total_epochs > 0:
            decay_rate = 3.0  # WE NEED TO PASS ALSO AS HYPERPARAMETER ALTHOUGH NOT A BIG THING
            progress = self.current_epoch / self.total_epochs
            current_exploration = self.exploration_factor * math.exp(-decay_rate * progress)

        # here is where mp occurs
        for t in range(self.niter):
            if not active.any():
                break

            # dropout and normalization as authors did for more clear comparison.
            drop_edge_index, _ = dropout_edge(edge_index, p=0.5, training=self.training)
            edge_index_norm, norm = gcn_norm(
                drop_edge_index, None, sz, self.improved,
                self.add_self_loops, self.flow, local_preds.dtype
            )

            new_prop = self.propagate(edge_index_norm, x=prop, norm=norm)
            prop = torch.where(active.unsqueeze(1), new_prop, prop)  # prop only for active nodes
            x = prop

            # policy (halting probability) and value for all nodes
            p_hidden = F.relu(self.policy_hidden1(x))
            p_hidden = F.relu(self.policy_hidden2(p_hidden))

            halt_logits = self.policy(p_hidden).view(-1)
            p = torch.sigmoid(halt_logits)

            v_hidden = F.relu(self.value_hidden1(x))
            v_hidden = F.relu(self.value_hidden2(v_hidden))
            v = self.value(v_hidden).view(-1)

            entropy = -(p * torch.log(p + 1e-10) + (1 - p) * torch.log(
                1 - p + 1e-10))  # entropy of the policy to encourage different halting and check out
            noise = torch.randn_like(
                p) * current_exploration  # the idea of adding noise was to also encourage different ranges of certainty.
            noisy_p = torch.clamp(p + noise, 0.01, 0.99)
            u = torch.rand_like(noisy_p)
            action = (u < noisy_p)
            halt = active & action  # the ones that halt!

            # for those who halts then:
            if halt.any():
                halting_log_prob[halt] = torch.log(noisy_p[halt] + 1e-10)
                halting_value[halt] = v[halt]
                halting_entropy[halt] = entropy[halt]

            active = active & (~halt)
            steps[active] = t + 2  # If in last iteration and still does not halt, sets number steps one too large

        # last case in case no halt
        if active.any():
            halting_log_prob[active] = torch.log(noisy_p[active] + 1e-10)
            halting_value[active] = v[active]
            halting_entropy[active] = entropy[active]
            steps[active] = self.niter  # Make sure number steps is not one too large (see comment above)

        policy_state = {
            'halting_log_prob': halting_log_prob,  # [num_nodes]
            'halting_value': halting_value,
            'halting_entropy': halting_entropy
        }
        return x, steps, policy_state

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j


class RL_AP_GCN(torch.nn.Module):
    def __init__(self,
                 dataset,
                 niter=10,
                 computation_penalty=0.0,  # 0.0005,
                 exploration_factor=0.1,
                 use_scheduled_penalty=True,
                 hidden=[64],
                 dropout=0.5):
        super(RL_AP_GCN, self).__init__()
        num_features = [dataset.data.x.shape[1]] + hidden + [dataset.num_classes]
        layers = []
        for in_features, out_features in zip(num_features[:-1], num_features[1:]):
            layers.append(Linear(in_features, out_features))
        self.prop = RLAdaptiveProp(
            niter=niter,
            h_size=dataset.num_classes,
            computation_penalty=computation_penalty,
            exploration_factor=exploration_factor,
            use_scheduled_penalty=use_scheduled_penalty
        )
        self.layers = ModuleList(layers)
        self.reg_params = list(layers[0].parameters())
        self.non_reg_params = [p for l in layers[1:] for p in l.parameters()] + list(self.prop.parameters())
        self.dropout = Dropout(p=dropout)
        self.act_fn = ReLU()
        self.reset_parameters()


    def reset_parameters(self):
        self.prop.reset_parameters()
        for layer in self.layers:
            layer.reset_parameters()

    def set_epoch(self, current, total):
        self.prop.set_epoch(current, total)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        for i, layer in enumerate(self.layers):
            x = layer(self.dropout(x))
            if i < len(self.layers) - 1:
                x = self.act_fn(x)
        x, steps, policy_state = self.prop(x, edge_index)
        return F.log_softmax(x, dim=1), steps, policy_state


class RL_AP_GCN_TrainArgs(TrainArgs):
    def __init__(self, learning_rate: float, normalizer, weight_decay: float, entropy_weight: float,
                 value_weight: float, max_grad_norm: float):
        super().__init__(learning_rate)
        self.normalizer = normalizer
        self.weight_decay = weight_decay
        self.entropy_weight = entropy_weight
        self.value_weight = value_weight
        self.max_grad_norm = max_grad_norm


class RL_AP_GCN_ModelArgs(ModelArgs):
    def __init__(self, dataset, niter: int, computation_penalty: float, exploration_factor: float,
                 use_scheduled_penalty: bool, hidden, dropout: float):
        self.dataset = dataset
        self.niter = niter
        self.computation_penalty = computation_penalty
        self.exploration_factor = exploration_factor
        self.use_scheduled_penalty = use_scheduled_penalty
        self.hidden = hidden
        self.dropout = dropout


class RL_AP_GCN_Integrator(Integrator):
    def train_epoch(self, model, data, optimizer, epoch: int, total_epochs: int, train_args: RL_AP_GCN_TrainArgs) -> (
    float, np.array):
        model.train()
        model.set_epoch(epoch, total_epochs)
        optimizer.zero_grad()

        logits, steps, policy_state = model(data)
        task_loss = F.nll_loss(logits[data.train_mask], data.y[data.train_mask])
        l2_reg = sum(torch.sum(param ** 2) for param in model.reg_params)
        supervised_loss = task_loss + train_args.weight_decay * 0.5 * l2_reg

        # RL part
        train_mask = data.train_mask
        pred = logits.argmax(dim=1)
        correctness = (pred == data.y).float()  # 1 if correct, 0 otherwise
        current_penalty = model.prop.computation_penalty
        node_steps = steps  # shape: [num_nodes]
        reward = correctness - current_penalty * node_steps  # per-node reward -> THIS CAN BE EVEN COMMENTED OUT BUT ANYWAY I SET IT TO 0.

        halting_value = policy_state['halting_value']
        advantage = reward - halting_value  # per-node advantage

        mask_idx = train_mask.nonzero(as_tuple=True)[0]
        halting_log_prob = policy_state['halting_log_prob']
        policy_loss = - (halting_log_prob[mask_idx] * advantage[mask_idx].detach()).mean()
        value_loss = F.mse_loss(halting_value[mask_idx], reward[mask_idx])
        halting_entropy = policy_state['halting_entropy']
        entropy_loss = - halting_entropy[mask_idx].mean() * train_args.entropy_weight

        rl_loss = policy_loss + train_args.value_weight * value_loss + entropy_loss
        total_loss = supervised_loss + rl_loss

        total_loss.backward()
        if train_args.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_args.max_grad_norm)
        optimizer.step()

        return total_loss.item(), steps.cpu().numpy()[train_mask.cpu().numpy()]

    def evaluate(self, model, data, mask, eval_args: EvalArgs = None) -> (float, np.array):
        model.eval()
        with torch.no_grad():
            logits, steps, _ = model(data)

            pred = logits[mask].max(1)[1]
            correct = pred.eq(data.y[mask]).sum().item()
            total = mask.sum().item()
            acc = correct / total if total > 0 else 0

        return acc, steps.cpu().numpy()[mask.cpu().numpy()]


def get_RL_AP_GCN_configuration(dataset, dataset_name):
    weight_decay = 0.008

    if dataset_name == Dataset.APHOTO.label or dataset_name == Dataset.ACOMPUTER.label:
        weight_decay = 0

    integrator = RL_AP_GCN_Integrator()
    train_args = RL_AP_GCN_TrainArgs(0.01, RewardNormalizer(), weight_decay, 0.01, 0.5, 1.0)
    model_args = RL_AP_GCN_ModelArgs(dataset, 10, 0.0, 0.01, False, [64], 0.5)

    return RL_AP_GCN, integrator, train_args, None, model_args
