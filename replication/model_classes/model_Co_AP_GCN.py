from typing import Tuple

import math
import numpy as np
import torch
from torch import Tensor
from torch_geometric.typing import OptTensor, Adj
from torch.nn import CrossEntropyLoss

from replication.dataset import Dataset
from replication.model_classes.co_gnn_helper_classes.encoder_classes import DataSetEncoders
from replication.model_classes.co_gnn_helper_classes.type_classes import ModelType, ActivationType
from replication.model_classes.interfaces import Integrator, TrainArgs, EvalArgs, ModelArgs


# From Tobias main.ipynb


def load_env_net(in_dim: int, out_dim: int, env_args: dict) -> torch.nn.ModuleList:
    env_dim = env_args["env_dim"]
    num_layers = env_args["num_layers"]
    gin_mlp_func = env_args["gin_mlp_func"]
    dec_num_layers = env_args["dec_num_layers"]
    dropout = env_args["dropout"]
    act_type = env_args["act_type"]

    enc_list = [env_args["dataset_encoders"].node_encoder(in_dim=in_dim, emb_dim=env_dim)]

    component_list = env_args["model_type"].get_component_list(in_dim=env_dim,
                                                               hidden_dim=env_dim,
                                                               out_dim=env_dim,
                                                               num_layers=num_layers,
                                                               bias=True,
                                                               edges_required=True,
                                                               gin_mlp_func=gin_mlp_func)

    if dec_num_layers > 1:
        mlp_list = (dec_num_layers - 1) * [torch.nn.Linear(env_dim, env_dim), torch.nn.Dropout(dropout), act_type.nn()]
        mlp_list = mlp_list + [torch.nn.Linear(env_dim, out_dim)]
        dec_list = [torch.nn.Sequential(*mlp_list)]
    else:
        dec_list = [torch.nn.Linear(env_dim, out_dim)]

    return torch.nn.ModuleList(enc_list + component_list + dec_list)


def load_act_net(action_args: dict, env_args: dict) -> torch.nn.ModuleList:
    model_type = action_args["model_type"]
    env_dim = env_args["env_dim"]
    hidden_dim = action_args["hidden_dim"]
    num_layers = action_args["num_layers"]
    gin_mlp_func = action_args["gin_mlp_func"]

    net = model_type.get_component_list(in_dim=env_dim,
                                        hidden_dim=hidden_dim,
                                        out_dim=2,  # 2 because we can sample (edge, no edge)
                                        num_layers=num_layers,
                                        bias=True,
                                        edges_required=False,
                                        gin_mlp_func=gin_mlp_func)

    return torch.nn.ModuleList(net)


class ActionNet(torch.nn.Module):
    def __init__(self, action_args: dict, env_args: dict):
        """
        Create a model which represents the agent's policy.
        """
        super().__init__()
        self.num_layers = action_args["num_layers"]
        self.net = load_act_net(action_args=action_args, env_args=env_args)
        self.dropout = torch.nn.Dropout(action_args["dropout"])
        self.act = action_args["act_type"].get()

    def forward(self, x: Tensor, edge_index: Adj, env_edge_attr: OptTensor, act_edge_attr: OptTensor) -> Tensor:
        edge_attrs = [env_edge_attr] + (self.num_layers - 1) * [act_edge_attr]
        for (edge_attr, layer) in zip(edge_attrs[:-1], self.net[:-1]):
            x = layer(x=x, edge_index=edge_index, edge_attr=edge_attr)
            x = self.dropout(x)
            x = self.act(x)
        x = self.net[-1](x=x, edge_index=edge_index, edge_attr=edge_attrs[-1])
        return x


class TempSoftPlus(torch.nn.Module):
    """
        Network for dynamic temperature.
    """

    def __init__(self, gumbel_args: dict, env_dim: int):
        super(TempSoftPlus, self).__init__()
        model_list = gumbel_args["temp_model_type"].get_component_list(in_dim=env_dim,
                                                                       hidden_dim=env_dim,
                                                                       out_dim=1,
                                                                       num_layers=1,
                                                                       bias=False,
                                                                       edges_required=False,
                                                                       gin_mlp_func=gumbel_args["gin_mlp_func"])
        self.linear_model = torch.nn.ModuleList(model_list)
        self.softplus = torch.nn.Softplus(beta=1)
        self.tau0 = gumbel_args["tau0"]

    def forward(self, x: Tensor, edge_index: Adj, edge_attr: Tensor):
        x = self.linear_model[0](x=x, edge_index=edge_index, edge_attr=edge_attr)
        x = self.softplus(x) + self.tau0
        temp = x.pow_(-1)
        return temp.masked_fill_(temp == float("inf"), 0.0)


class CoGNN(torch.nn.Module):
    """
        CoGNN model class.
    """

    def __init__(
            self,
            dataset,
            gumbel_args: dict,
            env_args: dict,
            action_args: dict
    ):
        super(CoGNN, self).__init__()
        self.env_args = env_args
        self.learn_temp = gumbel_args["learn_temp"]
        if self.learn_temp:
            self.temp_model = TempSoftPlus(gumbel_args=gumbel_args, env_dim=env_args["env_dim"])
        self.temp = gumbel_args["temp"]

        self.num_layers = env_args["num_layers"]
        self.env_net = load_env_net(in_dim=dataset.data.x.shape[1], out_dim=dataset.num_classes, env_args=env_args)
        self.use_encoders = env_args["dataset_encoders"].use_encoders()

        layer_norm_cls = torch.nn.LayerNorm if env_args["layer_norm"] else torch.nn.Identity
        self.hidden_layer_norm = layer_norm_cls(env_args["env_dim"])
        self.skip = env_args["skip"]
        self.drop_ratio = env_args["dropout"]
        self.dropout = torch.nn.Dropout(p=self.drop_ratio)
        self.act = env_args["act_type"].get()
        self.in_act_net = ActionNet(action_args=action_args, env_args=env_args)
        self.out_act_net = ActionNet(action_args=action_args, env_args=env_args)

        # Encoder types
        self.dataset_encoder = env_args["dataset_encoders"]
        self.env_bond_encoder = self.dataset_encoder.edge_encoder(emb_dim=env_args["env_dim"],
                                                                  model_type=env_args["model_type"])
        self.act_bond_encoder = self.dataset_encoder.edge_encoder(emb_dim=action_args["hidden_dim"],
                                                                  model_type=action_args["model_type"])

    def forward(
            self,
            x: Tensor,
            edge_index: Adj,
            pestat,
            edge_attr: OptTensor = None,
    ) -> Tuple[Tensor, Tensor]:
        result = 0
        node_states = []

        # bond encode
        if edge_attr is None or self.env_bond_encoder is None:
            env_edge_embedding = None
        else:
            env_edge_embedding = self.env_bond_encoder(edge_attr)
        if edge_attr is None or self.act_bond_encoder is None:
            act_edge_embedding = None
        else:
            act_edge_embedding = self.act_bond_encoder(edge_attr)

        x = self.env_net[0](x, pestat)  # (N, F) encoder
        if not self.use_encoders:
            x = self.dropout(x)
            x = self.act(x)

        for gnn_idx in range(self.num_layers):
            x = self.hidden_layer_norm(x)

            # action
            in_logits = self.in_act_net(x=x, edge_index=edge_index, env_edge_attr=env_edge_embedding,
                                        act_edge_attr=act_edge_embedding)  # (N, 2)
            out_logits = self.out_act_net(x=x, edge_index=edge_index, env_edge_attr=env_edge_embedding,
                                          act_edge_attr=act_edge_embedding)  # (N, 2)

            temp = self.temp_model(x=x, edge_index=edge_index, edge_attr=env_edge_embedding) if self.learn_temp else self.temp
            in_probs = torch.nn.functional.gumbel_softmax(logits=in_logits, tau=temp, hard=True)
            out_probs = torch.nn.functional.gumbel_softmax(logits=out_logits, tau=temp, hard=True)
            edge_weight = self.create_edge_weight(edge_index=edge_index, keep_in_prob=in_probs[:, 0],
                                                  keep_out_prob=out_probs[:, 0])

            # environment
            out = self.env_net[1 + gnn_idx](x=x, edge_index=edge_index, edge_weight=edge_weight, edge_attr=env_edge_embedding)
            out = self.dropout(out)
            out = self.act(out)

            # Node state distribution: 0 -> standard, 1 -> broadcast, 2 -> listen, 3 -> isolate.
            active_edges = edge_index[:, edge_weight.bool()]
            nodes_u, nodes_v = active_edges
            broadcast_nodes = torch.unique(nodes_u)
            listen_nodes = torch.unique(nodes_v)
            standard_nodes = torch.tensor(list(set(broadcast_nodes.tolist()) & set(listen_nodes.tolist())),
                                          dtype=torch.long)
            node_state = torch.full((x.shape[0],), fill_value=3)  # initialize as isolate
            node_state[broadcast_nodes] = 1
            node_state[listen_nodes] = 2
            node_state[standard_nodes] = 0
            node_states.append(node_state)

            if self.skip:
                x = x + out
            else:
                x = out

        x = self.hidden_layer_norm(x)
        # Note: The authors use pooling but for our datasets this would always be None so we removed it from the code to simplify.
        x = self.env_net[-1](x)  # decoder
        result = result + x

        return result, node_states

    def create_edge_weight(
            self,
            edge_index: Adj,
            keep_in_prob: Tensor,
            keep_out_prob: Tensor
    ) -> Tensor:
        u, v = edge_index
        edge_in_prob = keep_in_prob[v]
        edge_out_prob = keep_out_prob[u]
        return edge_in_prob * edge_out_prob


class Co_AP_GCN_TrainArgs(TrainArgs):
    def __init__(self, learning_rate, halting_step, weight_decay, clip_grad):
        super().__init__(learning_rate)
        self.halting_step = halting_step
        self.weight_decay = weight_decay
        self.clip_grad = clip_grad


class Co_AP_GCN_ModelArgs(ModelArgs):
    def __init__(self, dataset, niter, gumbel_args: dict, env_args: dict, action_args: dict):
        # dataset and niter are required since they are used for all other models (do not remove although unused)
        self.dataset = dataset
        self.gumbel_args = gumbel_args
        self.env_args = env_args
        self.action_args = action_args


class Co_AP_GCN_Integrator(Integrator):
    def train_epoch(self, model, data, optimizer, epoch: int, total_epochs: int, train_args: Co_AP_GCN_TrainArgs) -> (float, np.array):
        model.train()
        loss = CrossEntropyLoss()
        optimizer.zero_grad()

        train_mask = data.train_mask

        predictions, _ = model(x=data.x,
                                  edge_index=data.edge_index,
                                  edge_attr=None,
                                  pestat=None)

        train_loss = loss(predictions[train_mask], data.y[train_mask])

        # Manual weight decay (L2 regularization)
        if train_args.weight_decay > 0:
            l2_reg = torch.tensor(0.0, device=train_loss.device)
            for param in model.parameters():
                if param.requires_grad:
                    l2_reg += torch.norm(param, p=2) ** 2

            train_loss += train_args.weight_decay * l2_reg

        train_loss.backward()

        #norm = torch.tensor(0.0, device=train_loss.device)
        #for name, param in model.named_parameters():
        #    if param.requires_grad:
        #        norm += torch.norm(param, p=2)

        #if epoch % 10 == 0:
        #    print(f"Epoch {epoch} - Parameter norm: {norm.item():.4f}")

        # Gradient norm (L2) of all parameters
        #grad_norm = torch.tensor(0.0, device=train_loss.device)
        #for param in model.parameters():
        #    if param.requires_grad and param.grad is not None:
        #        grad_norm += torch.norm(param.grad, p=2)

        #if epoch % 10 == 0:
        #    print(f"Epoch {epoch} - Gradient norm before clipping: {grad_norm.item():.4f}")

        if train_args.clip_grad:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)

        optimizer.step()

        return train_loss, np.zeros(len(predictions))

    def _get_state_distribution(self, node_states: list) -> np.array:
        distribution_per_layer = []
        for layer in node_states:
            # 0 -> standard, 1 -> broadcast, 2 -> listen, 3 -> isolate
            standard = layer[layer == 0].shape[0] / layer.shape[0]
            broadcast = layer[layer == 1].shape[0] / layer.shape[0]
            listen = layer[layer == 2].shape[0] / layer.shape[0]
            isolate = layer[layer == 3].shape[0] / layer.shape[0]
            if abs(1.0 - sum([standard, broadcast, listen, isolate])) >= 1e-6:
                print("Warning: state distribution does not sum to 1.")
            distribution_per_layer.append([standard, broadcast, listen, isolate])
        return np.array(distribution_per_layer)

    def evaluate(self, model, data, mask, eval_args: EvalArgs = None) -> (float, np.array):
        model.eval()

        with torch.no_grad():
            predictions, node_states = model(x=data.x,
                                             edge_index=data.edge_index,
                                             edge_attr=None,
                                             pestat=None)

            scores_np = predictions[mask].detach().cpu().numpy()
            labels_np = data.y[mask].detach().cpu().numpy()
            pred_labels = scores_np.argmax(axis=1)
            correct = (pred_labels == labels_np).sum().item()
            total = mask.sum().item()
            accuracy = correct / total

            node_state_distribution = self._get_state_distribution(node_states=node_states)

        return accuracy, node_state_distribution


def get_Cooperative_AP_GCN_configuration(dataset, dataset_name):
    clip_grad = False
    learning_rate = 0.005
    weight_decay = 0.003

    gumbel_args = {
        "tau0": 0.5,
        "learn_temp": True,
        "temp": 0.5,
        "temp_model_type": ModelType.GCN,
        "gin_mlp_func": None,  # Can set to None since we have model_type=ModelType.GCN
    }

    env_args = {
        "num_layers": 3,
        "env_dim": 32,
        "dropout": 0.5,
        "dec_num_layers": 1,
        "layer_norm": True,
        "skip": False,
        "act_type": ActivationType.RELU,
        "model_type": ModelType.GCN,
        "dataset_encoders": DataSetEncoders.NONE,
        "gin_mlp_func": None  # Can set to None since we have model_type=ModelType.GCN
    }

    action_args = {
        "hidden_dim": 64,
        "num_layers": 2,
        "dropout": 0.5,
        "model_type": ModelType.GCN,
        "act_type": ActivationType.RELU,
        "gin_mlp_func": None  # Can set to None since we have model_type=ModelType.GCN
    }

    if dataset_name == Dataset.CITESEER.label:
        pass
    elif dataset_name == Dataset.CORAML.label:
        pass
    elif dataset_name == Dataset.PUBMED.label:
        pass
    elif dataset_name == Dataset.MSACADEMIC.label:
        pass
    elif dataset_name == Dataset.ACOMPUTER.label:
        learning_rate = 0.001

    integrator = Co_AP_GCN_Integrator()
    train_args = Co_AP_GCN_TrainArgs(learning_rate, None, weight_decay, clip_grad)
    model_args = Co_AP_GCN_ModelArgs(dataset, None, gumbel_args, env_args, action_args)

    return CoGNN, integrator, train_args, None, model_args
