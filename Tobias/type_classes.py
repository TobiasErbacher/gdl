# OFFICIAL LIBRARY IMPORTS ------------------------------------------------------------------------
from torch import from_numpy, tensor
from torch.nn import Module, ReLU, GELU, CrossEntropyLoss
import torch.nn.functional as F
from torchmetrics import Accuracy
from torch_geometric.data import Data
from torch_geometric.nn.pool import global_mean_pool, global_add_pool
import numpy as np
from math import inf
from enum import Enum, auto
from typing import Callable, List, NamedTuple

# CUSTOM IMPORTS ----------------------------------------------------------------------------------
from architectures import WeightedGCNConv, WeightedGINConv, WeightedGNNConv, GraphLinear, BatchIdentity

# CUSTOM CLASSES ----------------------------------------------------------------------------------
class ModelType(Enum):
    """
        Class for different model architectures.
    """
    GCN = auto()
    GIN = auto()
    LIN = auto()

    SUM_GNN = auto()
    MEAN_GNN = auto()

    @staticmethod
    def from_string(s: str):
        try:
            return ModelType[s]
        except KeyError:
            raise ValueError()

    def load_component_cls(self):
        if self is ModelType.GCN:
            return WeightedGCNConv
        elif self is ModelType.GIN:
            return WeightedGINConv
        elif self in [ModelType.SUM_GNN, ModelType.MEAN_GNN]:
            return WeightedGNNConv
        elif self is ModelType.LIN:
            return GraphLinear
        else:
            raise ValueError(f"model {self.name} not supported")

    def is_gcn(self):
        return self is ModelType.GCN

    def get_component_list(self, in_dim: int, hidden_dim: int, out_dim: int, num_layers: int, bias: bool,
                           edges_required: bool, gin_mlp_func: Callable) -> List[Module]:
        dim_list = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]
        if self is ModelType.GCN:
            component_list = [self.load_component_cls()(in_channels=in_dim_i, out_channels=out_dim_i, bias=bias)
                              for in_dim_i, out_dim_i in zip(dim_list[:-1], dim_list[1:])]
        elif self is ModelType.GIN:
            component_list = [self.load_component_cls()(in_channels=in_dim_i, out_channels=out_dim_i, bias=bias, mlp_func=gin_mlp_func) 
                              for in_dim_i, out_dim_i in zip(dim_list[:-1], dim_list[1:])]
        elif self in [ModelType.SUM_GNN, ModelType.MEAN_GNN]:
            aggr = "mean" if self is ModelType.MEAN_GNN else "sum"
            component_list = [self.load_component_cls()(in_channels=in_dim_i, out_channels=out_dim_i, aggr=aggr, bias=bias) 
                              for in_dim_i, out_dim_i in zip(dim_list[:-1], dim_list[1:])]
        elif self is ModelType.LIN:
            assert not edges_required, f"env does not support {self.name}"
            component_list = [self.load_component_cls()(in_features=in_dim_i, out_features=out_dim_i, bias=bias) 
                              for in_dim_i, out_dim_i in zip(dim_list[:-1], dim_list[1:])]
        else:
            raise ValueError(f"model {self.name} not supported")
        return component_list

class ActivationType(Enum):
    """
        Class for the different activation types.
    """
    RELU = auto()
    GELU = auto()

    @staticmethod
    def from_string(s: str):
        try:
            return ActivationType[s]
        except KeyError:
            raise ValueError()

    def get(self):
        if self is ActivationType.RELU:
            return F.relu
        elif self is ActivationType.GELU:
            return F.gelu
        else:
            raise ValueError(f'ActivationType {self.name} not supported')

    def nn(self) -> Module:
        if self is ActivationType.RELU:
            return ReLU()
        elif self is ActivationType.GELU:
            return GELU()
        else:
            raise ValueError(f'ActivationType {self.name} not supported')

class LossesAndMetrics(NamedTuple):
    """
        Class the metric losses and metrics.
    """
    train_loss: float
    val_loss: float
    test_loss: float
    train_metric: float
    val_metric: float
    test_metric: float

    def get_fold_metrics(self):
        return tensor([self.train_metric, self.val_metric, self.test_metric])

class Metric:
    def __init__(self, task: str, num_classes: int, **kwargs):

        self.metric = Accuracy(task=task, num_classes=num_classes, **kwargs)
        self.is_classification = True
        self.is_multilabel = False
        self.higher_is_better = True
        self.task_loss = CrossEntropyLoss()
        self.get_worst_losses_n_metrics = LossesAndMetrics(train_loss=inf, val_loss=inf, test_loss=inf, train_metric=-inf, val_metric=-inf, test_metric=-inf)
    
    def apply_metric(self, scores: np.ndarray, target: np.ndarray) -> float:
        #num_classes = scores.size(1)  # target.max().item() + 1
        if isinstance(scores, np.ndarray):
            scores = from_numpy(scores)
        if isinstance(target, np.ndarray):
            target = from_numpy(target)
        self.metric.to(scores.device)
        return self.metric(scores, target).item()
    
    def get_out_dim(self, data: Data) -> int:
        return int(data.y.max().item() + 1)
        
    def src_better_than_other(self, src: float, other: float) -> bool:
        return src > other

class Pool(Enum):
    """
        an object for the different activation types
    """
    NONE = auto()
    MEAN = auto()
    SUM = auto()

    @staticmethod
    def from_string(s: str):
        try:
            return Pool[s]
        except KeyError:
            raise ValueError()

    def get(self):
        if self is Pool.MEAN:
            return global_mean_pool
        elif self is Pool.SUM:
            return global_add_pool
        elif self is Pool.NONE:
            return BatchIdentity()
        else:
            raise ValueError(f'Pool {self.name} not supported')