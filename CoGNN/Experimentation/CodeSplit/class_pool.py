from enum import Enum, auto
from torch.nn import Module
from torch import Tensor
from typing import Any
from torch_geometric.nn.pool import global_mean_pool, global_add_pool

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

class BatchIdentity(Module):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()

    def forward(self, x: Tensor, batch: Tensor) -> Tensor:
        return x