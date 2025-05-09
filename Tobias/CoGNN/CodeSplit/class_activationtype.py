from enum import Enum, auto
from torch.nn import Module, ReLU, GELU
import torch.nn.functional as F

class ActivationType(Enum):
    """
        an object for the different activation types
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