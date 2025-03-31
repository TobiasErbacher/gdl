from typing import NamedTuple, Callable
from class_model import ModelType

class GumbelArgs(NamedTuple):
    learn_temp: bool
    temp_model_type: ModelType
    tau0: float
    temp: float
    gin_mlp_func: Callable