from typing import NamedTuple, Callable
from class_model import ModelType
from class_activationtype import ActivationType
from torch.nn import ModuleList

class ActionNetArgs(NamedTuple):
    model_type: ModelType
    num_layers: int
    hidden_dim: int

    dropout: float
    act_type: ActivationType

    env_dim: int
    gin_mlp_func: Callable
    
    def load_net(self) -> ModuleList:
        net = self.model_type.get_component_list(in_dim=self.env_dim, hidden_dim=self.hidden_dim, out_dim=2,
                                                 num_layers=self.num_layers, bias=True, edges_required=False,
                                                 gin_mlp_func=self.gin_mlp_func)
        return ModuleList(net)