from typing import NamedTuple, Callable
from torch.nn import Linear, Dropout, ModuleList, Sequential
from class_model import ModelType
from class_activationtype import ActivationType
from class_encoder import PosEncoder, DataSetEncoders
from class_metric import MetricType
from class_concat2node import Concat2NodeEncoder

class EnvArgs(NamedTuple):
    model_type: ModelType
    num_layers: int
    env_dim: int

    layer_norm: bool
    skip: bool
    batch_norm: bool
    dropout: float
    act_type: ActivationType
    dec_num_layers: int
    pos_enc: PosEncoder
    dataset_encoders: DataSetEncoders

    metric_type: MetricType
    in_dim: int
    out_dim: int

    gin_mlp_func: Callable

    def load_net(self) -> ModuleList:
        if self.pos_enc is PosEncoder.NONE:
            enc_list = [self.dataset_encoders.node_encoder(in_dim=self.in_dim, emb_dim=self.env_dim)]
        else:
            if self.dataset_encoders is DataSetEncoders.NONE:
                enc_list = [self.pos_enc.get(in_dim=self.in_dim, emb_dim=self.env_dim)]
            else:
                enc_list = [Concat2NodeEncoder(enc1_cls=self.dataset_encoders.node_encoder,
                                               enc2_cls=self.pos_enc.get,
                                               in_dim=self.in_dim, emb_dim=self.env_dim,
                                               enc2_dim_pe=self.pos_enc.DIM_PE())]

        component_list =\
            self.model_type.get_component_list(in_dim=self.env_dim, hidden_dim=self.env_dim,  out_dim=self.env_dim,
                                               num_layers=self.num_layers, bias=True, edges_required=True,
                                               gin_mlp_func=self.gin_mlp_func)

        if self.dec_num_layers > 1:
            mlp_list = (self.dec_num_layers - 1) * [Linear(self.env_dim, self.env_dim),
                                                    Dropout(self.dropout), self.act_type.nn()]
            mlp_list = mlp_list + [Linear(self.env_dim, self.out_dim)]
            dec_list = [Sequential(*mlp_list)]
        else:
            dec_list = [Linear(self.env_dim, self.out_dim)]

        return ModuleList(enc_list + component_list + dec_list)