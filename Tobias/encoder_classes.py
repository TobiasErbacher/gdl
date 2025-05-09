from enum import Enum, auto
from torch import cat, rand, isnan, sum, Tensor
from torch.nn import Module, ModuleList, Linear, Sequential, BatchNorm1d, ReLU, TransformerEncoder, TransformerEncoderLayer, Embedding
from torch.nn.init import xavier_uniform_
from ogb.utils.features import get_atom_feature_dims, get_bond_feature_dims
from torch_geometric.data import Data

KER_DIM_PE = 28
NUM_RW_STEPS = 20
KER_MODEL = 'Linear' # renamed from original code
KER_LAYERS = 3 # renamed from original code
KER_RAW_NORM_TYPE = 'BatchNorm' # renamed from original code
KER_PASS_AS_VAR = False # renamed from original code

LAP_DIM_PE = 16
LAP_MODEL = 'DeepSet' # renamed from original code
LAP_LAYERS = 2 # renamed from original code
N_HEADS = 4
POST_LAYERS = 0
LAP_MAX_FREQS = 10
LAP_RAW_NORM_TYPE = 'none' # renamed from original code
LAP_PASS_AS_VAR = False # renamed from original code

class LapPENodeEncoder(Module):
    """Laplace Positional Embedding node encoder.

    LapPE of size dim_pe will get appended to each node feature vector.
    If `expand_x` set True, original node features will be first linearly
    projected to (dim_emb - dim_pe) size and the concatenated with LapPE.

    Args:
        dim_emb: Size of final node embedding
        expand_x: Expand node features `x` from dim_in to (dim_emb - dim_pe)
    """

    def __init__(self, dim_in, dim_emb, expand_x=True):
        super().__init__()
        dim_pe = LAP_DIM_PE  # Size of Laplace PE embedding
        model_type = LAP_MODEL  # Encoder NN model type for PEs
        if model_type not in ['Transformer', 'DeepSet']:
            raise ValueError(f"Unexpected PE model {model_type}")
        self.model_type = model_type
        n_layers = LAP_LAYERS  # Num. layers in PE encoder model
        n_heads = N_HEADS  # Num. attention heads in Trf PE encoder
        post_n_layers = POST_LAYERS  # Num. layers to apply after pooling
        max_freqs = LAP_MAX_FREQS  # Num. eigenvectors (frequencies)
        norm_type = LAP_RAW_NORM_TYPE.lower()  # Raw PE normalization layer type
        self.pass_as_var = LAP_PASS_AS_VAR  # Pass PE also as a separate variable

        if dim_emb - dim_pe < 0: # formerly 1, but you could have zero feature size
            raise ValueError(f"LapPE size {dim_pe} is too large for "
                             f"desired embedding size of {dim_emb}.")

        if expand_x and dim_emb - dim_pe > 0:
            self.linear_x = Linear(dim_in, dim_emb - dim_pe)
        self.expand_x = expand_x and dim_emb - dim_pe > 0

        # Initial projection of eigenvalue and the node's eigenvector value
        self.linear_A = Linear(2, dim_pe)
        if norm_type == 'batchnorm':
            self.raw_norm = BatchNorm1d(max_freqs)
        else:
            self.raw_norm = None

        activation = ReLU  # register.act_dict[cfg.gnn.act]
        if model_type == 'Transformer':
            # Transformer model for LapPE
            encoder_layer = TransformerEncoderLayer(d_model=dim_pe,
                                                       nhead=n_heads,
                                                       batch_first=True)
            self.pe_encoder = TransformerEncoder(encoder_layer,
                                                    num_layers=n_layers)
        else:
            # DeepSet model for LapPE
            layers = []
            if n_layers == 1:
                layers.append(activation())
            else:
                self.linear_A = Linear(2, 2 * dim_pe)
                layers.append(activation())
                for _ in range(n_layers - 2):
                    layers.append(Linear(2 * dim_pe, 2 * dim_pe))
                    layers.append(activation())
                layers.append(Linear(2 * dim_pe, dim_pe))
                layers.append(activation())
            self.pe_encoder = Sequential(*layers)

        self.post_mlp = None
        if post_n_layers > 0:
            # MLP to apply post pooling
            layers = []
            if post_n_layers == 1:
                layers.append(Linear(dim_pe, dim_pe))
                layers.append(activation())
            else:
                layers.append(Linear(dim_pe, 2 * dim_pe))
                layers.append(activation())
                for _ in range(post_n_layers - 2):
                    layers.append(Linear(2 * dim_pe, 2 * dim_pe))
                    layers.append(activation())
                layers.append(Linear(2 * dim_pe, dim_pe))
                layers.append(activation())
            self.post_mlp = Sequential(*layers)

    def forward(self, x, pestat):
        EigVals = pestat[0]
        EigVecs = pestat[1]

        if self.training:
            sign_flip = rand(EigVecs.size(1), device=EigVecs.device)
            sign_flip[sign_flip >= 0.5] = 1.0
            sign_flip[sign_flip < 0.5] = -1.0
            EigVecs = EigVecs * sign_flip.unsqueeze(0)

        pos_enc = cat((EigVecs.unsqueeze(2), EigVals), dim=2) # (Num nodes) x (Num Eigenvectors) x 2
        empty_mask = isnan(pos_enc)  # (Num nodes) x (Num Eigenvectors) x 2

        pos_enc[empty_mask] = 0  # (Num nodes) x (Num Eigenvectors) x 2
        if self.raw_norm:
            pos_enc = self.raw_norm(pos_enc)
        pos_enc = self.linear_A(pos_enc)  # (Num nodes) x (Num Eigenvectors) x dim_pe

        # PE encoder: a Transformer or DeepSet model
        if self.model_type == 'Transformer':
            pos_enc = self.pe_encoder(src=pos_enc,
                                      src_key_padding_mask=empty_mask[:, :, 0])
        else:
            pos_enc = self.pe_encoder(pos_enc)

        # Remove masked sequences; must clone before overwriting masked elements
        pos_enc = pos_enc.clone().masked_fill_(empty_mask[:, :, 0].unsqueeze(2), 0.)

        # Sum pooling
        pos_enc = sum(pos_enc, 1, keepdim=False)  # (Num nodes) x dim_pe

        # MLP post pooling
        if self.post_mlp is not None:
            pos_enc = self.post_mlp(pos_enc)  # (Num nodes) x dim_pe

        # Expand node features if needed
        if self.expand_x:
            h = self.linear_x(x)
        else:
            h = x
        # Concatenate final PEs to input embedding
        x = cat((h, pos_enc), 1)
        return x

class KernelPENodeEncoder(Module):
    """Configurable kernel-based Positional Encoding node encoder.

    The choice of which kernel-based statistics to use is configurable through
    setting of `kernel_type`. Based on this, the appropriate config is selected,
    and also the appropriate variable with precomputed kernel stats is then
    selected from PyG Data graphs in `forward` function.
    E.g., supported are 'RWSE', 'HKdiagSE', 'ElstaticSE'.

    PE of size `dim_pe` will get appended to each node feature vector.
    If `expand_x` set True, original node features will be first linearly
    projected to (dim_emb - dim_pe) size and the concatenated with PE.

    Args:
        dim_emb: Size of final node embedding
        expand_x: Expand node features `x` from dim_in to (dim_emb - dim_pe)
    """

    kernel_type = None  # Instantiated type of the KernelPE, e.g. RWSE

    def __init__(self, dim_in, dim_emb, expand_x=True):
        super().__init__()
        if self.kernel_type is None:
            raise ValueError(f"{self.__class__.__name__} has to be "
                             f"preconfigured by setting 'kernel_type' class"
                             f"variable before calling the constructor.")

        dim_pe = KER_DIM_PE  # Size of the kernel-based PE embedding
        num_rw_steps = NUM_RW_STEPS
        model_type = KER_MODEL.lower()  # Encoder NN model type for PEs
        n_layers = KER_LAYERS  # Num. layers in PE encoder model
        norm_type = KER_RAW_NORM_TYPE.lower()  # Raw PE normalization layer type
        self.pass_as_var = KER_PASS_AS_VAR  # Pass PE also as a separate variable

        if dim_emb - dim_pe < 0: # formerly 1, but you could have zero feature size
            raise ValueError(f"PE dim size {dim_pe} is too large for "
                             f"desired embedding size of {dim_emb}.")

        if expand_x and dim_emb - dim_pe > 0:
            self.linear_x = Linear(dim_in, dim_emb - dim_pe)
        self.expand_x = expand_x and dim_emb - dim_pe > 0

        if norm_type == 'batchnorm':
            self.raw_norm = BatchNorm1d(num_rw_steps)
        else:
            self.raw_norm = None

        activation = ReLU  # register.act_dict[cfg.gnn.act]
        if model_type == 'mlp':
            layers = []
            if n_layers == 1:
                layers.append(Linear(num_rw_steps, dim_pe))
                layers.append(activation())
            else:
                layers.append(Linear(num_rw_steps, 2 * dim_pe))
                layers.append(activation())
                for _ in range(n_layers - 2):
                    layers.append(Linear(2 * dim_pe, 2 * dim_pe))
                    layers.append(activation())
                layers.append(Linear(2 * dim_pe, dim_pe))
                layers.append(activation())
            self.pe_encoder = Sequential(*layers)
        elif model_type == 'linear':
            self.pe_encoder = Linear(num_rw_steps, dim_pe)
        else:
            raise ValueError(f"{self.__class__.__name__}: Does not support "
                             f"'{model_type}' encoder model.")

    def forward(self, x, pestat):
        pos_enc = pestat  # (Num nodes) x (Num kernel times)
        # pos_enc = batch.rw_landing  # (Num nodes) x (Num kernel times)
        if self.raw_norm:
            pos_enc = self.raw_norm(pos_enc)
        pos_enc = self.pe_encoder(pos_enc)  # (Num nodes) x dim_pe

        # Expand node features if needed
        if self.expand_x:
            h = self.linear_x(x)
        else:
            h = x
        # Concatenate final PEs to input embedding
        x = cat((h, pos_enc), 1)
        return x

class RWSENodeEncoder(KernelPENodeEncoder):
    """
        Random Walk Structural Encoding node encoder.
    """
    kernel_type = 'RWSE'

class PosEncoder(Enum):
    """
        Class for the different encoders.
    """
    NONE = auto()
    LAP = auto()
    RWSE = auto()

    @staticmethod
    def from_string(s: str):
        try:
            return PosEncoder[s]
        except KeyError:
            raise ValueError()

    def get(self, in_dim: int, emb_dim: int, expand_x: bool):
        if self is PosEncoder.NONE:
            return None
        elif self is PosEncoder.LAP:
            return LapPENodeEncoder(dim_in=in_dim, dim_emb=emb_dim, expand_x=expand_x)
        elif self is PosEncoder.RWSE:
            return RWSENodeEncoder(dim_in=in_dim, dim_emb=emb_dim, expand_x=expand_x)
        else:
            raise ValueError(f'DataSetEncoders {self.name} not supported')

    def DIM_PE(self):
        if self is PosEncoder.NONE:
            return None
        elif self is PosEncoder.LAP:
            return LAP_DIM_PE
        elif self is PosEncoder.RWSE:
            return KER_DIM_PE
        else:
            raise ValueError(f'DataSetEncoders {self.name} not supported')

    def get_pe(self, data: Data, device):
        if self is PosEncoder.NONE:
            return None
        elif self is PosEncoder.LAP:
            return [data.EigVals.to(device), data.EigVecs.to(device)]
        elif self is PosEncoder.RWSE:
            return data.pestat_RWSE.to(device)
        else:
            raise ValueError(f'DataSetEncoders {self.name} not supported')

class EncoderLinear(Linear):
    def forward(self, x: Tensor, pestat=None) -> Tensor:
        return super().forward(x)

class AtomEncoder(Module):
    def __init__(self, emb_dim):
        super(AtomEncoder, self).__init__()

        self.atom_embedding_list = ModuleList()

        for i, dim in enumerate(get_atom_feature_dims()):
            emb = Embedding(dim, emb_dim)
            xavier_uniform_(emb.weight.data)
            self.atom_embedding_list.append(emb)

    def forward(self, x, pestat):
        x_embedding = 0
        for i in range(x.shape[1]):
            x_embedding += self.atom_embedding_list[i](x[:, i])

        return x_embedding

class BondEncoder(Module):

    def __init__(self, emb_dim):
        super(BondEncoder, self).__init__()

        self.bond_embedding_list = ModuleList()

        for i, dim in enumerate(get_bond_feature_dims()):
            emb = Embedding(dim, emb_dim)
            xavier_uniform_(emb.weight.data)
            self.bond_embedding_list.append(emb)

    def forward(self, edge_attr):
        bond_embedding = 0
        for i in range(edge_attr.shape[1]):
            bond_embedding += self.bond_embedding_list[i](edge_attr[:, i])

        return bond_embedding

class DataSetEncoders(Enum):
    """
        an object for the different encoders
    """
    NONE = auto()
    MOL = auto()

    @staticmethod
    def from_string(s: str):
        try:
            return DataSetEncoders[s]
        except KeyError:
            raise ValueError()

    def node_encoder(self, in_dim: int, emb_dim: int):
        if self is DataSetEncoders.NONE:
            return EncoderLinear(in_features=in_dim, out_features=emb_dim)
        elif self is DataSetEncoders.MOL:
            return AtomEncoder(emb_dim)
        else:
            raise ValueError(f'DataSetEncoders {self.name} not supported')

    def edge_encoder(self, emb_dim: int, model_type):
        if self is DataSetEncoders.NONE:
            return None
        elif self is DataSetEncoders.MOL:
            if model_type.is_gcn():
                return None
            else:
                return BondEncoder(emb_dim)
        else:
            raise ValueError(f'DataSetEncoders {self.name} not supported')

    def use_encoders(self) -> bool:
        return self is not DataSetEncoders.NONE