# From Jonathan/APGCN_WITH_PLOTS
import math
import torch
from torch.nn import Dropout, Linear, ReLU, ModuleList
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.utils import dropout_edge


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
        for _ in range(self.niter):
            old_prop = prop  # h^(t-1)

            continue_fmask = continue_mask.float().to(local_preds.device)
            drop_edge_index, _ = dropout_edge(edge_index, p=0.5, training=self.training)  # default is 0.5 as they did.

            # GCN normalization using the util that is now available.
            # -> https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/conv/gcn_conv.html#GCNConv
            edge_index_norm, norm = gcn_norm(
                drop_edge_index, None,
                sz, self.improved,
                self.add_self_loops,
                self.flow, local_preds.dtype
            )

            prop = self.propagate(edge_index_norm, x=prop, norm=norm)
            h = torch.sigmoid(self.halt(prop)).t().squeeze()  # h^k_i = non-linearity(Qz^k_i + q)

            # here we do the soft update based on equation 7
            # K_i = min{k : sum(j=1 to k) h^j_i >= 1 - eps}
            # 0.99 is equivalent to (1 - eps) where eps = 0.01
            prob_mask = (((sum_h + h) < 0.99) & continue_mask).squeeze()
            prob_fmask = prob_mask.float().to(local_preds.device)

            # we add another step for those nodes that continue and that the accum prob is lower than threshold.
            steps = steps + prob_fmask
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
            # Note that this is permanent
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
