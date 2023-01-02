import torch
import torch.nn as nn
from torch_scatter import scatter

from typing import Optional, Tuple


from torch import Tensor
from torch_scatter import scatter_add

def to_dense_batch(x: Tensor, batch: Optional[Tensor] = None,
                   fill_value: float = 0., max_num_nodes: Optional[int] = None,
                   batch_size: Optional[int] = None) -> Tuple[Tensor, Tensor]:
    r"""Given a sparse batch of node features
    :math:`\mathbf{X} \in \mathbb{R}^{(N_1 + \ldots + N_B) \times F}` (with
    :math:`N_i` indicating the number of nodes in graph :math:`i`), creates a
    dense node feature tensor
    :math:`\mathbf{X} \in \mathbb{R}^{B \times N_{\max} \times F}` (with
    :math:`N_{\max} = \max_i^B N_i`).
    In addition, a mask of shape :math:`\mathbf{M} \in \{ 0, 1 \}^{B \times
    N_{\max}}` is returned, holding information about the existence of
    fake-nodes in the dense representation.

    Args:
        x (Tensor): Node feature matrix
            :math:`\mathbf{X} \in \mathbb{R}^{(N_1 + \ldots + N_B) \times F}`.
        batch (LongTensor, optional): Batch vector
            :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns each
            node to a specific example. Must be ordered. (default: :obj:`None`)
        fill_value (float, optional): The value for invalid entries in the
            resulting dense output tensor. (default: :obj:`0`)
        max_num_nodes (int, optional): The size of the output node dimension.
            (default: :obj:`None`)
        batch_size (int, optional) The batch size. (default: :obj:`None`)

    :rtype: (:class:`Tensor`, :class:`BoolTensor`)
    """
    if batch is None and max_num_nodes is None:
        mask = torch.ones(1, x.size(0), dtype=torch.bool, device=x.device)
        return x.unsqueeze(0), mask

    if batch is None:
        batch = x.new_zeros(x.size(0), dtype=torch.long)

    if batch_size is None:
        batch_size = int(batch.max()) + 1

    num_nodes = scatter_add(batch.new_ones(x.size(0)), batch, dim=0,
                            dim_size=batch_size)
    cum_nodes = torch.cat([batch.new_zeros(1), num_nodes.cumsum(dim=0)])

    if max_num_nodes is None:
        max_num_nodes = int(num_nodes.max())

    idx = torch.arange(batch.size(0), dtype=torch.long, device=x.device)
    idx = (idx - cum_nodes[batch]) + (batch * max_num_nodes)

    size = [batch_size * max_num_nodes] + list(x.size())[1:]
    out = x.new_full(size, fill_value)
    out[idx] = x
    out = out.view([batch_size, max_num_nodes] + list(x.size())[1:])

    mask = torch.zeros(batch_size * max_num_nodes, dtype=torch.bool,
                       device=x.device)
    mask[idx] = 1
    mask = mask.view(batch_size, max_num_nodes)

    return out, mask


def to_dense_adj(edge_index, batch=None, edge_attr=None, max_num_nodes=None):
    r"""Converts batched sparse adjacency matrices given by edge indices and
    edge attributes to a single dense batched adjacency matrix.

    Args:
        edge_index (LongTensor): The edge indices.
        batch (LongTensor, optional): Batch vector
            :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns each
            node to a specific example. (default: :obj:`None`)
        edge_attr (Tensor, optional): Edge weights or multi-dimensional edge
            features. (default: :obj:`None`)
        max_num_nodes (int, optional): The size of the output node dimension.
            (default: :obj:`None`)

    :rtype: :class:`Tensor`
    """
    if batch is None:
        batch = edge_index.new_zeros(edge_index.max().item() + 1)

    batch_size = batch.max().item() + 1
    one = batch.new_ones(batch.size(0))
    num_nodes = scatter(one, batch, dim=0, dim_size=batch_size, reduce='add')
    cum_nodes = torch.cat([batch.new_zeros(1), num_nodes.cumsum(dim=0)])

    idx0 = batch[edge_index[0]]
    idx1 = edge_index[0] - cum_nodes[batch][edge_index[0]]
    idx2 = edge_index[1] - cum_nodes[batch][edge_index[1]]

    if max_num_nodes is None:
        max_num_nodes = num_nodes.max().item()

    elif idx1.max() >= max_num_nodes or idx2.max() >= max_num_nodes:
        mask = (idx1 < max_num_nodes) & (idx2 < max_num_nodes)
        idx0 = idx0[mask]
        idx1 = idx1[mask]
        idx2 = idx2[mask]
        edge_attr = None if edge_attr is None else edge_attr[mask]

    if edge_attr is None:
        edge_attr = torch.ones(idx0.numel(), device=edge_index.device)

    size = [batch_size, max_num_nodes, max_num_nodes]
    size += list(edge_attr.size())[1:]
    adj = torch.zeros(size, dtype=edge_attr.dtype, device=edge_index.device)

    flattened_size = batch_size * max_num_nodes * max_num_nodes
    adj = adj.view([flattened_size] + list(adj.size())[3:])
    idx = idx0 * max_num_nodes * max_num_nodes + idx1 * max_num_nodes + idx2
    scatter(edge_attr, idx, dim=0, out=adj, reduce='add')
    adj = adj.view(size)

    return adj


class MaskedSoftmax(nn.Module):
    def __init__(self):
        super(MaskedSoftmax, self).__init__()
        self.softmax = nn.Softmax(1)

    def forward(self, x, mask=None):
        """
        Performs masked softmax, as simply masking post-softmax can be
        inaccurate
        :param x: [batch_size, num_items]
        :param mask: [batch_size, num_items]
        :return:
        """
        if mask is not None:
            mask = mask.float()
        if mask is not None:
            x_masked = x * mask + (1 - 1 / mask)
        else:
            x_masked = x
        x_max = x_masked.max(1)[0]
        x_exp = (x - x_max.unsqueeze(-1)).exp()
        if mask is not None:
            x_exp = x_exp * mask.float()
        return x_exp / x_exp.sum(1).unsqueeze(-1)

def masked_softmax(x, mask, **kwargs):
    x_masked = x.clone()
    x_masked[mask == 0] = -float("inf")

    return torch.softmax(x_masked, **kwargs)