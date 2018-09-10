import math

import torch
import torch.nn.functional as F
from torch.nn import Parameter
from torch_scatter import scatter_add


class GraphConv(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 norm=True,
                 bias=True,
                 dropout=0):
        super(GraphConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.norm = norm
        self.dropout = dropout
        self.weight = Parameter(torch.Tensor(in_channels, out_channels))
        self.root_weight = Parameter(torch.Tensor(in_channels, out_channels))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.in_channels)
        self.weight.data.uniform_(-stdv, stdv)
        self.root_weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, edge_index):
        if edge_index.numel() > 0:
            row, col = edge_index

            out = torch.mm(x, self.weight)
            out_col = out[col]

            # mask = out_col.new_ones(out_col.size(0))
            # mask = F.dropout(mask, self.dropout, training=self.training)
            # out_col = mask.view(-1, 1) * out_col

            out_col = F.dropout(out_col, self.dropout, training=self.training)

            out = scatter_add(out_col, row, dim=0, dim_size=x.size(0))

            # Normalize output by node degree.
            if self.norm:
                deg = scatter_add(
                    x.new_ones((row.size())), row, dim=0, dim_size=x.size(0))
                out = out / deg.unsqueeze(-1).clamp(min=1)

            # Weight root node separately.
            out = out + torch.mm(x, self.root_weight)
        else:
            out = torch.mm(x, self.root_weight)

        # Add bias (if wished).
        if self.bias is not None:
            out = out + self.bias

        return out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)
