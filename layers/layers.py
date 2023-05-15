"""Euclidean layers."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter

from torch_geometric.nn import SAGEConv, GCNConv

import numpy as np


def get_dim_act(args):
    """
    Helper function to get dimension and activation at every layer.
    :param args:
    :return:
    """
    if not args.act:
        act = lambda x: x
    else:
        act = getattr(F, args.act)
    acts = [act] * (args.num_layers - 1)
    dims = [args.feat_dim] + ([args.dim] * (args.num_layers - 1))
    if args.task in ['lp', 'rec']:
        dims += [args.dim]
        acts += [act]
    return dims, acts


class GraphConvolution(Module):
    """
    Simple GCN layer.
    """

    def __init__(self, in_features, out_features, dropout, act, use_bias):
        super(GraphConvolution, self).__init__()
        self.dropout = dropout
        self.linear = nn.Linear(in_features, out_features, use_bias)
        self.act = act
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, input):
        x, adj = input
        hidden = self.linear.forward(x)
        hidden = F.dropout(hidden, self.dropout, training=self.training)
        if adj.is_sparse:
            support = torch.spmm(adj, hidden)
        else:
            support = torch.mm(adj, hidden)
        output = self.act(support), adj
        return output

    def extra_repr(self):
        return 'input_dim={}, output_dim={}'.format(
                self.in_features, self.out_features
        )


class GraphConvolutionTorch(Module):
    """
    Simple GCN layer with torch
    """

    def __init__(self, in_features, out_features, dropout, act, use_bias):
        super(GraphConvolutionTorch, self).__init__()
        self.dropout = dropout
        self.gcn = GCNConv(in_features, out_features, bias=use_bias)
        torch.nn.init.xavier_uniform(self.gcn.lin.weight)
        self.act = act
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, input):
        x, adj = input
        if not hasattr(self, 'edge_index'):
            self.edge_index = torch.Tensor(np.stack(np.where(adj.detach().cpu().to_dense()))).to(dtype=torch.long, device=x.get_device())
        support = self.gcn(x, self.edge_index)
        output = self.act(support), adj
        return output

    def extra_repr(self):
        return 'input_dim={}, output_dim={}'.format(
                self.in_features, self.out_features
        )


class Linear(Module):
    """
    Simple Linear layer with dropout.
    """

    def __init__(self, in_features, out_features, dropout, act, use_bias):
        super(Linear, self).__init__()
        self.dropout = dropout
        self.linear = nn.Linear(in_features, out_features, use_bias)
        self.act = act

    def forward(self, x):
        hidden = self.linear.forward(x)
        hidden = F.dropout(hidden, self.dropout, training=self.training)
        out = self.act(hidden)
        return out


class FermiDiracDecoder(Module):
    """Fermi Dirac to compute edge probabilities based on distances."""

    def __init__(self, r, t):
        super(FermiDiracDecoder, self).__init__()
        self.r = r
        self.t = t

    def forward(self, dist):
        probs = 1. / (torch.exp((dist - self.r) / self.t) + 1.0)
        return probs


class SAGELayer(nn.Module):
    """
    Encodes a node's using torch_geomtrics GraphSage
    """

    def __init__(self, in_features, out_features, dropout, act, bias, last=False):
        super(SAGELayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.act = act
        self.dropout = dropout
        self.last = last
        self.conv = SAGEConv(in_features, out_features, bias=bias)
        torch.nn.init.xavier_uniform(self.conv.lin_l.weight)
        torch.nn.init.xavier_uniform(self.conv.lin_r.weight)

    def forward(self, input):
        x, adj = input
        if not hasattr(self, 'edge_index'):
            self.edge_index = torch.Tensor(np.stack(np.where(adj.detach().cpu().to_dense()))).to(dtype=torch.long, device=x.get_device())
        x = self.conv(x, self.edge_index)
        if not self.last:
            output = self.act(x), adj
        else:
            output = x, adj
        return output

    def extra_repr(self):
        return 'input_dim={}, output_dim={}'.format(
                self.in_features, self.out_features
        )

