import torch
from torch import nn
from torch_geometric.nn import TransformerConv
from torch_geometric.data import Data

class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(d_model))
        self.b_2 = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = torch.sqrt(x.var(dim=-1, keepdim=True, unbiased=False) + self.eps)
        x = self.a_2*(x-mean)
        x /= std
        x += self.b_2
        return x
        #return self.a_2 * (x - mean) / std + self.b_2

class UniMPBlock(nn.Module):
    '''https://arxiv.org/pdf/2009.03509.pdf'''
    """Graph Transformer Network"""

    def __init__(self, node_dim=300, edge_dim=4, heads=4, dropout=0.0):
        super(UniMPBlock, self).__init__()

        self.TConv = TransformerConv(node_dim, node_dim, heads, dropout=dropout, edge_dim=edge_dim)
        self.LNorm = LayerNorm(node_dim * heads)
        self.Linear = nn.Linear(node_dim * heads, node_dim)
        self.Activ = nn.ELU(inplace=True)

    # @torch.cuda.amp.autocast(enabled=True)
    def forward(self, xin, e_idx, e_attr):
        x = self.TConv(xin, e_idx, e_attr)
        x = self.LNorm(x)
        x = self.Linear(x)
        out = self.Activ(x + xin)
        return Data(x=out, edge_index=e_idx, edge_attr=e_attr)