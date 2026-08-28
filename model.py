import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCNEncoder(nn.Module):
    """Shared two-layer GCN with a residual connection."""

    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.5, self_loops=True):
        super().__init__()
        self.conv1 = GCNConv(
            in_dim,
            hidden_dim,
            add_self_loops=self_loops,
            normalize=True,
        )
        self.conv2 = GCNConv(
            hidden_dim,
            out_dim,
            add_self_loops=self_loops,
            normalize=True,
        )
        self.dropout = dropout
        self.res_proj = (
            nn.Identity()
            if hidden_dim == out_dim
            else nn.Linear(hidden_dim, out_dim, bias=False)
        )

    def forward(self, x, edge_index):
        h1 = F.relu(self.conv1(x, edge_index))
        h1 = F.dropout(
            h1,
            p=self.dropout,
            training=self.training,
        )
        h2 = self.conv2(h1, edge_index)
        return h2 + self.res_proj(h1)


class FusionNodeHead(nn.Module):
    """Classify genes from CPDB, mean scRNA, and raw feature representations."""

    def __init__(self, out_dim, in_dim, cls_hidden, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * out_dim + in_dim, cls_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(cls_hidden, 1),
        )

    def forward(self, h_cpdb, h_scrna_mean, x):
        fused = torch.cat([h_cpdb, h_scrna_mean, x], dim=1)
        return self.net(fused).squeeze(-1)


class EmbeddingFusionModel(nn.Module):
    def __init__(
        self,
        in_dim,
        hidden_dim,
        out_dim,
        node_cls_hidden,
        dropout,
        self_loops=True,
    ):
        super().__init__()
        self.encoder = GCNEncoder(
            in_dim,
            hidden_dim,
            out_dim,
            dropout,
            self_loops=self_loops,
        )
        self.node_head = FusionNodeHead(
            out_dim,
            in_dim,
            node_cls_hidden,
            dropout,
        )

    def encode(self, x, edge_index):
        return self.encoder(x, edge_index)

    def classify(self, h_cpdb, h_scrna_mean, x):
        return self.node_head(h_cpdb, h_scrna_mean, x)
