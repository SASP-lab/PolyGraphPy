import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GraphUNet
from torch_geometric.nn import global_mean_pool

class GraphUNetModel(torch.nn.Module):
    def __init__(self, input_dim: int, conv_hidden_channels: int, mlp_hidden_channels: int, depth: int = 5, pool_ratios: float = 0.5) -> None:
        super(GraphUNetModel, self).__init__()
        self.input_dim = input_dim
        self.conv_hidden_channels = conv_hidden_channels
        self.mlp_hidden_channels = mlp_hidden_channels
        self.unet = GraphUNet(input_dim, conv_hidden_channels, conv_hidden_channels, depth=depth, pool_ratios=pool_ratios)
        self.lin1 = Linear(conv_hidden_channels, mlp_hidden_channels)
        self.lin2 = Linear(mlp_hidden_channels, mlp_hidden_channels)
        self.lin3 = Linear(mlp_hidden_channels, mlp_hidden_channels)
        self.output = Linear(mlp_hidden_channels, 1)

    def forward(self, x, edge_index, edge_weight, batch, chain_size=None):
        h = self.unet(x, edge_index, batch=batch)

        h = global_mean_pool(h, batch)
        
        h = self.lin1(h)
        h = F.dropout(h)
        h = h.tanh()
        h = self.lin2(h)
        h = F.dropout(h)
        h = h.tanh()
        h = self.lin3(h)
        h = h.tanh()

        h = torch.abs(self.output(h))

        return h