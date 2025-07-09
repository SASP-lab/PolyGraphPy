import torch
import torch.nn.functional as F

from torch.nn import Linear
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool

class GCN(torch.nn.Module):
    def __init__(self, input_dim: int, conv_hidden_channels: int, mlp_hidden_channels: int) -> None:
        super(GCN, self).__init__()

        self.input_dim = input_dim
        self.conv_hidden_channels = conv_hidden_channels
        self.mlp_hidden_channels = mlp_hidden_channels

        self.conv1 = GCNConv(input_dim, conv_hidden_channels, normalize=True)
        self.conv2 = GCNConv(conv_hidden_channels, conv_hidden_channels, normalize=True)
        self.conv3 = GCNConv(conv_hidden_channels, conv_hidden_channels, normalize=True)

        self.lin1 = Linear(conv_hidden_channels+1, mlp_hidden_channels) #adding the chain size to the embbeding

        self.output = Linear(mlp_hidden_channels, 1)
    
    def forward(self, x, edge_index, edge_weight, batch, chain_size):
        h = self.conv1(x, edge_index=edge_index, edge_weight=edge_weight)
        h = h.tanh()
        h = self.conv2(h, edge_index=edge_index, edge_weight=edge_weight)
        h = h.tanh()
        h = self.conv3(h, edge_index=edge_index, edge_weight=edge_weight)
        h = h.tanh()

        h = global_mean_pool(h, batch)

        chain_size = chain_size.unsqueeze(1)    # shape: [2, 1]
        h = torch.cat([h, chain_size], dim=1)  # shape: [2, 121]

        h = self.lin1(h)
        h = h.tanh()

        h = self.output(h)

        return h