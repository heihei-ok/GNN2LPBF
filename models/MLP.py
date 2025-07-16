import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, ChebConv, BatchNorm  # noqa
import torch.nn.functional as F


class MLP(torch.nn.Module):
    def __init__(self, feature, out_channel):
        super(MLP, self).__init__()

        self.MLP1 = nn.Linear(feature, 1024)
        self.bn1 = nn.BatchNorm1d(1024)

        self.MLP2 = nn.Linear(1024, 1024)
        self.bn2 = nn.BatchNorm1d(1024)

        self.fc = nn.Sequential(nn.Linear(1024, 512), nn.ReLU(inplace=True))
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Sequential(nn.Linear(512, out_channel))

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr

        x = self.MLP1(x)
        x = self.bn1(x)
        x = nn.ReLU(inplace=True)(x)

        x = self.MLP2(x)
        x = self.bn2(x)
        x = nn.ReLU(inplace=True)(x)

        x = self.fc(x)
        x = self.dropout(x)
        x = self.fc1(x)

        return x
