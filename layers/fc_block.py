import torch
import torch.nn as nn

class LinearBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.conv = nn.Linear(in_channels, out_channels, **kwargs)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm1d(num_features=out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.bn(x)
        return x
