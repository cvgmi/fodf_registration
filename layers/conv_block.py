import torch
import torch.nn as nn

class ConvBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, **kwargs)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        return self.bn(self.relu(self.conv(x)))
