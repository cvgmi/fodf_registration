import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.intra_voxel import IntraConvUpsample
from layers.sh2scalar import SHNorm, TangentMap, GeodesicDistanceMap
from layers.unet.unet import UNet
from layers.fc_block import LinearBlock
from layers.transformations import affine_exponential
from layers.reorient import FODFReorientation
from layers.resample import resample

class BasicAffineUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.intra_voxel = nn.Sequential(IntraConvUpsample(2, 16, kernel_size=3, target_size=32),
                                         IntraConvUpsample(16, 8, kernel_size=3, target_size=16),
                                         IntraConvUpsample(8, 1, kernel_size=3, target_size=8))

        self.scalar = SHNorm()

        self.conv_blocks = UNet(64, 1)
        
        self.fc = nn.Sequential(LinearBlock(512, 128))

        self.fc_last = nn.Linear(128, 16)
        self.fc_last.weight = nn.Parameter(torch.zeros_like(self.fc_last.weight))
        self.fc_last.bias = nn.Parameter(torch.zeros_like(self.fc_last.bias))
        self.reorient = FODFReorientation(num_vectors=1024)

    def forward(self, x):
        moving_image = x[:, 0]
        y = self.intra_voxel(x)
        y = self.scalar(y)
        y = y.view(y.shape[0], -1)
        y = self.fc(y)
        y = self.fc_last(y)
        y = y.view(y.shape[0], 4, 4)
        y = affine_exponential(y)

        # resample image (transformer layer)
        warped = resample(y, moving_image)
        warped = self.reorient(warped, y[:,:3,:3])
        return warped, y