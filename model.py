import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.intra_voxel import IntraConvUpsample, IntraVolterraUpsample
from layers.sh2scalar import SHNorm, TangentMap, GeodesicDistanceMap
from layers.unet.unet import UNet
from layers.fc_block import LinearBlock
from layers.transformations import affine_exponential
from layers.reorient import DeformationJacobian, FODFReorientation
from layers.resample import SpatialTransformer

class fODFBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.intra_voxel = nn.Sequential(IntraConvUpsample(2, 4, kernel_size=3, target_size=(64, 80, 64), zero_init=True),
                                         IntraConvUpsample(4, 8, kernel_size=3, target_size=(64, 80, 64), zero_init=True),
                                         IntraConvUpsample(8, 8, kernel_size=3, target_size=(64, 80, 64), zero_init=True))

        self.scalar = GeodesicDistanceMap()
        self.conv_blocks = UNet(10, 3)

    def forward(self, x):
        y = self.intra_voxel(x)
        y = self.scalar(y)
        y2 = self.scalar(x)
        y = torch.cat([y, y2], dim=1)
        y = self.conv_blocks(y)
        return y

class FABackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.unet = UNet(1, 3)
    
    def forward(self, x):
        y = self.unet(x)
        return y

class FusionHead(nn.Module):
    def __init__(self, size=(32, 40, 32)):
        super().__init__()
        self.transformer = SpatialTransformer(size)
        self.jacobian = DeformationJacobian(size)
        self.reorient = FODFReorientation()
    
    def forward(self, fodf, d_fodf):
        # resample image (transformer layer)
        affine = self.jacobian(d_fodf)

        fodf_ = fodf.permute(0, 1, 5, 2, 3, 4)[:, 0]
        warped_fodf_ = self.transformer(fodf_, d_fodf)
        warped_fodf = warped_fodf_.permute(0, 2, 3, 4, 1)
        warped_fodf = self.reorient(warped_fodf, affine)

        return warped_fodf, d_fodf

class MVCRegNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fodf = fODFBackbone()
        self.fusion_head = FusionHead()
    
    def forward(self, fodf):
        d_fodf = self.fodf(fodf)
        warped_fodf, d = self.fusion_head(fodf, d_fodf)

        return warped_fodf, d
