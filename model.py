import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.intra_voxel import IntraConvUpsample
from layers.sh2scalar import SHNorm, TangentMap, GeodesicDistanceMap
from layers.unet.unet import UNet
from layers.fc_block import LinearBlock
from layers.transformations import affine_exponential
from layers.reorient import FODFReorientation

class BasicAffineUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.intra_voxel = nn.Sequential(IntraConvUpsample(2, 8, kernel_size=3, stride=1, target_size=8, zero_init=True))

        self.scalar = SHNorm()

        self.conv_blocks = UNet(8, 1)
        
        self.fc = nn.Sequential(LinearBlock(512, 512), 
                                LinearBlock(512, 256))

        self.fc_last = nn.Linear(256, 16)
        self.fc_last.weight = nn.Parameter(torch.zeros_like(self.fc_last.weight))
        self.fc_last.bias = nn.Parameter(torch.zeros_like(self.fc_last.bias))

    def forward(self, x):
        moving_image = x[:, 0]
        y = self.intra_voxel(x)
        y = self.scalar(y)
        y = self.conv_blocks(y)
        y = y.view(y.shape[0], -1)
        y = self.fc(y)
        y = self.fc_last(y)
        y = y.view(y.shape[0], 4, 4)
        y = affine_exponential(y)
        #y = torch.tensor([[[1.0,0,0,0],[0,1,0,0],[0,0,1,0]]]).to(y.device).repeat(6,1,1)

        # resample image (transformer layer)
        target_size = (moving_image.shape[0], moving_image.shape[-1], *moving_image.shape[1:-1])
        resample_grid  = F.affine_grid(y[:,:3,:], target_size, align_corners=False)
        moving_image = moving_image.permute(0,4,1,2,3)
        warped = F.grid_sample(moving_image, resample_grid, mode='bilinear', align_corners=False)
        warped = warped.permute(0,2,3,4,1)
        return warped, y

class BasicAffineUNetLog(nn.Module):
    def __init__(self):
        super().__init__()
        self.intra_voxel = nn.Sequential(IntraConvUpsample(2, 8, kernel_size=3, stride=1, target_size=8, zero_init=True))

        self.scalar = TangentMap()

        self.conv_blocks = UNet(45*8, 1)
        
        self.fc = nn.Sequential(LinearBlock(512, 512), 
                                LinearBlock(512, 256),
                                LinearBlock(256, 128),
                                LinearBlock(128, 64))

        self.fc_last = nn.Linear(64, 16)
        self.fc_last.weight = nn.Parameter(torch.zeros_like(self.fc_last.weight))
        self.fc_last.bias = nn.Parameter(torch.zeros_like(self.fc_last.bias))
        self.reorient = FODFReorientation(num_vectors=1024)

    def forward(self, x):
        moving_image = x[:, 0]
        y = self.intra_voxel(x)
        y = self.scalar(y)
        y = self.conv_blocks(y)
        y = y.view(y.shape[0], -1)
        y = self.fc(y)
        y = self.fc_last(y)
        y = y.view(y.shape[0], 4, 4)
        y = affine_exponential(y)
        #y = torch.tensor([[[1.2, 0, 0, 0], [0,1,0,0], [0,0,1,0], [0,0,0,1]]]).to(x.device).float().repeat(6,1,1)

        # resample image (transformer layer)
        target_size = (moving_image.shape[0], moving_image.shape[-1], *moving_image.shape[1:-1])
        resample_grid  = F.affine_grid(y[:,:3,:], target_size, align_corners=False)
        moving_image = moving_image.permute(0,4,1,2,3)
        warped = F.grid_sample(moving_image, resample_grid, mode='bilinear', align_corners=False)
        warped = warped.permute(0,2,3,4,1)

        # reorient image
        warped = self.reorient(warped, y[:,:3,:3])
        return warped, y

class BasicAffineUNetGeodesic(nn.Module):
    def __init__(self):
        super().__init__()
        self.intra_voxel = nn.Sequential(IntraConvUpsample(2, 8, kernel_size=3, stride=1, target_size=8, zero_init=True))

        self.scalar = GeodesicDistanceMap()

        self.conv_blocks = UNet(8, 1)
        
        self.fc = nn.Sequential(LinearBlock(512, 512), 
                                LinearBlock(512, 256))

        self.fc_last = nn.Linear(256, 16)
        self.fc_last.weight = nn.Parameter(torch.zeros_like(self.fc_last.weight))
        self.fc_last.bias = nn.Parameter(torch.zeros_like(self.fc_last.bias))
        self.reorient = FODFReorientation(num_vectors=1024)

    def forward(self, x):
        moving_image = x[:, 0]
        y = self.intra_voxel(x)
        y = self.scalar(y)
        y = self.conv_blocks(y)
        y = y.view(y.shape[0], -1)
        y = self.fc(y)
        y = self.fc_last(y)
        y = y.view(y.shape[0], 4, 4)
        y = affine_exponential(y)

        # resample image (transformer layer)
        target_size = (moving_image.shape[0], moving_image.shape[-1], *moving_image.shape[1:-1])
        resample_grid  = F.affine_grid(y[:,:3,:], target_size, align_corners=False)
        moving_image = moving_image.permute(0,4,1,2,3)
        warped = F.grid_sample(moving_image, resample_grid, mode='bilinear', align_corners=False)
        warped = warped.permute(0,2,3,4,1)
        
        # reorient image
        warped = self.reorient(warped, y[:,:3,:3])
        return warped, y

class BasicAffineUNetNoMVC(nn.Module):
    def __init__(self):
        super().__init__()
        self.scalar = SHNorm()

        self.conv_blocks = UNet(2, 1)
        
        self.fc = nn.Sequential(LinearBlock(1331, 512), 
                                LinearBlock(512, 256))

        self.fc_last = nn.Linear(256, 16)
        self.fc_last.weight = nn.Parameter(torch.zeros_like(self.fc_last.weight))
        self.fc_last.bias = nn.Parameter(torch.zeros_like(self.fc_last.bias))

    def forward(self, x):
        moving_image = x[:, 0]
        y = self.scalar(x)
        y = self.conv_blocks(y)
        y = y.view(y.shape[0], -1)
        y = self.fc(y)
        y = self.fc_last(y)
        y = y.view(y.shape[0], 4, 4)
        y = affine_exponential(y)

        # resample image (transformer layer)
        target_size = (moving_image.shape[0], moving_image.shape[-1], *moving_image.shape[1:-1])
        resample_grid  = F.affine_grid(y[:,:3,:], target_size, align_corners=False)
        moving_image = moving_image.permute(0,4,1,2,3)
        warped = F.grid_sample(moving_image, resample_grid, mode='bilinear', align_corners=False)
        warped = warped.permute(0,2,3,4,1)
        return warped, y
