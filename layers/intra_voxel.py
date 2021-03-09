import torch
import torch.nn as nn
import torch.nn.functional as F

import layers.hilbert_sphere as sphere_ops

class IntraVoxelConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, zero_init=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

        if zero_init:
            self.weight_matrix = torch.nn.Parameter(
                torch.zeros(
                    out_channels,
                    (kernel_size**3) *
                    in_channels),
                requires_grad=True)
        else:
            self.weight_matrix = torch.nn.Parameter(
                torch.rand(
                    out_channels,
                    (kernel_size**3) *
                    in_channels),
                requires_grad=True)
        
    # x: [batches, channels, rows, cols, depth, N]
    def forward(self, x):
        # x: [batches, channels, rows, cols, depth, N] ->
        #    [batches, channels, N, rows, cols, depth]
        x = x.permute(0, 1, 5, 2, 3, 4).contiguous()

        # x_windows: [batches, channels, N, rows_reduced, cols_reduced, depth_reduced, window_x, window_y]
        x_windows = x.unfold(3, self.kernel_size, self.stride).contiguous()
        x_windows = x_windows.unfold(4, self.kernel_size, self.stride).contiguous()
        x_windows = x_windows.unfold(5, self.kernel_size, self.stride).contiguous()

        x_s = x_windows.shape
        #x_windows: [batches, channels, N  rows_reduced, cols_reduced, depth_reduced, window]
        x_windows = x_windows.view(x_s[0], x_s[1], x_s[2], x_s[3], x_s[4], x_s[5], -1)

        #x_windows: [batches, rows_reduced, cols_reduced, depth_reduced, window, channels, N]
        x_windows = x_windows.permute(0, 3, 4, 5, 6, 1, 2).contiguous()

        x_s = x_windows.shape
        # x_windows: [batches, rows_reduced, cols_reduced, depth_reduced, window*channels, N]
        x_windows = x_windows.view(x_s[0], x_s[1], x_s[2], x_s[3], -1, x_s[6]).contiguous()

        return sphere_ops.tangentCombination(x_windows, self.weight_matrix)

def interpolator(x, target_size):
        x_s = x.shape
        #x : [batches, EAP, channels, rows, cols, depth]
        x = x.permute(0,5,1,2,3,4).contiguous()
        #x : [batches, EAP*channels, rows, cols, depth]
        x = x.view(x_s[0], -1, x_s[2], x_s[3], x_s[4])
        #x : [batches, EAP*channels, rows_up, cols_up, depth_up]
        x = F.interpolate(x, size=target_size, mode='trilinear', align_corners=True)
        #x : [batches, EAP,channels, rows_up, cols_up, depth_up]
        x = x.view(x_s[0], x_s[-1], x_s[1], target_size, target_size, target_size)
        #x : [batches, channels, rows_up, cols_up, depth_up EAP]
        x = x.permute(0,2,3,4,5,1).contiguous()

        return x

class IntraConvUpsample(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, zero_init=False, target_size=None):
        super().__init__()
        self.conv = IntraVoxelConv(in_channels, out_channels, kernel_size, stride, zero_init)
        self.target_size = target_size

    def forward(self, x):
        x = self.conv(x)
        if self.target_size is not None:
            x = interpolator(x, self.target_size)
        return x
