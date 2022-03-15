import torch
import torch.nn as nn
import torch.nn.functional as F

import layers.hilbert_sphere as sphere_ops

import MVC
import time

import MVC
import torch

class LogFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weights):
        output_log = MVC.log_forward(input, weights)
        variables = [input, weights]
        ctx.save_for_backward(*variables)
        return output_log

    @staticmethod
    def backward(ctx, grad_log):
        input, weights = ctx.saved_tensors
        grad_x = MVC.log_backward(input, weights, grad_log)
        grad_weights = MVC.log_weights_backward(input, weights, grad_log)
        return grad_x, grad_weights
MVLog = LogFunction.apply

class ExpFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, original_input):
        output_exp = MVC.exp_forward(input, original_input)
        variables = [input, original_input]
        ctx.save_for_backward(*variables)
        return output_exp

    @staticmethod
    def backward(ctx, grad_exp):
        input, original_input = ctx.saved_tensors
        grad_log, grad_M = MVC.exp_backward(input, original_input, grad_exp.contiguous())
        return grad_log, grad_M
MVExp = ExpFunction.apply

class MVC_function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weights):
        output_log = MVC.log_forward(input, weights)
        output_exp = MVC.exp_forward(output_log, input)
        variables = [input, output_log, weights]
        ctx.save_for_backward(*variables)
        return output_exp

    @staticmethod
    def backward(ctx, grad_exp):
        input, output_log, weights = ctx.saved_tensors
        grad_log, grad_M = MVC.exp_backward(output_log, input, grad_exp.contiguous())
        grad_x = MVC.log_backward(input, weights, grad_log)
        grad_weights = MVC.log_weights_backward(input, weights, grad_log)
        return grad_x+grad_M, grad_weights


MVC_apply = MVC_function.apply
class IntraVoxelConv(torch.nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size=3, zero_init=True):
        super().__init__()
        if zero_init:
            self.weight_mask = torch.nn.Parameter(torch.zeros([output_channels, kernel_size, kernel_size, kernel_size, input_channels], requires_grad=True))
        else:
            self.weight_mask = torch.nn.Parameter(torch.rand([output_channels, kernel_size, kernel_size, kernel_size, input_channels], requires_grad=True))
    
    def forward(self, x):
        return MVC_apply(x, self.weight_mask)

class IntraVoxelVolterra(torch.nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size=3, zero_init=False):
        super().__init__()
        if zero_init:
            self.weight_mask1 = torch.nn.Parameter(torch.zeros([output_channels, kernel_size, kernel_size, kernel_size, input_channels], requires_grad=True))
            self.weight_mask2 = torch.nn.Parameter(torch.zeros([output_channels, kernel_size, kernel_size, kernel_size, input_channels], requires_grad=True))
            self.weight_mask3 = torch.nn.Parameter(torch.zeros([output_channels, kernel_size, kernel_size, kernel_size, input_channels], requires_grad=True))
        else:
            self.weight_mask1 = torch.nn.Parameter(torch.rand([output_channels, kernel_size, kernel_size, kernel_size, input_channels], requires_grad=True))
            self.weight_mask2 = torch.nn.Parameter(torch.rand([output_channels, kernel_size, kernel_size, kernel_size, input_channels], requires_grad=True))
            self.weight_mask3 = torch.nn.Parameter(torch.rand([output_channels, kernel_size, kernel_size, kernel_size, input_channels], requires_grad=True))
    
    def forward(self, x):
        log1 = MVLog(x, self.weight_mask1)
        log2 = MVLog(x, self.weight_mask2)
        log3 = MVLog(x, self.weight_mask3)
        return MVExp(log1+log2*log3, x)

class IntraVoxelConvTorch(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, zero_init=False):
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
        if type(target_size) == int:
            x = x.view(x_s[0], x_s[-1], x_s[1], target_size, target_size, target_size)
        else:
            x = x.view(x_s[0], x_s[-1], x_s[1], *target_size)
        #x : [batches, channels, rows_up, cols_up, depth_up EAP]
        x = x.permute(0,2,3,4,5,1).contiguous()

        return x

class IntraConvUpsample(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, target_size=None, torch=False, zero_init=False):
        super().__init__()
        if torch:
            self.conv = IntraVoxelConvTorch(in_channels, out_channels, kernel_size, zero_init=zero_init)
        else:
            self.conv = IntraVoxelConv(in_channels, out_channels, kernel_size, zero_init=zero_init)
        self.target_size = target_size

    def forward(self, x):
        x = self.conv(x)
        if self.target_size is not None:
            x = interpolator(x, self.target_size)
        return x

class IntraVolterraUpsample(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, target_size=None, zero_init=False):
        super().__init__()
        self.conv = IntraVoxelVolterra(in_channels, out_channels, kernel_size, zero_init=zero_init)
        self.target_size = target_size

    def forward(self, x):
        x = self.conv(x)
        if self.target_size is not None:
            x = interpolator(x, self.target_size)
        return x
