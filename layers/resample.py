import torch
import torch.nn as nn
import torch.nn.functional as F

def fix(A):
    B = torch.zeros_like(A)
    
    B[:,0,0] = A[:,2,2]
    B[:,0,1] = A[:,2,1]
    B[:,0,2] = A[:,2,0]
    
    B[:,1,0] = A[:,1,2]
    B[:,1,1] = A[:,1,1]
    B[:,1,2] = A[:,1,0]
    
    B[:,2,0] = A[:,0,2]
    B[:,2,1] = A[:,0,1]
    B[:,2,2] = A[:,0,0]
    
    B[:,0,3] = A[:,2,3]
    B[:,1,3] = A[:,1,3]
    B[:,2,3] = A[:,0,3]
    
    return B

def resample(affine, fodf, fa):
    affine_fixed = fix(affine)
    target_size = fa.shape
    resample_grid  = F.affine_grid(affine_fixed[:,:3,:], target_size, align_corners=True)
    fodf = fodf.permute(0,4,1,2,3)
    warped_fodf = F.grid_sample(fodf, resample_grid, mode='bilinear', align_corners=True)
    warped_fodf = warped_fodf.permute(0,2,3,4,1)
    # This is wrong?
    warped_fa = F.grid_sample(fa, resample_grid, mode='bilinear', align_corners=True)
    return warped_fodf, warped_fa

def resample_fodf(affine, fodf):
    affine_fixed = fix(affine)
    target_size = (fodf.shape[0], fodf.shape[-1], fodf.shape[1], fodf.shape[2], fodf.shape[3])
    resample_grid  = F.affine_grid(affine_fixed[:,:3,:], target_size, align_corners=True)
    fodf = fodf.permute(0,4,1,2,3)
    warped_fodf = F.grid_sample(fodf, resample_grid, mode='bilinear', align_corners=True)
    warped_fodf = warped_fodf.permute(0,2,3,4,1)
    return warped_fodf

def resample_fa(affine, fa):
    affine_fixed = fix(affine)
    target_size = fa.shape
    resample_grid  = F.affine_grid(affine_fixed[:,:3,:], target_size, align_corners=False)
    warped_fa = F.grid_sample(fa, resample_grid, mode='bilinear', align_corners=False)
    return warped_fa

class SpatialTransformer(nn.Module):
    """
    N-D Spatial Transformer
    """

    def __init__(self, size, mode='bilinear'):
        super().__init__()

        self.mode = mode

        # create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)

        # registering the grid as a buffer cleanly moves it to the GPU, but it also
        # adds it to the state dict. this is annoying since everything in the state dict
        # is included when saving weights to disk, so the model files are way bigger
        # than they need to be. so far, there does not appear to be an elegant solution.
        # see: https://discuss.pytorch.org/t/how-to-register-buffer-without-polluting-state-dict
        self.register_buffer('grid', grid)

    def forward(self, src, flow):
        # new locations
        new_locs = self.grid + flow
        shape = flow.shape[2:]

        # need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        # move channels dim to last position
        # also not sure why, but the channels need to be reversed
        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        return torch.nn.functional.grid_sample(src, new_locs, align_corners=True, mode=self.mode)
