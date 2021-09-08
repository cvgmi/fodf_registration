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

def resample(affine, moving_image):
    affine_fixed = fix(affine)
    target_size = (moving_image.shape[0], moving_image.shape[-1], *moving_image.shape[1:-1])
    resample_grid  = F.affine_grid(affine_fixed[:,:3,:], target_size, align_corners=True)
    moving_image = moving_image.permute(0,4,1,2,3)
    warped = F.grid_sample(moving_image, resample_grid, mode='bilinear', align_corners=True)
    warped = warped.permute(0,2,3,4,1)

    return warped
