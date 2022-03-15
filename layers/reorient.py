import torch
import torch.nn as nn

from e3nn.o3.cartesian_spherical_harmonics import spherical_harmonics

import numpy as np
from dipy.core.sphere import disperse_charges, HemiSphere

import sys
import os
import logging

def cart2sphere(cart):
    x = cart[:,0]
    y = cart[:,1]
    z = cart[:,2]
    r = torch.sqrt(x * x + y * y + z * z)
    theta = torch.arccos(z/r)
    phi = torch.atan2(y, x)
    return torch.stack([theta,phi,r]).T

def generate_PSF_vectors(n_pts, device):
    theta = np.pi * np.random.rand(n_pts)
    phi = 2 * np.pi * np.random.rand(n_pts)
    hsph_initial = HemiSphere(theta=theta, phi=phi)
    hsph_updated, potential = disperse_charges(hsph_initial, 1000)
    PSF_vectors = hsph_updated.vertices
    
    return torch.from_numpy(PSF_vectors).to(device)

def spherical_harmonics_tournier(xyz, order):
    yzx = xyz[..., [1,2,0]]
    sh_raw = spherical_harmonics(order, yzx, True)
    # normalize correctly :
    # even entries: 1
    # odd entries: -1
    # 0: 1

    degrees = sh_raw.shape[-1]
    if degrees not in spherical_harmonics_tournier.normalization_constant_cache or \
       spherical_harmonics_tournier.normalization_constant_cache[degrees].device != xyz.device:

        normalization_constant = []
        for i in range(-degrees//2+1, degrees//2+1):
            if i == 0:
                normalization_constant.append(1)
            elif i % 2 == 0:
                normalization_constant.append(1)
            elif i % 2 == 1:
                normalization_constant.append(-1)

        norm_constant_torch = torch.tensor(normalization_constant).to(xyz.device).float()
        spherical_harmonics_tournier.normalization_constant_cache[degrees] = norm_constant_torch

    norm = spherical_harmonics_tournier.normalization_constant_cache[degrees]
    sh_normalized = sh_raw*norm
    return sh_normalized

spherical_harmonics_tournier.normalization_constant_cache = {}


def PSF2M(xyz, lmax):
    if len(xyz.shape) == 3:
        return torch.cat([spherical_harmonics_tournier(xyz, i).permute(0,2,1) for i in range(0, lmax+1, 2)], dim=1)
    elif len(xyz.shape) == 6:
        #return spherical_harmonics_tournier(xyz, range(0, lmax+1, 2)).permute(0,5,1,2,3,4)
        return torch.cat([spherical_harmonics_tournier(xyz, i).permute(0,5,1,2,3,4) for i in range(0, lmax+1, 2)], dim=1)

class FODFReorientation(nn.Module):
    def __init__(self, num_vectors=64, lmax=8, override_cache=False):
        super().__init__()
        self.lmax = lmax

        # psf vector caching since this usually takes some time to generate
        cache_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".psf_vectors/")
        os.makedirs(cache_path, exist_ok=True)
        vector_path = os.path.join(cache_path, str(num_vectors)+'.npy')
        if os.path.exists(vector_path) and not override_cache:
            logging.info("FODFReorientation: Loaded cached PSF vectors from {}".format(vector_path))
            _psf_vectors = torch.from_numpy(np.load(vector_path)).float()
        else:
            logging.info("FODFReorientation: No cached vectors found. Generating into {}".format(vector_path))
            _psf_vectors = generate_PSF_vectors(num_vectors, 'cpu').float()
            np.save(vector_path, _psf_vectors.numpy())
        self.register_buffer('psf_vectors', _psf_vectors)

        _M_p = torch.pinverse(PSF2M(self.psf_vectors[None], self.lmax)).float()
        self.register_buffer('M_p', _M_p)
        
    def forward(self, image, affine, modulate=True):
        # image: [B, W, H, D, N]
        # affine: [B, W, H, D, 3,3]
        
        image_s = image.shape
        image = image.view(image_s[0], -1, image_s[-1]).permute(0,2,1)
        affine_inv = torch.inverse(affine)
        w = torch.matmul(self.M_p, image)
        vi = self.psf_vectors.permute(1,0)

        vi_hat = torch.matmul(affine_inv, vi)
        if len(vi_hat.shape) == 6:
            vi_hat = vi_hat.permute(0,1,2,3,5,4)
        elif len(vi_hat.shape) == 3:
            vi_hat = vi_hat.permute(0,2,1)
        M_prime = PSF2M(vi_hat, self.lmax)
        # TEMP
        if (M_prime != M_prime).any():
            logging.warning("M_PRIME")
            logging.warning(affine_inv)
            logging.warning(affine)

        if modulate:
            modulation_factors = torch.norm(torch.matmul(affine_inv, vi), dim=-2)/torch.det(affine_inv)[...,None]
            if len(modulation_factors.shape) == 5:
                modulation_factors = modulation_factors.permute(0, 4, 1, 2, 3)
                modulation_factors = modulation_factors.view(modulation_factors.shape[0], modulation_factors.shape[1], -1)
            else:
                modulation_factors = modulation_factors[...,None]
            w_modulated = modulation_factors*w
        else:
            w_modulated = w

        w_modulated = w_modulated[...,None].permute(0,2,1,3)
        if len(M_prime.shape) == 3:
            M_prime = M_prime[:, None]
        else:
            M_prime = M_prime.view(M_prime.shape[0], M_prime.shape[1], -1, M_prime.shape[-1])
            M_prime = M_prime.permute(0, 2, 1, 3)

        reoriented = torch.matmul(M_prime, w_modulated)
        reoriented = reoriented.view(*image_s)
        return reoriented


def sampling_grid(size):
    vectors = [torch.arange(0, s) for s in size]
    grids = torch.meshgrid(vectors)
    grid = torch.stack(grids)
    grid = torch.unsqueeze(grid, 0)
    grid = grid.type(torch.FloatTensor)
    return grid

class DeformationJacobian(nn.Module):
    def __init__(self, size):
        super().__init__()
        grid = sampling_grid(size)
        self.register_buffer('grid', grid)
    
    def forward(self, d):
        deformation = (self.grid+d).view(-1, d.shape[2], d.shape[3], d.shape[4]) # work-around for gradient bug: https://github.com/pytorch/pytorch/issues/67919
        (dx, dy, dz) = torch.gradient(deformation, dim=(1,2,3))
        dx = dx[None] # work-around for gradient bug: https://github.com/pytorch/pytorch/issues/67919
        dy = dy[None] # work-around for gradient bug: https://github.com/pytorch/pytorch/issues/67919
        dz = dz[None] # work-around for gradient bug: https://github.com/pytorch/pytorch/issues/67919
        J = torch.stack([dx.permute(0,2,3,4,1), dy.permute(0,2,3,4,1), dz.permute(0,2,3,4,1)], dim=-2)
        return J