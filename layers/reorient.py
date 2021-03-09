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
    sh_raw = spherical_harmonics(order, yzx, False)

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
    sh_normalized = sh_raw*norm[None,None]
    return sh_normalized

spherical_harmonics_tournier.normalization_constant_cache = {}


def PSF2M(xyz, lmax):
    return torch.cat([spherical_harmonics_tournier(xyz, i).permute(0,2,1) for i in range(0, lmax+1, 2)], dim=1)

class FODFReorientation(nn.Module):
    def __init__(self, num_vectors=1024, lmax=8, override_cache=False):
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
        
    def forward(self, image, affine):
        # image: [B,W, H, D, N]
        # affine: [B,3,3]
        image_s = image.shape
        image = image.reshape(image_s[0], -1, image_s[-1]).permute(0,2,1)
        affine_no_scale = affine/(torch.det(affine)[:, None, None])
        affine_inv = torch.inverse(affine_no_scale)
        w = torch.matmul(self.M_p, image)
        vi = self.psf_vectors.permute(1,0)
        vi_hat = torch.matmul(affine_inv, vi)
        vi_hat = vi_hat.permute(0,2,1)
        M_prime = PSF2M(vi_hat, self.lmax)
        if (M_prime != M_prime).any():
            logging.warning("M_PRIME")
            logging.warning(affine_inv)
            logging.warning(affine)
        reoriented = torch.matmul(M_prime, w)
        reoriented = reoriented.permute(0,2,1).reshape(*image_s)
        return reoriented



