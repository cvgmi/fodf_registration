import torch
import torch.nn as nn

from .hilbert_sphere import SphereLog, tangentCombination, GeodesicDistance

class SHNorm(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, image):
        return torch.norm(image, dim=-1)

class TangentMap(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, image):
        # image: [B, C, rows, cols, depth, N]
        image_s = image.shape
        image_flat = image.view(image_s[0], -1, image_s[-1])
        B = image_flat[:,0]*0.0
        B[:,0] = 1
        image_log = SphereLog(image_flat, B)
        image_log = image_log.view(image_s)
        image_log.permute(0,1,5,2,3,4)
        image_log = image_log.view(image_s[0], -1, image_s[2], image_s[3], image_s[4])
        return image_log

class GeodesicDistanceMap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, image):
        image_s = image.shape
        image_flat = image.view(image_s[0], 1, 1, 1, -1, image_s[-1])
        weights = torch.ones([1, image_flat.shape[4]]).to(image_flat.device)
        #mean = tangentCombination(image_flat, weights)
        #mean = mean.view(image_s[0], 1, image_s[-1]).repeat(1, image_flat.shape[4], 1)
        image_flat = image_flat.view(image_s[0], image_flat.shape[4], image_s[-1])
        mean = torch.zeros_like(image_flat)
        mean[:,:,1] = 1
        distances = GeodesicDistance(image_flat, mean)
        distances = distances.view(image_s[0], image_s[1], image_s[2], image_s[3], image_s[4])
        return distances
