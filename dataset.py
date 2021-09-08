import torch
from torch.utils.data import Dataset

import nibabel as nib

import numpy as np

import os
import logging

def load_subject(file):
    data = nib.load(file)
    data_numpy = data.get_fdata()
    data_torch = torch.from_numpy(data_numpy).float()
    return data_torch

class DiffusionDataset(Dataset):
    """
    Load our processed HCP dataset.
    Samples are loaded onto a "common" grid so that all output tensors are of the same size and with aligned metadata.
    """
    def __init__(self, base_dir, tissue_name='wm_prealigned.nii', affine_name=None):
        self.subject_ids = [os.path.join(base_dir, subject) for subject in os.listdir(base_dir)]
        self.tissue_name = tissue_name
        self.affine_name = affine_name
        logging.info("HCPDataset: Found {} subjects in base directory {}.".format(len(self.subject_ids), base_dir))

    def __len__(self):
        return len(self.subject_ids)

    def __getitem__(self, index):
        subject_id = self.subject_ids[index]
        subject_file = os.path.join(subject_id, self.tissue_name)
        if self.affine_name is not None:
            affine_file = np.loadtxt(os.path.join(subject_id, self.affine_name))
            return load_subject(subject_file), affine_file

        else:
            return load_subject(subject_file)

def pairs_dataset(dataset_class, *args, **kwargs):
    """
    Constructs a dataset of pairs. Used for subject to subject registration.
    """
    class PairsDataset(Dataset):
        def __init__(self):
            self.d1 = dataset_class(*args, **kwargs)
            self.d2 = dataset_class(*args, **kwargs)

            self.pair_indices = [(ind1, ind2) for ind1 in range(len(self.d1)) for ind2 in range(len(self.d2))]
            logging.info("PairsDataset: Initialized with {} pair samples.".format(len(self.pair_indices)))

        def __len__(self):
            return len(self.pair_indices)

        def __getitem__(self, index):
            ind1, ind2 = self.pair_indices[index]
            sample = torch.stack((self.d1[ind1], self.d2[ind2]), dim=0)
            return sample
    return PairsDataset()

def atlas_dataset(dataset_class, atlas_fname, affine, *args, **kwargs):
    class AtlasDataset(Dataset):
        def __init__(self):
            self.d1 = dataset_class(*args, **kwargs)
            self.atlas = load_subject(atlas_fname)

        def __len__(self):
            return len(self.d1)
        
        def __getitem__(self, index):
            if affine:
                affine_gt = self.d1[index][1]
                sample = torch.stack((self.atlas, self.d1[index][0]), dim=0)
                return sample, affine_gt

            else:
                sample = torch.stack((self.atlas, self.d1[index]), dim=0)
                return sample

    return AtlasDataset()

class SingleSampleDataset(Dataset):
    """
    Trivial single sample dataset for testing overfitting of model.
    """
    def __init__(self, moving_fname, fixed_fname, copies=1):
        moving = load_subject(moving_fname)
        fixed = load_subject(fixed_fname)
        self.stacked = torch.stack((moving, fixed), dim=0)
        self.copies = copies
        
    def __len__(self):
        return self.copies

    def __getitem__(self, index):
        return self.stacked
