import torch
from torch.utils.data import Dataset
from functools import lru_cache

import nibabel as nib

import numpy as np

import os
import logging

def load_subject(file):
    data = nib.load(file)
    data_numpy = data.get_fdata()
    data_numpy[data_numpy != data_numpy] = 0
    data_torch = torch.from_numpy(data_numpy).float()
    return data_torch

class DiffusionDataset(Dataset):
    """
    Load our processed HCP dataset.
    Samples are loaded onto a "common" grid so that all output tensors are of the same size and with aligned metadata.
    """
    def __init__(self, base_dir, tissue_name='fodf.nii', fa_name=None, affine=False):
        files = os.listdir(base_dir)
        files.sort()
        self.subject_ids = [os.path.join(base_dir, subject) for subject in files][:400]
        self.tissue_name = tissue_name
        self.fa_name = fa_name
        self.affine = affine
        logging.info("HCPDataset: Found {} subjects in base directory {}.".format(len(self.subject_ids), base_dir))

    def __len__(self):
        return len(self.subject_ids)

    def __getitem__(self, index):
        subject_id = self.subject_ids[index]
        subject_file = os.path.join(subject_id, self.tissue_name)
        if self.fa_name is not None:
            fa_file = os.path.join(subject_id, self.fa_name)
            if self.affine:
                affine_mtrx = np.loadtxt(os.path.join(subject_id, "affine.txt"))
                return load_subject(subject_file), load_subject(fa_file), affine_mtrx
            return load_subject(subject_file), load_subject(fa_file)

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
            return self.d1[ind1], self.d2[ind2]

    return PairsDataset()

def atlas_dataset(dataset_class, fodf_atlas, *args, **kwargs):
    class AtlasDataset(Dataset):
        def __init__(self):
            self.d1 = dataset_class(*args, **kwargs)
            self.fodf_atlas = load_subject(fodf_atlas)

        def __len__(self):
            return len(self.d1)
        
        def __getitem__(self, index):
                return (self.d1[index], self.fodf_atlas)
    return AtlasDataset()

def atlas_dataset_with_fa(dataset_class, fodf_atlas, fa_atlas, *args, **kwargs):
    class AtlasDataset(Dataset):
        def __init__(self):
            self.d1 = dataset_class(*args, **kwargs)
            self.fodf_atlas = load_subject(fodf_atlas)
            self.fa_atlas = load_subject(fa_atlas)
            self.cache = {}

        def __len__(self):
            return len(self.d1)
        
        def __getitem__(self, index):
                return (self.d1[index], (self.fodf_atlas, self.fa_atlas))

    return AtlasDataset()

class SingleSampleDataset(Dataset):
    """
    Trivial single sample dataset for testing overfitting of model.
    """
    def __init__(self, moving_fodf, moving_fa, fixed_fodf, fixed_fa, affine, copies=1):
        self.affine = np.loadtxt(affine)
        self.moving = (load_subject(moving_fodf), load_subject(moving_fa), self.affine)
        self.fixed = (load_subject(fixed_fodf), load_subject(fixed_fa))
        self.copies = copies
        
    def __len__(self):
        return self.copies

    def __getitem__(self, index):
        return self.moving, self.fixed