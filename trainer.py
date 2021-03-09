import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split

from dataset import DiffusionDataset, atlas_dataset
from model import BasicAffineUNetLog, BasicAffineUNetGeodesic, BasicAffineUNetNoMVC

import nibabel as nib

from tqdm import tqdm

from datetime import datetime
import logging
import os
import sys


class Trainer:
    def __init__(self, 
                 model,
                 loss,
                 training_dataset,
                 testing_dataset,
                 lr,
                 batch_size,
                 device):

        self.device = device
        self.model = model.to(self.device)
        self.loss = loss.to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.training_dataset = training_dataset
        self.testing_dataset = testing_dataset

        self.train_dataloader = torch.utils.data.DataLoader(self.training_dataset,
                                                            batch_size=batch_size,
                                                            shuffle=True,
                                                            num_workers=0)

        self.test_dataloader = torch.utils.data.DataLoader(self.testing_dataset,
                                                           batch_size=1,
                                                           shuffle=True,
                                                           num_workers=0)
    def get_model(self):
        return self.model

    def train(self, print_status=False):
        loss_history = []
        self.model.train()
        iterator = tqdm(self.train_dataloader) if print_status else self.train_dataloader
        for sample in iterator:
            self.optimizer.zero_grad()
            sample = sample.to(self.device)
            warped, affine = self.model(sample)
            loss = self.loss(warped, sample[:,1])

            if print_status:
                iterator.set_description("Training loss: {}".format(loss))

            loss.backward()
            self.optimizer.step()
            loss_history.append(loss.item())

        return sum(loss_history)/len(loss_history)

    def test(self, print_status=True):
        loss_history = []
        iterator = tqdm(self.test_dataloader) if print_status else self.test_dataloader
        with torch.no_grad():
            self.model.eval()
            for sample in iterator:
                sample = sample.to(self.device)
                warped, affine = self.model(sample)
                loss_value = self.loss(warped, sample[:, 1])
                #loss_value = self.loss(sample[:,0], sample[:,1])


                if print_status:
                    iterator.set_description("Testing loss: {}".format(loss_value))

                loss_history.append(loss_value.item())

        return sum(loss_history)/len(loss_history)
    
    def save_samples(self, base_dir, header_img):
        self.model.eval()
        with torch.no_grad():
            for id, sample in enumerate(self.test_dataloader):
                sample = sample.to(self.device)
                warped, affine = self.model(sample)
                warped = warped[0].cpu().detach().numpy()
                fixed = sample[0,1].cpu().detach().numpy()
                folder_name = os.path.join(base_dir, str(id))
                os.makedirs(folder_name, exist_ok=True)
                nib.save(nib.Nifti1Image(warped, header_img.affine), os.path.join(folder_name, 'warped.nii'))
                nib.save(nib.Nifti1Image(fixed, header_img.affine), os.path.join(folder_name, 'fixed.nii'))

