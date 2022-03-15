from dataset import DiffusionDataset, SingleSampleDataset, atlas_dataset, load_subject
from model_affine import MVCORegAffineNet, VMRegAffineNet
from model import MVCRegNet
from tqdm import tqdm

import nibabel as nib

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--initial', type=str)
args = parser.parse_args()

dataset = atlas_dataset(DiffusionDataset, '/blue/vemuri/josebouza/data/deformable_preaffine/fixed/298455/fodf_downsamples.nii',\
                        base_dir='/blue/vemuri/josebouza/data/deformable_preaffine/moving/',
                        tissue_name='fodf_downsamples.nii')

training_data, test_data = random_split(dataset, [90,10])

training_dataloader = torch.utils.data.DataLoader(training_data, batch_size=1, num_workers=2)
testing_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, num_workers=2)

header_img = nib.load('/blue/vemuri/josebouza/data/deformable_preaffine/fixed/298455/fodf_downsamples.nii')

device = 'cuda:0'
model = MVCRegNet().to(device)

if args.initial:
    model.load_state_dict(torch.load(args.initial))

optimizer = optim.Adam(model.parameters(), lr=0.0008)
loss_fn = nn.MSELoss()

base_save_dir = '/blue/vemuri/josebouza/projects/fodf_registration/deformable_mvc_layer3/'
epochs = [int(''.join(filter(str.isdigit, epoch_name))) for epoch_name in os.listdir(base_save_dir)]
current_epoch = max(epochs)+1 if len(epochs) > 0 else 0

for i in range(current_epoch, 10000):
    if True:
        print("Running evaluation at epoch {}".format(i))
        epoch_dir = os.path.join(base_save_dir, 'epoch{}/'.format(i))
        losses_fodf = []
        for id, sample in enumerate(testing_dataloader):
            with torch.no_grad():
                moving_fodf = sample[0].to(device).unsqueeze(1)
                fixed_fodf = sample[1].to(device).unsqueeze(1)

                combined_fodf = torch.cat([moving_fodf, fixed_fodf], axis=1)

                warped, affine = model(combined_fodf)
                loss_fodf = loss_fn(warped, fixed_fodf)

                losses_fodf.append(loss_fodf.item())
                print("fODF Loss: ", loss_fodf)

                iteration_folder = os.path.join(epoch_dir, 'iteration_{}/'.format(id))
                os.makedirs(iteration_folder, exist_ok=True)
                nib.save(nib.Nifti1Image(warped[0][0].cpu().numpy(), header_img.affine), os.path.join(iteration_folder, 'warped.nii.gz'))
                nib.save(nib.Nifti1Image(fixed_fodf[0].cpu().numpy(), header_img.affine), os.path.join(iteration_folder, 'fixed.nii.gz'))

        print("Mean fODF Loss: {}".format(sum(losses_fodf)/len(losses_fodf)))
    iterator = tqdm(enumerate(training_dataloader), total=len(training_data))
    loss_training = []
    for id, sample in iterator:
        moving = sample[0]
        fixed = sample[1]

        moving_fodf = moving.to(device).unsqueeze(1)
        fixed_fodf = fixed.to(device).unsqueeze(1)

        combined_fodf = torch.cat([moving_fodf, fixed_fodf], axis=1)

        optimizer.zero_grad()

        warped, affine = model(combined_fodf)
        loss_fodf = loss_fn(warped, fixed_fodf)

        loss = loss_fodf
        loss_training.append(loss.item())
        iterator.set_description("Training loss: {}".format(loss.item()))

        loss.backward()
        optimizer.step()
    print("Mean Training Loss: {}".format(sum(loss_training)/len(loss_training)))
    torch.save(model.state_dict(), os.path.join(epoch_dir, "model.pt"))
