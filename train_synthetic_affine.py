from trainer import *

from dataset import DiffusionDataset, atlas_dataset
from model import BasicAffineUNetLog, BasicAffineUNetGeodesic, BasicAffineUNetNoMVC

import nibabel as nib

from datetime import datetime
import logging
import sys

project_name = datetime.now().strftime("reorient_%m-%d-%Y_%H-%M-%S")
#logging.basicConfig(filename='{}.log'.format(project_name), level=logging.DEBUG)
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


if __name__ == "__main__":
    dataset = atlas_dataset(DiffusionDataset, \
                            '/blue/vemuri/josebouza/data/HCP/data_affinepre_torch/moving/patch.nii',\
                            base_dir='/blue/vemuri/josebouza/data/HCP/data_affinepre_torch/fixed',\
                            tissue_name='patch.nii')

    training_data, test_data = random_split(dataset, [294, 6])

    model_instance = BasicAffineUNetLog()
    logging.info(model_instance)
    trainer = Trainer(model_instance, 
                      nn.MSELoss(),
                      training_data,
                      test_data,
                      0.0005,
                      6,
                      'cuda:0')

    for epoch in range(5000):
        logging.info("Starting epoch {}".format(epoch))
        mean_training_loss = trainer.train(print_status=True)
        mean_test_loss = trainer.test(print_status=True)
        logging.info("Epoch {} done. Mean testing loss: {} | Mean training loss: {}".format(epoch, mean_test_loss, mean_training_loss))

        # save fixed and warped test images 
        if epoch != 0 and epoch % 20 == 0:
            header_img = nib.load('/blue/vemuri/josebouza/data/HCP/data_affinepre_multiple/moving/patch.nii')
            epoch_save_name = project_name+"/epoch_{}/".format(epoch)
            trainer.save_samples("/blue/vemuri/josebouza/projects/fodf_registration/affinepre_multiple/{}".
                                 format(epoch_save_name),\
                                 header_img)


