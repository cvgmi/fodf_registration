from trainer import *

from dataset import SingleSampleDataset
from model import BasicAffineUNetLog, BasicAffineUNetGeodesic, BasicAffineUNetNoMVC

import nibabel as nib

from datetime import datetime
import logging
import sys

project_name = datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
#logging.basicConfig(filename='{}.log'.format(project_name), level=logging.DEBUG)
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


if __name__ == "__main__":
    dataset = SingleSampleDataset('/blue/vemuri/josebouza/data/HCP/data_affinepre_torch/test/moving/patch.nii',\
                                  '/blue/vemuri/josebouza/data/HCP/data_affinepre_torch/test/fixed/patch2.nii',\
                                  copies=6)

    model_instance = BasicAffineUNetLog()
    logging.info(model_instance)
    trainer = Trainer(model_instance, 
                      nn.MSELoss(),
                      dataset,
                      dataset,
                      0.004,
                      6,
                      'cuda:0')

    for epoch in range(5000):
        logging.info("Starting epoch {}".format(epoch))
        mean_training_loss = trainer.train(print_status=False)
        #mean_test_loss = trainer.test(print_status=True)
        mean_test_loss = 0
        logging.info("Epoch {} done. Mean testing loss: {} | Mean training loss: {}".format(epoch, mean_test_loss, mean_training_loss))



