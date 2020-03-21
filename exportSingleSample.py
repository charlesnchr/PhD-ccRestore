import numpy as np
from datahandler import *

import argparse

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
toPIL = transforms.ToPILImage()

from options import opt

opt.workers = 0
opt.scale = 1
opt.batchSize = 20
opt.ntrain = 100

# dataloader = load_HDF5_dataset(root='C:/phd-data/datasets/camelyonpatch_level_2_split_valid_x.h5',category='valid',shuffle=True,batchSize=10,num_workers=0,imageSize=32)
# dataloader = load_randomised_dataset_2shade_onlyblur(category='valid',batchSize=10,num_workers=0,imageSize=32)
# dataloader = load_HDF5_dataset('C:/phd-data/datasets/camelyonpatch_level_2_split_valid_x.h5','train',opt)
# dataloader = load_deadleaves_dataset(root='C:/phd-data/set3_32x32',category='valid',batchSize=10,num_workers=0,imageSize=32)
# dataloader = load_SIM_dataset(root='C:/Users/cnc39_admin/Dropbox/0phd/SIMRec/TrainingData_nsamples100',category='train',batchSize=1,num_workers=0)
dataloader = load_GenericPickle_dataset('G:/Data/segmentation/partitioned_192','train',opt)

import matplotlib.pyplot as plt


for bat in dataloader:
    stim,refl = bat[0][0],bat[1][0]
    print(stim.dtype,refl.dtype)
    # stim = torch.mean(stim,0).unsqueeze(0)
    
    plt.subplot(1,2,1)
    plt.imshow(toPIL(stim),cmap='gray',vmin=0,vmax=1)
    print(stim.shape)
    plt.subplot(1,2,2)
    print(refl.shape)
    plt.imshow(refl,cmap='gray',vmin=0,vmax=1)
    plt.show()
    break