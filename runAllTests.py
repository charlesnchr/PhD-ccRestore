import math
import os

import torch
import time 

import torch.optim as optim
import torchvision
from torch.autograd import Variable

from models import *
from datahandler import *

from plotting import testAndMakeCombinedPlots

from tensorboardX import SummaryWriter

import cv2

from options import parser

opt = parser.parse_args()

if opt.norm == '':
    opt.norm = opt.dataset
elif opt.norm.lower() == 'none':
    opt.norm = None

if len(opt.basedir) > 0:
    opt.root = opt.root.replace('basedir',opt.basedir)
    opt.weights = opt.weights.replace('basedir',opt.basedir)
    opt.out = opt.out.replace('basedir',opt.basedir)

if opt.task != 'sr':
    opt.scale = 1

if 'unet' in opt.model:
    opt.scale = 1

if __name__ == '__main__':

    os.makedirs(opt.out,exist_ok=True)

    opt.fid = open(opt.out + '/log.txt','w')
    print(opt)
    print(opt,'\n',file=opt.fid)
    
    dataloader, validloader = GetDataloaders(opt)
    net = GetModel(opt)

    if len(opt.weights) > 0: # load previous weights?
        checkpoint = torch.load(opt.weights)
        print('loading checkpoint',opt.weights)
        net.load_state_dict(checkpoint['state_dict'])
    testAndMakeCombinedPlots(net,validloader,opt)
