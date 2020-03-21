from datahandler import *
from options import opt

dataloader = load_EstimateNL_dataset(opt.root,'train',opt)
validloader = load_EstimateNL_dataset(opt.root,'valid',opt)                

for i,bat in enumerate(dataloader):
    lq, std = bat[0], bat[1]
    print('%d:  %0.1f  %0.1f' % (i,torch.std(lq)*255,std*255))
    if i == 5:
        break