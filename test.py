from options import parser
# import testfunctions
from models import GetModel, ESRGAN_Discriminator, ESRGAN_FeatureExtractor
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np

opt = parser.parse_args()

opt.model = 'rcan'
opt.out = '/Users/cc/Desktop/test'
opt.rootTesting = '/Volumes/GoogleDrive/My Drive/01datasets/Meng/filesForTesting_20201211'
opt.testFunction = 'experimentalER'
opt.testFunctionArgs = 'imageSize,1000'
opt.cpu = True
opt.scale = 1
opt.task = "segment"
opt.n_resblocks = 10
opt.n_resgroups = 2
opt.nch_in = 3
opt.nch_out = 3
opt.n_feats = 64
opt.reduction = 16
opt.weka_colours = True

model = GetModel(opt)
device = torch.device('cuda' if torch.cuda.is_available() and not opt.cpu else 'cpu')
checkpoint = torch.load("/Volumes/lag.radian.dk/meng_3colours_20201213/final.pth",map_location=device)
model.load_state_dict(checkpoint['state_dict'])

toTensor = transforms.ToTensor()  
toPIL = transforms.ToPILImage()  


# testfunctions.parse(model,opt, opt.testFunction, 1)
img_in_pre,img_in,img_in_pos, img_gt = np.load("/Users/cc/Desktop/20201213_partitioned_testset_256/0-3.npy",allow_pickle=True)
img_in_pre,img_in,img_in_pos, hq = toTensor(img_in_pre).float(),toTensor(img_in).float(),toTensor(img_in_pos).float(), toTensor(img_gt).float()
lq = torch.cat((img_in_pre,img_in,img_in_pos), 0)
with torch.no_grad():
    sr = model(lq.unsqueeze(0))


print(sr.mean(),sr.shape)
sr = sr[0]
sr2 = sr.clone()
m = nn.LogSoftmax(dim=0)
sr = m(sr)
sr = sr.argmax(dim=0, keepdim=True)
sr = sr.float() / (opt.nch_out-1)
# sr = sr.detach()[0].permute(1,2,0)*255
# sr = sr.numpy().astype('uint8')
sr = toPIL(sr)


sr2 = sr2.argmax(dim=0, keepdim=True)
sr2 = sr2.float() / (opt.nch_out-1)
sr2 = toPIL(sr2)

import matplotlib.pyplot as plt
plt.subplot(141)
plt.imshow(img_in.numpy()[0])

plt.subplot(142)
plt.imshow(hq.numpy()[0])

plt.subplot(143)
plt.imshow(sr)

plt.subplot(144)
plt.imshow(sr2)

plt.show()