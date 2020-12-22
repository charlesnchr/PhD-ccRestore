from options import parser
# import testfunctions
from models import GetModel, ESRGAN_Discriminator, ESRGAN_FeatureExtractor
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt


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
blackTensor = torch.zeros(img_in.shape)
lq2 = torch.cat((blackTensor,img_in,blackTensor), 0)
lq3 = torch.cat((img_in,img_in,img_in), 0)


with torch.no_grad():
    sr = model(lq.unsqueeze(0))
    sr2 = model(lq2.unsqueeze(0))
    sr3 = model(lq3.unsqueeze(0))

print(sr.mean(),sr2.mean(),sr.shape)
sr = sr[0]
sr = sr.argmax(dim=0, keepdim=True)
sr = sr.float() / (opt.nch_out-1)
# sr = sr.detach()[0].permute(1,2,0)*255
# sr = sr.numpy().astype('uint8')
sr = toPIL(sr)

sr2 = sr2[0]
sr2 = sr2.argmax(dim=0, keepdim=True)
sr2 = sr2.float() / (opt.nch_out-1)
sr2 = toPIL(sr2)

sr3 = sr3[0]
sr3 = sr3.argmax(dim=0, keepdim=True)
sr3 = sr3.float() / (opt.nch_out-1)
sr3 = toPIL(sr3)

plt.figure(figsize=(25,5))
plt.subplot(151)
plt.imshow(img_in.numpy()[0])
plt.title("img_in")

plt.subplot(152)
plt.imshow(hq.numpy()[0])
plt.title("hq")

plt.subplot(153)
plt.imshow(sr)
plt.title("sr - sequential with displaced neighbours")

plt.subplot(154)
plt.imshow(sr2)
plt.title("sr2 - center + black neighbours")

plt.subplot(155)
plt.imshow(sr3)
plt.title("sr3 - triple center / single image")

plt.savefig('comparison.png',dpi=300)
plt.show()