import testfunctions
from options import parser
from models import GetModel, ESRGAN_Discriminator, ESRGAN_FeatureExtractor
import torch

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
checkpoint = torch.load("/Volumes/lag.radian.dk/meng_3colours_4_20201210/final.pth",map_location=device)
model.load_state_dict(checkpoint['state_dict'])

testfunctions.parse(model,opt, opt.testFunction, 1)