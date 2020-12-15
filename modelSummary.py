#%%

from options import parser

from models import *

# from MLPNetwork import *

from torchsummary import summary

opt = parser.parse_args()


# net = RCAN()
# net = UNet_n2n(3,3)
# net = UNet(3,3)
# net = UNetRep(3,3)
# net = UNetGreedy(3,3)
# net = UNet(3,3)
# net.cuda()

opt.nch_in = 1
opt.nch_out = 1

net = RCAN(opt)
net.cuda()

summary(net, input_size=(1,128,128))


