#%%

from options import parser

from models import *

# from MLPNetwork import *

class VGGFeatureExtractor(nn.Module):
    def __init__(
        self,
        feature_layer=34,
        use_bn=False,
        use_input_norm=True,
        device=torch.device("cpu"),
    ):
        super(VGGFeatureExtractor, self).__init__()
        self.use_input_norm = use_input_norm
        model = vgg19()
        if self.use_input_norm:
            mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
            # [0.485 - 1, 0.456 - 1, 0.406 - 1] if input in range [-1, 1]
            std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
            # [0.229 * 2, 0.224 * 2, 0.225 * 2] if input in range [-1, 1]
            self.register_buffer("mean", mean)
            self.register_buffer("std", std)

        self.features = nn.Sequential(
            *list(model.features.children())[: (feature_layer + 1)]
        )
        # No need to BP to variable
        for k, v in self.features.named_parameters():
            v.requires_grad = False

    def forward(self, x):
        # Assume input range is [0, 1]
        if self.use_input_norm:
            x = (x - self.mean) / self.std
        output = self.features(x)
        return output


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

# net = RCAN(opt)
# net.cuda()

net = VGGFeatureExtractor()

summary(net, input_size=(3,300,300))


