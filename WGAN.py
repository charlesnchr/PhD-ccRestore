from torch import nn
from torchvision import models
import torch.nn.functional as F
import torch
from torch.autograd import Variable,grad
from torch.optim import Adam
import time
import tqdm
from plotting import testAndMakeCombinedPlots

def make_trainable(model, val):
    for p in model.parameters():
        p.requires_grad = val


def calc_gradient_penalty(netD, real_data, fake_data,LAMBDA=10):
    BATCH=real_data.size()[0]
    alpha = torch.rand(BATCH, 1)
    #print(alpha.size(),real_data.size())
    alpha = alpha.unsqueeze(-1).unsqueeze(-1).expand(real_data.size())
    alpha = alpha.cuda()
    interpolates = alpha * real_data + ((1 - alpha) * fake_data)
    interpolates = interpolates.cuda()
    interpolates = Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates)

    gradients = grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda(),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty


class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss = nn.NLLLoss2d(weight, size_average)

    def forward(self, inputs, targets):
        #print(inputs.size())
        return self.nll_loss(F.log_softmax(inputs,dim=1), targets)

class BCE_Loss(nn.Module):
    def __init__(self):
        super(BCE_Loss,self).__init__()
        self.bce=nn.BCELoss()
    def forward(self,inputs,targets):
        return self.bce(inputs,targets)


class block(nn.Module):
    def __init__(self,in_filters,n_filters):
        super(block,self).__init__()
        self.deconv1 = nn.Sequential(
            nn.Conv2d(in_filters, n_filters, 3, stride=1, padding=1),
            nn.BatchNorm2d(n_filters),
            nn.ReLU())
    def forward(self, x):
        x=self.deconv1(x)
        return x

class generator(nn.Module):
    # initializers
    def __init__(self, n_filters=32):
        super(generator, self).__init__()
        self.down1=nn.Sequential(
            block(1,n_filters),
            block(n_filters,n_filters),
            nn.MaxPool2d((2,2)))
        self.down2 = nn.Sequential(
            block(n_filters, 2*n_filters),
            block(2*n_filters, 2*n_filters),
            nn.MaxPool2d((2, 2)))
        self.down3 = nn.Sequential(
            block(2*n_filters, 4*n_filters),
            block(4*n_filters, 4*n_filters),
            nn.MaxPool2d((2, 2)))
        self.down4 = nn.Sequential(
            block(4*n_filters, 8 * n_filters),
            block(8 * n_filters, 8 * n_filters),
            nn.MaxPool2d((2, 2)))
        self.down5 = nn.Sequential(
            block(8 * n_filters, 16 * n_filters),
            block(16 * n_filters, 16 * n_filters))

        self.up1=nn.Sequential(
            block(16 * n_filters+8*n_filters, 8 * n_filters),
            block(8 * n_filters, 8 * n_filters))
        self.up2 = nn.Sequential(
            block(8 * n_filters+4*n_filters, 4 * n_filters),
            block(4 * n_filters, 4 * n_filters))
        self.up3 = nn.Sequential(
            block(4 * n_filters+2*n_filters,2 * n_filters),
            block(2 * n_filters, 2 * n_filters))
        self.up4 = nn.Sequential(
            block(2 * n_filters+n_filters,  n_filters),
            block( n_filters,  n_filters))

        self.out=nn.Sequential(
            nn.Conv2d(n_filters,1,kernel_size=1)
        )
    # forward method
    def forward(self, x):
        #print(x.size())
        x1=self.down1(x)
        #print(x1.size())
        x2=self.down2(x1)
        #print(x2.size())
        x3 = self.down3(x2)
        #print(x3.size())
        x4 = self.down4(x3)
        #print(x4.size())
        x5 = self.down5(x4)
        #print(x5.size())
        x = self.up1(F.upsample(torch.cat((x4,x5),dim=1),scale_factor=2))
        x = self.up2(F.upsample(torch.cat((x, x3), dim=1), scale_factor=2))
        x = self.up3(F.upsample(torch.cat((x, x2), dim=1), scale_factor=2))
        x = self.up4(F.upsample(torch.cat((x, x1), dim=1), scale_factor=2))
        x=F.sigmoid(self.out(x))
        return x#b,1,w,h

class discriminator(nn.Module):
    def __init__(self,n_filters):
        super(discriminator,self).__init__()
        self.down1 = nn.Sequential(
            block(2, n_filters),
            block(n_filters, n_filters),
            nn.MaxPool2d((2, 2)))
        self.down2 = nn.Sequential(
            block(n_filters, 2 * n_filters),
            block(2 * n_filters, 2 * n_filters),
            nn.MaxPool2d((2, 2)))
        self.down3 = nn.Sequential(
            block(2 * n_filters, 4 * n_filters),
            block(4 * n_filters, 4 * n_filters),
            nn.MaxPool2d((2, 2)))
        self.down4 = nn.Sequential(
            block(4 * n_filters, 8 * n_filters),
            block(8 * n_filters, 8 * n_filters),
            nn.MaxPool2d((2, 2)))
        self.down5 = nn.Sequential(
            block(8 * n_filters, 16 * n_filters),
            block(16 * n_filters, 16 * n_filters))
        self.out = nn.Linear(16*n_filters,1)
    def forward(self, x):
        x=self.down1(x)
        #print(x.size())
        x = self.down2(x)
        #print(x.size())
        x = self.down3(x)
        #print(x.size())
        x = self.down4(x)
        #print(x.size())
        x = self.down5(x)
        #print(x.size())
        x=F.avg_pool2d(x, kernel_size=x.size()[2:]).view(x.size()[0], -1)


        x=self.out(x)
        x = F.sigmoid(x)
        #print(x.size())
        return x#b,1


# ===================== WGAN train =====================


def train(opt, dataloader, validloader):

    start_epoch = 0

    # Losses
    D=discriminator(n_filters=opt.n_feats).cuda()
    G=generator(n_filters=opt.n_feats).cuda()
    gan_loss_percent=opt.gan_loss

    one=torch.tensor(1, dtype=torch.float)
    mone=one*-1
    moneg=one*-1*gan_loss_percent

    one=one.cuda()
    mone=mone.cuda()
    moneg=moneg.cuda()

    # Optimizers
    loss_func=BCE_Loss()
    optimizer_D=Adam(D.parameters(),lr=opt.lr,betas=(0.5,0.9),eps=10e-8)
    optimizer_G=Adam(G.parameters(),lr=opt.lr,betas=(0.5,0.9),eps=10e-8)

    opt.t0 = time.perf_counter()

    for epoch in range(start_epoch, opt.nepoch):
        
        D.train()
        G.train()
        #train D
        make_trainable(D,True)
        make_trainable(G,False)

        for i, (real_imgs,real_labels) in tqdm.tqdm(enumerate(dataloader)):
            real_imgs=Variable(real_imgs).cuda()
            real_labels=Variable(real_labels).cuda()
            D.zero_grad()
            optimizer_D.zero_grad()
            real_pair = torch.cat((real_imgs, real_labels), dim=1)
            #real_pair_y=Variable(torch.ones((real_pair.size()[0],1))).cuda()
            d_real = D(real_pair)
            d_real = d_real.mean()
            d_real.backward(mone)

            fake_pair=torch.cat((real_imgs, G(real_imgs)), dim=1)
            #fake_pair_y=Variable(torch.zeros((real_pair.size()[0],1))).cuda()
            d_fake=D(fake_pair)
            d_fake=d_fake.mean()
            d_fake.backward(one)

            #d_loss=loss_func(D(real_pair),real_pair_y)+loss_func(D(fake_pair),fake_pair_y)
            #d_loss.backward()
            gradient_penalty=calc_gradient_penalty(D,real_pair.data,fake_pair.data)
            gradient_penalty.backward()

            Wasserstein_D=d_real-d_fake
            optimizer_D.step()
        #train G

        make_trainable(D,False)
        make_trainable(G,True)
        for i,(real_imgs,real_labels) in tqdm.tqdm(enumerate(dataloader)):
            G.zero_grad()
            optimizer_G.zero_grad()
            real_imgs=Variable(real_imgs).cuda()
            real_labels=Variable(real_labels).cuda()
            pred_labels=G(real_imgs)
            Seg_Loss=loss_func(pred_labels,real_labels.unsqueeze(1))#Seg Loss
            Seg_Loss.backward(retain_graph=True)
            fake_pair=torch.cat((real_imgs,pred_labels),dim=1)
            gd_fake=D(fake_pair)
            gd_fake=gd_fake.mean()
            gd_fake.backward(moneg)
            #Gan_Loss=loss_func(D_fack,Variable(torch.ones(fake_pair.size()[0],1)).cuda())
            #g_loss=Gan_Loss*gan_loss_percent+Seg_Loss
            #g_loss.backward()
            optimizer_G.step()
        print("epoch[%d/%d] W:%f segloss%f"%(epoch,opt.nepoch,Wasserstein_D,Seg_Loss))
        
        # ---------------- Printing -----------------
        if opt.log:
            t1 = time.perf_counter() - opt.t0
            mem = torch.cuda.memory_allocated()
            opt.writer.add_scalar('loss/W', Wasserstein_D, epoch)
            opt.writer.add_scalar('loss/segloss', Seg_Loss, epoch)
            opt.writer.add_scalar('data/time', t1, epoch)
            opt.writer.add_scalar('data/mem', mem, epoch)
            
        # ---------------- TEST -----------------
        if (epoch + 1) % opt.testinterval == 0:
            testAndMakeCombinedPlots(G, validloader, opt, epoch)
            # if opt.scheduler:
            # scheduler.step(mean_loss / len(dataloader))

    # torch.save(checkpoint, opt.out + '/final.pth')