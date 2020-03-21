import math
import os

import torch
import torch.nn as nn
import time 


import torch.optim as optim
import torchvision
from torch.autograd import Variable

from models import GetModel
from datahandler import GetDataloaders

from plotting import testAndMakeCombinedPlots

# from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter

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

if opt.out[:4] == 'root':
    opt.out = opt.out.replace('root',opt.root)


# convenience function
if len(opt.weights) > 0 and not os.path.isfile(opt.weights):
    # folder provided, trying to infer model options
    
    logfile = opt.weights + '/log.txt'
    opt.weights += '/final.pth'
    if not os.path.isfile(opt.weights):
        opt.weights = opt.weights.replace('final.pth','prelim.pth')

    if os.path.isfile(logfile):
        fid = open(logfile,'r')
        optstr = fid.read()
        optlist = optstr.split(', ')

        def getopt(optname,typestr):
            opt_e = [e.split('=')[-1].strip("\'") for e in optlist if (optname.split('.')[-1] + '=') in e]
            return eval(optname) if len(opt_e) == 0 else typestr(opt_e[0])
            
        opt.model = getopt('opt.model',str)
        opt.task = getopt('opt.task',str)
        opt.nch_in = getopt('opt.nch_in',int)
        opt.nch_out = getopt('opt.nch_out',int)
        opt.n_resgroups = getopt('opt.n_resgroups',int)
        opt.n_resblocks = getopt('opt.n_resblocks',int)
        opt.n_feats = getopt('opt.n_feats',int)


def remove_dataparallel_wrapper(state_dict):
	r"""Converts a DataParallel model to a normal one by removing the "module."
	wrapper in the module dictionary

	Args:
		state_dict: a torch.nn.DataParallel state dictionary
	"""
	from collections import OrderedDict

	new_state_dict = OrderedDict()
	for k, vl in state_dict.items():
		name = k[7:] # remove 'module.' of DataParallel
		new_state_dict[name] = vl

	return new_state_dict




def ESRGANtrain(dataloader, validloader, generator, nepoch=10):
    
    start_epoch = 0
    
    discriminator = Discriminator_VGG_128(in_nc=3, nf=64).cuda()
    netF = VGGFeatureExtractor(feature_layer=34, use_bn=False,
                                          use_input_norm=True).cuda()

    # feature_extractor = FeatureExtractor(torchvision.models.vgg19(pretrained=True)) 
    # feature_extractor.cuda()
    # content_criterion = nn.MSELoss()

    # adversarial_criterion = nn.BCELoss()

    cri_pix = nn.L1Loss().cuda()
    cri_fea = nn.L1Loss().cuda()
    cri_gan = nn.BCEWithLogitsLoss().cuda()

    # discriminator.cuda()
    # content_criterion.cuda()
    # adversarial_criterion.cuda()

    optim_generator = optim.Adam(generator.parameters(), lr=opt.lr)

    if len(opt.weights) > 0: # load previous weights?
        checkpoint = torch.load(opt.weights)
        print('loading checkpoint',opt.weights)
        if opt.undomulti:
            checkpoint['state_dict'] = remove_dataparallel_wrapper(checkpoint['state_dict'])
        else:
            generator.load_state_dict(checkpoint['state_dict'])
            optim_generator.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch']
    else:
        # pretraining
        print('Starting pretraining')
        for epoch in range(20):
            mean_loss = 0

            for i, bat in enumerate(dataloader):
                lr, hr = bat[0], bat[1]
                
                sr = generator(lr.cuda())

                optim_generator.zero_grad()
                loss = cri_pix(sr, hr.cuda())            
                loss.backward()
                optim_generator.step()
                
                ######### Status and display #########
                mean_loss += loss.data.item()
                print('\r[%d/%d][%d/%d] Loss: %0.6f' % (epoch+1,3,i+1,len(dataloader),loss.data.item()),end='')

            print('\nEpoch %d done, %0.8f' % (epoch,(mean_loss / len(dataloader))))
    

    ones_const = torch.ones(opt.batchSize, 1).cuda()

    optim_generator = optim.Adam(generator.parameters(), lr=opt.lr, betas=(0.9,0.99))
    optim_discriminator = optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(0.9,0.99))

    if len(opt.scheduler) > 0:
        # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True, threshold=0.0001, threshold_mode='rel', cooldown=5, min_lr=0, eps=1e-08)
        stepsize, gamma = int(opt.scheduler.split(',')[0]), float(opt.scheduler.split(',')[1])
        scheduler = optim.lr_scheduler.StepLR(optim_generator, stepsize, gamma=gamma, last_epoch=start_epoch-1)


    # GAN training
    print('GAN training')
    for epoch in range(start_epoch, nepoch):
        mean_loss = 0
        mean_discriminator_loss = 0
        mean_generator_content_loss = 0
        mean_generator_adversarial_loss = 0
        mean_generator_total_loss = 0

        if len(opt.scheduler) > 0:
            scheduler.step()
            for param_group in optim_generator.param_groups:
                print('Learning rate',param_group['lr'])
                break


        for i, bat in enumerate(dataloader):
            var_L, var_H = bat[0].cuda(), bat[1].cuda()

            # G
            for p in discriminator.parameters():
                p.requires_grad = False

            optim_generator.zero_grad()
            fake_H = generator(var_L)

            l_g_total = 0
            l_pix_w = 1e-2
            l_fea_w = 1e-2
            l_gan_w = 5e-3

            l_g_pix = l_pix_w * cri_pix(fake_H, var_H)
            l_g_total += l_g_pix

            real_fea = netF(var_H).detach()
            fake_fea = netF(fake_H)
            l_g_fea = l_fea_w * cri_fea(fake_fea, real_fea)
            l_g_total += l_g_fea

            pred_d_real = discriminator(var_H).detach()
            pred_g_fake = discriminator(fake_H)
            
            input1 = pred_d_real - torch.mean(pred_g_fake)
            input2 = pred_g_fake - torch.mean(pred_d_real)
            target_label_fake = torch.empty_like(input1).fill_(0)
            target_label_real = torch.empty_like(input2).fill_(1)

            l_g_gan = l_gan_w * (
                cri_gan(input1, target_label_fake) +
                cri_gan(input2, target_label_real)) / 2
            l_g_total += l_g_gan

            l_g_total.backward()
            optim_generator.step()   


            # D
            for p in discriminator.parameters():
                p.requires_grad = True


            optim_discriminator.zero_grad()
            pred_d_fake = discriminator(fake_H.detach()).detach()
            pred_d_real = discriminator(var_H)
            
            input1 = pred_d_real - torch.mean(pred_d_fake)
            l_d_real = cri_gan(input1, target_label_real) * 0.5
            l_d_real.backward()
            pred_d_fake = discriminator(fake_H.detach())
            input2 = pred_d_fake - torch.mean(pred_d_real.detach())
            l_d_fake = cri_gan(input2, target_label_fake) * 0.5
            l_d_fake.backward()
            optim_discriminator.step()

            ######### Status and display #########
            print('\r[%d/%d][%d/%d]  (real/fake): (%.6f/%.6f) : %.6f/%.6f/%.6f/%.6f' % 
            (epoch+1,nepoch,i+1,len(dataloader),l_d_real.data.item(), l_d_fake.data.item(), l_g_pix.data.item(), l_g_fea.data.item(), l_g_gan.data.item(), l_g_total.data.item()),end='')

        print('\nEpoch %d done, Discriminator_Loss: %.6f Generator_Loss (Content/Advers/Total): %.6f/%.6f/%.6f' % (epoch,mean_discriminator_loss/len(dataloader), mean_generator_content_loss/len(dataloader), 
    mean_generator_adversarial_loss/len(dataloader), mean_generator_total_loss/len(dataloader)))

        # ---------------- TEST -----------------
        if (epoch + 1) % opt.testinterval == 0:
            testAndMakeCombinedPlots(net,validloader,opt,epoch)
            # if opt.scheduler:
                # scheduler.step(mean_loss / len(dataloader))

        if (epoch + 1) % opt.saveinterval == 0:
            # torch.save(net.state_dict(), opt.out + '/prelim.pth')
            checkpoint = {'epoch': epoch + 1,
            'state_dict': net.state_dict(),
            'optim_generator' : optim_generator.state_dict(),
            'optim_discriminator' : optim_discriminator.state_dict() }
            torch.save(checkpoint, opt.out + '/prelim.pth')
    
    checkpoint = {'epoch': nepoch,
    'state_dict': net.state_dict(),
    'optim_generator' : optim_generator.state_dict(),
    'optim_discriminator' : optim_discriminator.state_dict() }
    torch.save(checkpoint, opt.out + '/final.pth')        
        




def GANtrain(dataloader, validloader, generator, nepoch=10):
    
    start_epoch = 0
    
    discriminator = Discriminator(opt)
    
    if opt.task == 'segment':
        content_criterion = nn.CrossEntropyLoss()
    else:
        feature_extractor = FeatureExtractor(torchvision.models.vgg19(pretrained=True)) 
        feature_extractor.cuda()
        content_criterion = nn.MSELoss()

    adversarial_criterion = nn.BCELoss()

    discriminator.cuda()
    content_criterion.cuda()
    adversarial_criterion.cuda()

    optim_generator = optim.Adam(generator.parameters(), lr=opt.lr)

    # if len(opt.weights) > 0: # load previous weights?
    #     checkpoint = torch.load(opt.weights)
    #     print('loading checkpoint',opt.weights)
    #     if opt.undomulti:
    #         checkpoint['state_dict'] = remove_dataparallel_wrapper(checkpoint['state_dict'])
    #     else:
    #         net.load_state_dict(checkpoint['state_dict'])
    #         optimizer.load_state_dict(checkpoint['optimizer'])
    #         start_epoch = checkpoint['epoch']

    # pretraining
    print('Starting pretraining')
    for epoch in range(3):
        mean_loss = 0

        for i, bat in enumerate(dataloader):
            lr, hr = bat[0], bat[1]
            
            sr = generator(lr.cuda())

            optim_generator.zero_grad()
            if opt.task == 'segment':
                loss = content_criterion(sr.squeeze(), hr.long().squeeze().cuda())
            else:
                loss = content_criterion(sr, hr.cuda())            
            loss.backward()
            optim_generator.step()
            
            ######### Status and display #########
            mean_loss += loss.data.item()
            print('\r[%d/%d][%d/%d] Loss: %0.6f' % (epoch+1,3,i+1,len(dataloader),loss.data.item()),end='')

        print('\nEpoch %d done, %0.8f' % (epoch,(mean_loss / len(dataloader))))
    

    ones_const = torch.ones(opt.batchSize, 1).cuda()

    optim_generator = optim.Adam(generator.parameters(), lr=opt.lr)
    optim_discriminator = optim.Adam(discriminator.parameters(), lr=opt.lr)

    if len(opt.scheduler) > 0:
        # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True, threshold=0.0001, threshold_mode='rel', cooldown=5, min_lr=0, eps=1e-08)
        stepsize, gamma = int(opt.scheduler.split(',')[0]), float(opt.scheduler.split(',')[1])
        scheduler = optim.lr_scheduler.StepLR(optim_generator, stepsize, gamma=gamma, last_epoch=start_epoch-1)


    # GAN training
    print('GAN training')
    for epoch in range(start_epoch, nepoch):
        mean_loss = 0
        mean_discriminator_loss = 0
        mean_generator_content_loss = 0
        mean_generator_adversarial_loss = 0
        mean_generator_total_loss = 0

        if len(opt.scheduler) > 0:
            scheduler.step()
            for param_group in optim_generator.param_groups:
                print('Learning rate',param_group['lr'])
                break


        for i, bat in enumerate(dataloader):
            lr, hr = bat[0].cuda(), bat[1].cuda()

            sr = generator(lr)

            target_real = (torch.rand(opt.batchSize,1)*0.5 + 0.7).cuda()
            target_fake = (torch.rand(opt.batchSize,1)*0.3).cuda()
 
            ######### Train discriminator #########
            optim_discriminator.zero_grad()
            # discriminator.zero_grad()
            
            if opt.task == 'segment':
                m = nn.LogSoftmax(dim=1)
                sr_org = sr
                sr = m(sr_org)
                sr = sr.argmax(dim=1, keepdim=True)
                sr = sr.float()

            discriminator_loss = adversarial_criterion(discriminator(hr), target_real) + adversarial_criterion(discriminator(Variable(sr.data)), target_fake)
            mean_discriminator_loss += discriminator_loss.data.item()
            discriminator_loss.backward()
            optim_discriminator.step()

            ######### Train generator #########
            optim_generator.zero_grad()
            # generator.zero_grad()

            if opt.task == 'segment':
                generator_content_loss = content_criterion(sr_org.squeeze(), hr.long().squeeze())
            else:
                real_features = Variable(feature_extractor(hr))
                fake_features = Variable(feature_extractor(sr))
                generator_content_loss = content_criterion(sr, hr) + 0.006*content_criterion(fake_features, real_features)

            mean_generator_content_loss += generator_content_loss.data.item()
            
            generator_adversarial_loss = adversarial_criterion(discriminator(Variable(sr.data)), ones_const)
            mean_generator_adversarial_loss += generator_adversarial_loss.data.item()

            generator_total_loss = generator_content_loss + 1e-1*generator_adversarial_loss
            mean_generator_total_loss += generator_total_loss.data.item()
            
            generator_total_loss.backward()
            optim_generator.step()   

            ######### Status and display #########
            print('\r[%d/%d][%d/%d] Discriminator_Loss: %.6f Generator_Loss (Content/Advers/Total): %.6f/%.6f/%.6f' % (epoch+1,nepoch,i+1,len(dataloader),discriminator_loss.data.item(), generator_content_loss.data.item(), generator_adversarial_loss.data.item(), generator_total_loss.data.item()),end='')

        print('\nEpoch %d done, Discriminator_Loss: %.6f Generator_Loss (Content/Advers/Total): %.6f/%.6f/%.6f' % (epoch,mean_discriminator_loss/len(dataloader), mean_generator_content_loss/len(dataloader), 
    mean_generator_adversarial_loss/len(dataloader), mean_generator_total_loss/len(dataloader)))

        # ---------------- TEST -----------------
        if (epoch + 1) % opt.testinterval == 0:
            testAndMakeCombinedPlots(net,validloader,opt,epoch)
            # if opt.scheduler:
                # scheduler.step(mean_loss / len(dataloader))

        if (epoch + 1) % opt.saveinterval == 0:
            # torch.save(net.state_dict(), opt.out + '/prelim.pth')
            checkpoint = {'epoch': epoch + 1,
            'state_dict': net.state_dict(),
            'optim_generator' : optim_generator.state_dict(),
            'optim_discriminator' : optim_discriminator.state_dict() }
            torch.save(checkpoint, opt.out + '/prelim.pth')
    
    checkpoint = {'epoch': nepoch,
    'state_dict': net.state_dict(),
    'optim_generator' : optim_generator.state_dict(),
    'optim_discriminator' : optim_discriminator.state_dict() }
    torch.save(checkpoint, opt.out + '/final.pth')        
        


def train(dataloader, validloader, net, nepoch=10):
    
    start_epoch = 0
    if opt.task == 'segment':
        loss_function = nn.CrossEntropyLoss()
    else:
        loss_function = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=opt.lr)
    loss_function.cuda()

    loss_function_custom = nn.MSELoss()
    loss_function_custom.cuda()


    if len(opt.weights) > 0: # load previous weights?
        checkpoint = torch.load(opt.weights)
        print('loading checkpoint',opt.weights)
        if opt.undomulti:
            checkpoint['state_dict'] = remove_dataparallel_wrapper(checkpoint['state_dict'])
        if opt.modifyPretrainedModel:
            pretrained_dict = checkpoint['state_dict']
            model_dict = net.state_dict()
            # 1. filter out unnecessary keys
            for k,v in list(pretrained_dict.items()):
                print(k)
            pretrained_dict = {k: v for k, v in list(pretrained_dict.items())[:-2]}
            # 2. overwrite entries in the existing state dict
            model_dict.update(pretrained_dict)
            # 3. load the new state dict
            net.load_state_dict(model_dict)

            # optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch']
        else:
            net.load_state_dict(checkpoint['state_dict'])
            if opt.lr == -1: # continue as it was
                optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch']

        # if opt.modifyPretrainedModel:
        #     mod = list(net.children())
        #     mod.pop()
        #     mod.append(nn.Conv2d(64, 2, 1))
        #     net = torch.nn.Sequential(*mod)
        #     net.cuda()
        #     opt.task = 'segment'


    if len(opt.scheduler) > 0:
        # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True, threshold=0.0001, threshold_mode='rel', cooldown=5, min_lr=0, eps=1e-08)
        stepsize, gamma = int(opt.scheduler.split(',')[0]), float(opt.scheduler.split(',')[1])
        scheduler = optim.lr_scheduler.StepLR(optimizer, stepsize, gamma=gamma, last_epoch=start_epoch-1)

    count = 0
    opt.t0 = time.perf_counter()

    for epoch in range(start_epoch, nepoch):
        mean_loss = 0

        # if len(opt.lrseq) > 0:
        #     t = '[5,1e-4,10,1e-5]'
        #     t = np.array(t)
        #     epochvec = t[::2].astype('int')
        #     lrvec = t[1::2].astype('float')

        #     idx = epochvec.indexOf(epoch)
        #     opt.lr = lrvec[idx]
        #     optimizer = optim.Adam(net.parameters(), lr=opt.lr)


        for i, bat in enumerate(dataloader):
            lr, hr = bat[0], bat[1]
            
            optimizer.zero_grad()
            if opt.model == 'ffdnet':
                stdvec = torch.zeros(lr.shape[0])
                for j in range(lr.shape[0]):
                    noise = lr[j] - hr[j]
                    stdvec[j] = torch.std(noise)
                noise = net(lr.cuda(), stdvec.cuda())
                sr = torch.clamp( lr.cuda() - noise,0,1 )
                gt_noise = lr.cuda() - hr.cuda()
                loss = loss_function(noise, gt_noise)
            elif opt.task == 'residualdenoising':
                noise = net(lr.cuda())
                gt_noise = lr.cuda() - hr.cuda()
                loss = loss_function(noise, gt_noise)
            else:
                sr = net(lr.cuda())
                if opt.task == 'segment':
                    if opt.nch_out > 2:
                        hr_classes = torch.round((opt.nch_out+1)*hr).long()
                        loss = loss_function(sr.squeeze(), hr_classes.squeeze().cuda())
                    else:
                        loss = loss_function(sr.squeeze(), hr.long().squeeze().cuda())
                else:
                    loss = loss_function(sr, hr.cuda())

            loss.backward()
            optimizer.step()
            
            
            ######### Status and display #########
            mean_loss += loss.data.item()
            print('\r[%d/%d][%d/%d] Loss: %0.6f' % (epoch+1,nepoch,i+1,len(dataloader),loss.data.item()),end='')
            
            count += 1
            if opt.log and count*opt.batchSize // 1000 > 0:
                t1 = time.perf_counter() - opt.t0
                mem = torch.cuda.memory_allocated()
                opt.writer.add_scalar('data/mean_loss_per_1000', mean_loss / count, epoch)
                opt.writer.add_scalar('data/time_per_1000', t1, epoch)
                print(epoch, count*opt.batchSize, t1, mem, mean_loss / count, file=opt.train_stats)
                opt.train_stats.flush()
                count = 0



        # ---------------- Scheduler -----------------
        if len(opt.scheduler) > 0:
            scheduler.step()
            for param_group in optimizer.param_groups:
                print('\nLearning rate',param_group['lr'])
                break        


        # ---------------- Printing -----------------
        print('\nEpoch %d done, %0.6f' % (epoch,(mean_loss / len(dataloader))))
        print('\nEpoch %d done, %0.6f' % (epoch,(mean_loss / len(dataloader))),file=opt.fid)
        opt.fid.flush()
        if opt.log:
            opt.writer.add_scalar('data/mean_loss', mean_loss / len(dataloader), epoch)


        # ---------------- TEST -----------------
        if (epoch + 1) % opt.testinterval == 0:
            testAndMakeCombinedPlots(net,validloader,opt,epoch)
            # if opt.scheduler:
                # scheduler.step(mean_loss / len(dataloader))

        if (epoch + 1) % opt.saveinterval == 0:
            # torch.save(net.state_dict(), opt.out + '/prelim.pth')
            checkpoint = {'epoch': epoch + 1,
            'state_dict': net.state_dict(),
            'optimizer' : optimizer.state_dict() }
            torch.save(checkpoint, '%s/prelim%d.pth' % (opt.out,epoch+1))
    
    checkpoint = {'epoch': nepoch,
    'state_dict': net.state_dict(),
    'optimizer' : optimizer.state_dict() }
    torch.save(checkpoint, opt.out + '/final.pth')

if __name__ == '__main__':

    try:
        os.makedirs(opt.out)
    except IOError:
        pass

    opt.fid = open(opt.out + '/log.txt','w')
    print(opt)
    print(opt,'\n',file=opt.fid)
    print('getting dataloader',opt.root)
    dataloader, validloader = GetDataloaders(opt)        
    net = GetModel(opt)
    
    if opt.log:
        opt.writer = SummaryWriter(logdir=opt.out, comment='_%s_%s' % (opt.out.replace('\\','/').split('/')[-1], opt.model))
        opt.train_stats = open(opt.out.replace('\\','/') + '/train_stats.csv','w')
        opt.test_stats = open(opt.out.replace('\\','/') + '/test_stats.csv','w')
        print('iter,nsample,time,memory,meanloss',file=opt.train_stats)
        print('iter,time,memory,psnr,ssim',file=opt.test_stats)

    import time
    t0 = time.perf_counter()
    if not opt.test:
        if opt.model.lower() == 'srgan':
            GANtrain(dataloader, validloader, net, nepoch=opt.nepoch)
        elif opt.model.lower() == 'esrgan':
            ESRGANtrain(dataloader, validloader, net, nepoch=opt.nepoch)
        else:
            train(dataloader, validloader, net, nepoch=opt.nepoch)
        # torch.save(net.state_dict(), opt.out + '/final.pth')
    else:
        if len(opt.weights) > 0: # load previous weights?
            checkpoint = torch.load(opt.weights)
            print('loading checkpoint',opt.weights)
            if opt.undomulti:
                checkpoint['state_dict'] = remove_dataparallel_wrapper(checkpoint['state_dict'])
            net.load_state_dict(checkpoint['state_dict'])
            print('time: ',time.perf_counter()-t0)
        testAndMakeCombinedPlots(net,validloader,opt)
    print('time: ',time.perf_counter()-t0)

    if opt.log:
        opt.writer.export_scalars_to_json(opt.out + "/rundata.json")
        opt.writer.close()
