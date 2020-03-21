import math
import os

import torch
import time 


import torch.optim as optim
import torchvision
from torch.autograd import Variable

from models import *
from datahandler import *

from options import opt
from plotting import testAndMakeCombinedPlots

from tensorboardX import SummaryWriter

import cv2

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
        


def remove_isolated_pixels(image):
    connectivity = 8

    output = cv2.connectedComponentsWithStats(image, connectivity, cv2.CV_32S)

    num_stats = output[0]
    labels = output[1]
    stats = output[2]

    new_image = image.copy()

    for label in range(num_stats):
        print(stats[label,cv2.CC_STAT_AREA],end=',')
        if stats[label,cv2.CC_STAT_AREA] < 50:
            new_image[labels == label] = 0
    print(' ')

    return new_image


def closed_image(image):
    kernel = np.ones((3,3),np.uint8)
    new_image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    return new_image


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

        if len(opt.scheduler) > 0:
            scheduler.step()
            for param_group in optimizer.param_groups:
                print('Learning rate',param_group['lr'])
                break
            


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
                    loss = loss_function(sr.squeeze(), hr.long().squeeze().cuda())
                else:
                    loss = loss_function(sr, hr.cuda())

            # custom loss test
            image_corr_bat = torch.zeros([opt.batchSize,opt.nch_out,opt.imageSize,opt.imageSize])
            image_pred_bat = torch.zeros([opt.batchSize,opt.nch_out,opt.imageSize,opt.imageSize])

            sr = sr.cpu()
            for loss_i,sr_sample in enumerate(sr):
                m = nn.LogSoftmax(dim=0)
                sr_sample = m(sr_sample)
                # print(sr)
                sr_sample = sr_sample.argmax(dim=0, keepdim=True)
                sr_sample = np.array(toPIL(sr_sample.float()))
                img2 = closed_image(sr_sample)
                                
                image_pred_bat[loss_i,:,:] = torch.tensor(sr_sample/255).float()
                image_corr_bat[loss_i,:,:] = torch.tensor(img2/255).float()
            
            image_pred_bat.cuda()
            image_corr_bat.cuda()
            loss2 = loss_function_custom(image_pred_bat,image_corr_bat)

            total_loss = 0.2*loss + 1000*loss2
            total_loss.backward()

            print('loss: %0.5f, %0.5f, %0.5f' % (total_loss.data.item(),loss.data.item(),loss2.data.item()))

            # loss.backward()
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
                # print('iter,nsample,time,memory,meanloss',file=opt.stats)
                print(epoch, count*opt.batchSize, t1, mem, mean_loss / count, file=opt.train_stats)
                count = 0


        print('\nEpoch %d done, %0.6f' % (epoch,(mean_loss / len(dataloader))))
        print('\nEpoch %d done, %0.6f' % (epoch,(mean_loss / len(dataloader))),file=opt.fid)
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
            torch.save(checkpoint, opt.out + '/prelim.pth')
    
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
    

    if opt.dataset.lower() == 'div2k':
        dataloader = load_Pickle_dataset(opt.root + '/DIV2K','train',opt)
        validloader = load_Pickle_dataset(opt.root + '/DIV2K','valid',opt)
    elif opt.dataset.lower() == 'pcam': 
        dataloader = load_HDF5_dataset(opt.root + '/camelyonpatch_level_2_split_valid_x.h5','train',opt)
        validloader = load_HDF5_dataset(opt.root + '/camelyonpatch_level_2_split_valid_x.h5','valid',opt)
    elif opt.dataset.lower() == 'imagedataset': 
        dataloader = load_image_dataset(opt.root,'train',opt)
        validloader = load_image_dataset(opt.root,'valid',opt)
    elif opt.dataset.lower() == 'doubleimagedataset': 
        dataloader = load_doubleimage_dataset(opt.root,'train',opt)
        validloader = load_doubleimage_dataset(opt.root,'valid',opt)
    elif opt.dataset.lower() == 'pickledataset': 
        dataloader = load_GenericPickle_dataset(opt.root,'train',opt)
        validloader = load_GenericPickle_dataset(opt.root,'valid',opt)
    elif opt.dataset.lower() == 'sim': 
        dataloader = load_SIM_dataset(opt.root,'train',opt)
        validloader = load_SIM_dataset(opt.root,'valid',opt)
    elif opt.dataset.lower() == 'realsim': 
        dataloader = load_real_SIM_dataset(opt.root,'train',opt)
        validloader = load_real_SIM_dataset(opt.root,'valid',opt)        
    elif opt.dataset.lower() == 'ntiredenoising': 
        dataloader = load_NTIREDenoising_dataset(opt.root,'train',opt)
        validloader = load_NTIREDenoising_dataset(opt.root,'valid',opt)                
    elif opt.dataset.lower() == 'ntireestimatenl': 
        dataloader = load_EstimateNL_dataset(opt.root,'train',opt)
        validloader = load_EstimateNL_dataset(opt.root,'valid',opt)                
    elif opt.dataset.lower() == 'er':
        dataloader = load_ER_dataset(opt.root,category='train',batchSize=opt.batchSize,num_workers=opt.workers)
        validloader = load_ER_dataset(opt.root,category='valid',shuffle=False,batchSize=opt.batchSize,num_workers=0)        
    else:
        print('unknown dataset')
        import sys
        sys.exit()
        

    if opt.model.lower() == 'edsr':
        net = EDSR(opt)
    elif opt.model.lower() == 'edsr2max':
        net = EDSR2Max(normalization=opt.norm,nch_in=opt.nch_in,nch_out=opt.nch_out,scale=opt.scale)
    elif opt.model.lower() == 'edsr3max':
        net = EDSR3Max(normalization=opt.norm,nch_in=opt.nch_in,nch_out=opt.nch_out,scale=opt.scale)
    elif opt.model.lower() == 'rcan':
        net = RCAN(opt)
    elif opt.model.lower() == 'srresnet' or opt.model.lower() == 'srgan':
        net = Generator(16, opt)
    elif opt.model.lower() == 'unet':        
        net = UNet(opt.nch_in,opt.nch_out,opt)
    elif opt.model.lower() == 'unet_n2n':        
        net = UNet_n2n(opt.nch_in,opt.nch_out,opt)
    elif opt.model.lower() == 'unet60m':        
        net = UNet60M(opt.nch_in,opt.nch_out)
    elif opt.model.lower() == 'unetrep':        
        net = UNetRep(opt.nch_in,opt.nch_out)        
    elif opt.model.lower() == 'unetgreedy':        
        net = UNetGreedy(opt.nch_in,opt.nch_out)        
    elif opt.model.lower() == 'mlpnet':        
        net = MLPNet()                
    elif opt.model.lower() == 'ffdnet':        
        net = FFDNet(opt.nch_in)
    elif opt.model.lower() == 'dncnn':        
        net = DNCNN(opt.nch_in)
    else:
        print("model undefined")
    
    if not opt.cpu:
        net.cuda()
        if opt.multigpu:
            net = nn.DataParallel(net)
    
    # if len(opt.weights) > 0: # load previous weights?
    #     net.load_state_dict(torch.load(opt.weights))

    if opt.log:
        opt.writer = SummaryWriter(comment='_%s_%s' % (opt.out.replace('\\','/').split('/')[-1], opt.model))
        opt.train_stats = open(opt.out.replace('\\','/') + '/train_stats.csv','w')
        opt.test_stats = open(opt.out.replace('\\','/') + '/test_stats.csv','w')
        print('iter,nsample,time,memory,meanloss',file=opt.train_stats)
        print('iter,time,memory,psnr,ssim',file=opt.test_stats)

    import time
    t0 = time.perf_counter()
    if not opt.test:
        if opt.model.lower() == 'srgan':
            GANtrain(dataloader, validloader, net, nepoch=opt.nepoch)
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
