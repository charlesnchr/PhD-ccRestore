import math
import os

import torch
import time 

import torch.optim as optim
import torchvision
from torch.autograd import Variable

from models import *

from tensorboardX import SummaryWriter

from skimage.restoration import (denoise_tv_chambolle, denoise_bilateral,
                                 denoise_wavelet, estimate_sigma)

from torch.utils.data import Dataset, DataLoader

class BatchDataset(Dataset):

    def __init__(self, root, opt):
    
        if root.split('.')[-1] == 'png' or root.split('.')[-1] == 'jpg':
            self.images = [root]
        else:
            self.images = []
            self.images.extend(glob.glob(root + '/*.jpg'))
            self.images.extend(glob.glob(root + '/*.png'))

        self.len = len(self.images)
        
    def __getitem__(self, index):
        img = Image.open(self.images[index])

        dn = denoise_tv_chambolle(np.array(img), weight=0.05)
        
        denoised_filename = self.images[index].replace('channel_','denoised_')
        Image.fromarray((255*dn).astype('uint8')).save(denoised_filename)

        return toTensor(img)

    def __len__(self):
        return self.len


def load_batch_dataset(root,opt):
    dataset = BatchDataset(root, opt)
    dataloader = DataLoader(dataset, batch_size=opt.batchSize, shuffle=False, num_workers=opt.workers)
    return dataloader



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


def EvaluateModel(opt):

    try:
        os.makedirs(opt.out)
    except IOError:
        pass

    opt.fid = open(opt.out + '/log.txt','w')
    print(opt)
    print(opt,'\n',file=opt.fid)
    
        

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
    
    net.cuda()
    checkpoint = torch.load(opt.weights)
    if opt.cpu:
        net.cpu()

    print('loading checkpoint',opt.weights)
    if opt.undomulti:
        checkpoint['state_dict'] = remove_dataparallel_wrapper(checkpoint['state_dict'])
    
    net.load_state_dict(checkpoint['state_dict'])


    from skimage import io
    import matplotlib.pyplot as plt
    import glob

    equal_partition = False

    dataloader = load_image_dataset(opt.root,,opt)

    imageSize = opt.imageSize


    # for i, imgfile in enumerate(imgs):
    for i, bat in enumerate(dataloader):
        # img = np.array(Image.open(imgfile))/255

        h,w = img.shape[0], img.shape[1]
        if imageSize == 0:
            imageSize = 250
            while imageSize+250 < h and imageSize+250 < w:
                imageSize += 250
            print('Set imageSize to',imageSize)


        # img_norm = (img - np.min(img)) / (np.max(img) - np.min(img)) 
        
        if equal_partition:
            srcimg = np.zeros((imageSize*(h // imageSize),imageSize*(w // imageSize)))
            oimg = np.zeros((imageSize*(h // imageSize),imageSize*(w // imageSize)))
            woffset = int((w % imageSize)/2)
            hoffset = int((h % imageSize)/2)

            for row in range(int(h/imageSize)):
                for col in range(int(w/imageSize)):
                    
                    sub_img = img[hoffset + imageSize*row:hoffset + imageSize*row+imageSize,woffset + imageSize*col:woffset + imageSize*col+imageSize]
                    sub_img = (sub_img - np.min(sub_img)) / (np.max(sub_img) - np.min(sub_img))
                    sub_img2 = img_norm[hoffset + imageSize*row:hoffset + imageSize*row+imageSize,woffset + imageSize*col:woffset + imageSize*col+imageSize]

                    # sub_img = (sub_img + sub_img2)/2
                    sub_img = sub_img

                    pil_sub_img = Image.fromarray((sub_img*255).astype('uint8'))
                    sub_tensor = toTensor(pil_sub_img)
                    print('shape is',sub_tensor.shape)
                    sub_tensor = sub_tensor.unsqueeze(0)

                    with torch.no_grad():
                        if opt.cpu:
                            sr = net(sub_tensor)
                        else:
                            sr = net(sub_tensor.cuda())
                        sr = sr.cpu()
                        sr = torch.clamp(sr,min=0,max=1)

                        m = nn.LogSoftmax(dim=0)
                        sr = m(sr[0])
                        sr = sr.argmax(dim=0, keepdim=True)

                        oimg[row*imageSize:(row+1)*imageSize,col*imageSize:(col+1)*imageSize] = sr
                        srcimg[row*imageSize:(row+1)*imageSize,col*imageSize:(col+1)*imageSize] = np.array(sub_img)
                    # sr = toPIL(sr.float())


            Image.fromarray((oimg*255).astype('uint8')).save(opt.out + '/%04d.png' % i)
            # Image.fromarray((srcimg*255).astype('uint8')).save(opt.out + '/imageinput_' + str(i) + '.png')
            plt.figure()
            plt.subplot(121)
            plt.imshow(oimg,cmap='gray')
            plt.subplot(122)
            plt.imshow(srcimg,cmap='gray')
            plt.show()
        else:
            
            for j in range(len(bat)):
                img = bat.data[j]

                imgtensor = img[:,:imageSize,:imageSize]
                imgtensor.unsqueeze(0)
                torch.cat((imgtensor,img[:,h-imageSize:,:imageSize]),0)
                torch.cat((imgtensor,img[:,:imageSize,w-imageSize:]),0)
                torch.cat((imgtensor,img[:,h-imageSize:,w-imageSize:]),0)
            

            proc_images = []
            
            for idx,sub_img in enumerate(images):
                # sub_img = (sub_img - np.min(sub_img)) / (np.max(sub_img) - np.min(sub_img))
                pil_sub_img = Image.fromarray((sub_img*255).astype('uint8'))
                sub_tensor = toTensor(pil_sub_img)
                print('\r[%d/%d], shape is %dx%d - ' % (i+1,len(imgs),sub_tensor.shape[1],sub_tensor.shape[2]),end='')
                sub_tensor = sub_tensor.unsqueeze(0)

                with torch.no_grad():
                    if opt.cpu:
                        sr = net(sub_tensor)
                    else:
                        sr = net(sub_tensor.cuda())
                    sr = sr.cpu()
                    sr = torch.clamp(sr,min=0,max=1)

                    m = nn.LogSoftmax(dim=0)
                    sr = m(sr[0])
                    sr = sr.argmax(dim=0, keepdim=True)
                    

                    pil_sr_img = toPIL(sr.float())                    
                    # pil_sr_img.save(opt.out + '/segmeneted_output_' + str(i) + '_' + str(idx) + '.png')
                    # pil_sub_img.save(opt.out + '/imageinput_' + str(i) + '_' + str(idx) + '.png')

                    proc_images.append(pil_sr_img)
                

            img1 = proc_images[0]
            img2 = proc_images[1]
            img3 = proc_images[2]
            img4 = proc_images[3]

            woffset = (2*imageSize-w) // 2
            hoffset = (2*imageSize-h) // 2

            img1 = np.array(img1)[:imageSize-hoffset,:imageSize-woffset]
            img3 = np.array(img3)[:imageSize-hoffset,woffset:]
            top = np.concatenate((img1,img3),axis=1)

            img2 = np.array(img2)[hoffset:,:imageSize-woffset]
            img4 = np.array(img4)[hoffset:,woffset:]
            bot = np.concatenate((img2,img4),axis=1)

            supersized_img = np.concatenate((top,bot),axis=0)
            # supersized_img = supersized_img[10:-10,10:-10]
            # img = img[10:-10,10:-10]
            print(imgfile,i)
            Image.fromarray(supersized_img).save('%s/%04d.png' % (opt.out,i))
            # Image.fromarray((img*255).astype('uint8')).save('%s/input_%04d.png' % (opt.out,i))


if __name__ == '__main__':
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

        
    EvaluateModel(opt)






            



