import os
import math

import torch
import time 

import torch.optim as optim
import torchvision
from torch.autograd import Variable

from models import *
from datahandler import *

from skimage import io,exposure,img_as_ubyte
import matplotlib.pyplot as plt
import glob
from tqdm import tqdm


# from plotting import testAndMakeCombinedPlots


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


def changeColour(I): # change colours (used to match WEKA output, request by Meng)
    Inew = np.zeros(I.shape + (3,)).astype('uint8')
    for rowidx in range(I.shape[0]):
        for colidx in range(I.shape[1]):
            if I[rowidx][colidx] == 0:
                Inew[rowidx][colidx] = [198,118,255]
            elif I[rowidx][colidx] == 127:
                Inew[rowidx][colidx] = [79,255,130]
            elif I[rowidx][colidx] == 255:
                Inew[rowidx][colidx] = [255,0,0]
    return Inew


def processImage(net,opt,idxstr,imgfile,img):

    imageSize = opt.imageSize

    h,w = img.shape[0], img.shape[1]
    if imageSize == 0:
        imageSize = 250
        while imageSize+250 < h and imageSize+250 < w:
            imageSize += 250
        print('Set imageSize to',imageSize)

    # img_norm = (img - np.min(img)) / (np.max(img) - np.min(img)) 
    images = []

    images.append(img[:imageSize,:imageSize])
    images.append(img[h-imageSize:,:imageSize])
    images.append(img[:imageSize,w-imageSize:])
    images.append(img[h-imageSize:,w-imageSize:])

    proc_images = []
    for idx,sub_img in enumerate(tqdm(images,leave=False)):
        # sub_img = (sub_img - np.min(sub_img)) / (np.max(sub_img) - np.min(sub_img))
        pil_sub_img = Image.fromarray((sub_img*255).astype('uint8'))
        sub_tensor = toTensor(pil_sub_img)
        sub_tensor = sub_tensor.unsqueeze(0)

        with torch.no_grad():
            if opt.cpu:
                sr = net(sub_tensor)
            else:
                sr = net(sub_tensor.cuda())
            sr = sr.cpu()
            # sr = torch.clamp(sr,min=0,max=1)

            m = nn.LogSoftmax(dim=0)
            sr = m(sr[0])
            sr = sr.argmax(dim=0, keepdim=True)
            

            pil_sr_img = toPIL(sr.float() / (opt.nch_out - 1))
            # pil_sr_img.save(opt.out + '/segmeneted_output_' + str(i) + '_' + str(idx) + '.png')
            # pil_sub_img.save(opt.out + '/imageinput_' + str(i) + '_' + str(idx) + '.png')

            proc_images.append(pil_sr_img)
        
    # stitch together
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

    oimg = np.concatenate((top,bot),axis=0)
    # crop?
    # oimg = oimg[10:-10,10:-10]
    # img = img[10:-10,10:-10]
    # remove boundaries? 
    # oimg[:10,:] = 0
    # oimg[-10:,:] = 0
    # oimg[:,:10] = 0
    # oimg[:,-10:] = 0

    if opt.stats_tubule_sheet:
        fraction1 = np.sum(oimg == 255) # tubule
        fraction2 = np.sum(oimg == 127) # sheet
        npix = w*h
        opt.csvfid.write('%s:%s,%0.4f,%0.4f,%0.4f\n' % (os.path.basename(imgfile),idxstr,fraction1/npix,fraction2/npix,fraction1/fraction2))
    if opt.weka_colours:
        oimg = changeColour(oimg)

    if opt.out == 'root': # save next to orignal
        ext = imgfile.split('.')[-1]
        Image.fromarray(oimg).save(imgfile.replace('.' + ext,'_out_' + idxstr + '.png'))
        if opt.save_input:
            io.imsave(imgfile.replace('.' + ext,'_in_' + idxstr + '.png'),img_as_ubyte(img))
    else:
        Image.fromarray(oimg).save('%s/%s.png' % (opt.out,idxstr))
        if opt.save_input:
            io.imsave('%s/%s_in.png' % (opt.out,idxstr),img_as_ubyte(img))
    # Image.fromarray((img*255).astype('uint8')).save('%s/input_%04d.png' % (opt.out,i))



def EvaluateModel(opt):

    try:
        os.makedirs(opt.out)
    except IOError:
        pass

    if opt.stats_tubule_sheet:
        if opt.out == 'root':
            if opt.root.split('.')[-1].lower() in ['jpg','png','tif']:
                pardir = os.path.abspath(os.path.join(opt.root,os.pardir))
                opt.csvfid = open('%s/stats_tubule_sheet.csv' % pardir,'w')
            else:
                opt.csvfid = open('%s/stats_tubule_sheet.csv' % opt.root,'w')
        else:
            opt.csvfid = open('%s/stats_tubule_sheet.csv' % opt.out,'w')
        opt.csvfid.write('Filename,Tubule fraction,Sheet fraction,Tubule/sheet ratio\n')

    opt.fid = open(opt.out + '/log.txt','w')
    print(opt)
    print(opt,'\n',file=opt.fid)
    
    net = GetModel(opt)

    checkpoint = torch.load(opt.weights)
    if opt.cpu:
        net.cpu()
    print('loading checkpoint',opt.weights)
    if opt.undomulti:
        checkpoint['state_dict'] = remove_dataparallel_wrapper(checkpoint['state_dict'])
    net.load_state_dict(checkpoint['state_dict'])

    if opt.root.split('.')[-1].lower() in ['png','jpg','tif']:
        imgs = [opt.root]
    else:
        imgs = []
        for ext in opt.ext:
            # imgs.extend(glob.glob(opt.root + '/*.jpg')) # scan only folder
            if len(imgs) == 0: # scan everything
                imgs.extend(glob.glob(opt.root + '/**/*.%s' % ext,recursive=True))

    pbar_outer = tqdm(imgs)
    for imgidx, imgfile in enumerate(pbar_outer):

        pbar_outer.set_description(os.path.basename(imgfile))

        _, ext = os.path.splitext(imgfile)

        if ext.lower() == '.tif':
            img = io.imread(imgfile)
        else:
            img = np.array(Image.open(imgfile))/255

        # img = io.imread(imgfile)
        # img = (img - np.min(img)) / (np.max(img) - np.min(img)) 

        if len(img.shape) == 2:
            idxstr = '%04d' % imgidx
            processImage(net,opt,idxstr,imgfile,img)
        elif img.shape[2] <= 3:
            print('removing colour channel')
            img = img[:,:,0]
            idxstr = '%04d' % imgidx
            processImage(net,opt,idxstr,imgfile,img)
        else: # more than 3 channels, assuming stack
            for subimgidx in tqdm(range(img.shape[0]),leave=False,desc='Stack progress'):
                idxstr = '%04d_%04d' % (imgidx,subimgidx)
                p1,p99 = np.percentile(img[subimgidx],1),np.percentile(img[subimgidx],99)
                imgnorm = exposure.rescale_intensity(img[subimgidx],in_range=(p1,p99))
                processImage(net,opt,idxstr,imgfile,imgnorm)


    if opt.stats_tubule_sheet:
        opt.csvfid.close()


if __name__ == '__main__':
    import sys
    print('OPTIONS',sys.argv)
    from options import parser
    parser.add_argument('--ext', nargs='+', default=['jpg','png','tif'])
    parser.add_argument('--stats_tubule_sheet', action='store_true')
    parser.add_argument('--weka_colours', action='store_true')
    parser.add_argument('--save_input', action='store_true')
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






            



