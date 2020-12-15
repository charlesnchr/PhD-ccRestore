import torch
import matplotlib.pyplot as plt
import torchvision
import skimage
from skimage.measure import compare_ssim
import torchvision.transforms as transforms
import numpy as np
import time
from PIL import Image
import scipy.ndimage as ndimage
import torch.nn as nn
import os
from skimage import io,exposure,img_as_ubyte
import glob
import torchvision.transforms as transforms

toPIL = transforms.ToPILImage()
plt.switch_backend('agg')

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

def parse(model, opt, function, epoch):

    argin = opt.testFunctionArgs.split(',')
    args = {}
    for i in range(0,len(argin),2):
        val = argin[i+1]
        args[argin[i]] = val

    print('Running test script', function + '(model,opt)', 'with args',args)
    eval('%s(model,opt,args, epoch)' % function)

def experimentalER(model, opt, args, epoch, weka_colours=True):

    imSize = int(args['imageSize'])
    folders = sorted(glob.glob('%s/*' % opt.rootTesting))

    for fidx,folder in enumerate(folders):
        foldername = os.path.basename(folder)
        
        if foldername == 'ref': continue

        refimg = io.imread('%s/ref/%s.png' % (opt.rootTesting,foldername))
        images = sorted(glob.glob('%s/*.png' % folder))

        if(len(images) > 1):
            testImg = []
            for image in images:
                testImg.append(io.imread(image,as_gray=True))
            testImg = np.array(testImg)
        else: # only 1 image
            if opt.nch_in > 1:
                tmpimg = io.imread(images[0],as_gray=True)
                testImg = np.array([tmpimg,tmpimg,tmpimg])
            else:
                testImg = np.array([io.imread(images[0],as_gray=True)])
            
        # img format nhw
        h,w = testImg.shape[1],testImg.shape[2]
        subimgs = []
        subimgs.append(testImg[:,:imSize,:imSize])
        subimgs.append(testImg[:,h-imSize:,:imSize])
        subimgs.append(testImg[:,:imSize,w-imSize:])
        subimgs.append(testImg[:,h-imSize:,w-imSize:])


        # perform segmentation
        proc_images = []
        for idx,subimg in enumerate(subimgs):
            subimg = (subimg - np.min(subimg)) / (np.max(subimg) - np.min(subimg))
            sub_tensor = torch.tensor(subimg).float().unsqueeze(0)

            with torch.no_grad():
                if opt.cpu:
                    sr = model(sub_tensor)
                else:
                    sr = model(sub_tensor.cuda())
                sr = sr.cpu()
                # sr = torch.clamp(sr,min=0,max=1)

                m = nn.LogSoftmax(dim=0)
                sr = m(sr[0])
                sr = sr.argmax(dim=0, keepdim=True)
                
                pil_sr_img = toPIL(sr.float() / (opt.nch_out - 1))

                proc_images.append(pil_sr_img)        
        
        # stitch together
        img1 = proc_images[0]
        img2 = proc_images[1]
        img3 = proc_images[2]
        img4 = proc_images[3]

        woffset = (2*imSize-w) // 2
        hoffset = (2*imSize-h) // 2

        img1 = np.array(img1)[:imSize-hoffset,:imSize-woffset]
        img3 = np.array(img3)[:imSize-hoffset,woffset:]
        top = np.concatenate((img1,img3),axis=1)

        img2 = np.array(img2)[hoffset:,:imSize-woffset]
        img4 = np.array(img4)[hoffset:,woffset:]
        bot = np.concatenate((img2,img4),axis=1)

        oimg = np.concatenate((top,bot),axis=0)
    
        if weka_colours:
            oimg = changeColour(oimg)

        f, ax = plt.subplots(1,3,sharey=True,figsize=(30,10))
        ax[0].imshow(oimg)
        ax[1].imshow(refimg)
        ax[2].imshow(testImg[testImg.shape[0] // 2])
        plt.imshow(testImg[testImg.shape[0] // 2])
        plt.subplots_adjust(wspace=0.01, hspace=0.01)
        plt.savefig('%s/test_epoch%d_idx%d.jpg' % (opt.out,epoch,fidx), dpi=150, bbox_inches = 'tight', pad_inches = 0)
        plt.close()

