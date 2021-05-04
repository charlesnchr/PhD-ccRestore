import torch
import matplotlib.pyplot as plt
import torchvision
import skimage

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

def parse(model, opt, function, epoch):

    args = {}

    if opt.testFunctionArgs is not None:
        argin = opt.testFunctionArgs.split(',')
        for i in range(0,len(argin),2):
            val = argin[i+1]
            args[argin[i]] = val

    print('Running test script', function + '(model,opt)', 'with args',args)
    eval('%s(model,opt,args, epoch)' % function)


def experimentalSIMRec(model, opt, args, epoch):
    
    def prepimg(stack,self):

        inputimg = stack[:9]

        if self.nch_in == 6:
            inputimg = inputimg[[0,1,3,4,6,7]]
        elif self.nch_in == 3:
            inputimg = inputimg[[0,4,8]]

        if inputimg.shape[1] > 512 or inputimg.shape[2] > 512:
            print('Over 512x512! Cropping')
            inputimg = inputimg[:,:512,:512]

        inputimg = inputimg.astype('float') / np.max(inputimg) # used to be /255
        widefield = np.mean(inputimg,0) 

        if self.norm == 'adapthist':
            for i in range(len(inputimg)):
                inputimg[i] = exposure.equalize_adapthist(inputimg[i],clip_limit=0.001)
            widefield = exposure.equalize_adapthist(widefield,clip_limit=0.001)
            inputimg = torch.from_numpy(inputimg).float()
            widefield = torch.from_numpy(widefield).float()
        else:
            # normalise 
            inputimg = torch.from_numpy(inputimg).float()
            widefield = torch.from_numpy(widefield).float()
            widefield = (widefield - torch.min(widefield)) / (torch.max(widefield) - torch.min(widefield))

            if self.norm == 'minmax':
                for i in range(len(inputimg)):
                    inputimg[i] = (inputimg[i] - torch.min(inputimg[i])) / (torch.max(inputimg[i]) - torch.min(inputimg[i]))

        return inputimg,widefield


    testdir = 'experimentalTests'
    os.makedirs('%s/%s' % (opt.out,testdir),exist_ok=True)

    for iidx,imgfile in enumerate(glob.glob('%s/*.tif' % opt.rootTesting)):
        
        stack = io.imread(imgfile)
        
        inputimg, wf = prepimg(stack,opt)
        wf = (255*wf.numpy()).astype('uint8')

        with torch.no_grad():
            sr = model(inputimg.unsqueeze(0).to(opt.device))
            sr = sr.cpu()
            sr = torch.clamp(sr,min=0,max=1) 

        sr = sr.squeeze().numpy()
        sr = (255*sr).astype('uint8')
        if opt.norm == 'adapthist':
            sr = exposure.equalize_adapthist(sr,clip_limit=0.01)
        combined = np.concatenate((wf,sr),axis=1)
        skimage.io.imsave('%s/%s/test_epoch%d_idx%d.jpg' % (opt.out,testdir,epoch,iidx), combined) 



def experimentalER(model, opt, args, epoch, weka_colours=True):

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

