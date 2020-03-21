import subprocess
import glob
import shutil
import numpy as np
from PIL import Image
import os

from skimage.restoration import (denoise_tv_chambolle, denoise_bilateral,
                                 denoise_wavelet, estimate_sigma)

import tifffile

from evaluate_large_image import EvaluateModel

from options import parser
parser.add_argument('--binfile', type=str, default='C:/Users/cnc39_admin/Dropbox/4.Software/ndsafir-2.2-win-32bits/bin/ndsafir.exe')
parser.add_argument('--region', type=str, default=None)
opt = parser.parse_args()

print(opt)


def SaveImageSequences(stackfilename,savepathname,cidx,maxnum=1000,imagestep=1):

    stackbasename = stackfilename.split('/')[-1].split('.')[0].replace(' ','')

    try:
        os.makedirs(savepathname + '/' + stackbasename + '/denoised')
    except Exception:
        pass

    metadata = tifffile.TiffFile(stackfilename).imagej_metadata
    ranges = metadata['Ranges']
    
    img = tifffile.imread(stackfilename)
    n,c,h,w = img.shape

    if h % 2 != 0: h -= 1
    if w % 2 != 0: w -= 1

    for citer in range(c):
        for i in range(0,n,imagestep):
            if i == maxnum: break

            minval,maxval = ranges[citer*2:citer*2+2]
            minval,maxval = round(minval),round(maxval)

            try:
                os.makedirs(savepathname + '/' + stackbasename + '/channel_%d' % citer)
            except Exception:
                pass
            
            newfilename = savepathname + '/' + stackbasename + '/channel_%d/%04d.jpg' % (citer,i)
            frame = img[i,citer,:h,:w]

            frame[frame > maxval] = maxval
            frame[frame < minval] = minval

            frame = (frame - minval) / (maxval - minval)

            frame = Image.fromarray((frame*255).astype('uint8'))
            frame.save(newfilename)

            # denoising
            if citer == cidx:
                denoised_filename = savepathname + '/' + stackbasename + '/denoised/%04d.jpg' % i
                
                cmd = " -i %s -o %s -iter 8 -nthreads 8 -patch 2,2,0,0,0 --pval 0.01 -3D 0 -time 0" % (newfilename,denoised_filename)
                print(cmd)
                subprocess.check_output(opt.binfile + cmd)
                
                # dn = denoise_tv_chambolle(np.array(frame), weight=0.05)
                # Image.fromarray((255*dn).astype('uint8')).save(denoised_filename)

                # frame.save(denoised_filename)


import cv2

def remove_isolated_pixels(image):
    connectivity = 8

    output = cv2.connectedComponentsWithStats(image, connectivity, cv2.CV_32S)

    num_stats = output[0]
    labels = output[1]
    stats = output[2]

    new_image = image.copy()

    for label in range(num_stats):
        if stats[label,cv2.CC_STAT_AREA] < 50:
            new_image[labels == label] = 0

    return new_image

def closed_image(image):
    kernel = np.ones((3,3),np.uint8)
    new_image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    return new_image


def AssembleStacks(basefolder,region=None):
    # export to tif
    import numpy as np
    from skimage import io
    import glob

    from skimage.morphology import skeletonize

    folders = glob.glob(basefolder + '/channel_*')
    folders.append(basefolder + '/denoised')
    folders.append(basefolder + '/output') # actual output
    folders.append(basefolder + '/output') # skeletonised output

    imgs = glob.glob(basefolder + '/denoised/*')
    n = len(imgs)
    c = len(folders)
    h,w = io.imread(imgs[0]).shape

    I = np.zeros((n,c,h,w),dtype='uint8')

    for cidx, folder in enumerate(folders):
        
        imgs = glob.glob(folder + '/*.jpg')
        imgs.extend(glob.glob(folder + '/*.png'))

        for nidx, imgfile in enumerate(imgs):
            img = io.imread(imgfile)

            if '/output' in folder: # remove region?
                if region is not None:
                    img[region[0]:region[1],region[2]:region[3]] = 0        
                img = remove_isolated_pixels(img)
            if cidx == len(folders)-1: # skeletonize
                img = skeletonize(img/255)*255
                
            I[nidx,cidx,:,:] = img

        print('[%d/%d] loaded imgs %s' % (cidx+1,len(folders),folder))

    I.shape = n,1,c,h,w,1
    io.imsave(basefolder + '/' + basefolder.split('/')[-1] + '.tif',I,imagej=True)    
    print('saved stack: %s.tif' % basefolder)


if __name__ == '__main__':

    if len(opt.basedir) > 0:
        opt.root = opt.root.replace('basedir',opt.basedir)
        opt.weights = opt.weights.replace('basedir',opt.basedir)
        opt.out = opt.out.replace('basedir',opt.basedir)

    if '.tif' not in opt.root:
        queue = glob.glob(opt.root + '/*.tif')
    else:
        queue = opt.root.split(',')

    initial_out = opt.out

    for stack in queue:
        print('\n----------------------------------------\nProcessing %s\n----------------------------------------' % stack)
        print('\nSAVING IMAGE SEQUENCES:')
        opt.out = initial_out
        stack = stack.replace('\\','/')
        SaveImageSequences(stack,opt.out,1)
        stackbasename = stack.split('/')[-1].split('.')[0].replace(' ','')
        basefolder = opt.out + '/' + stackbasename
        print(stackbasename)

        print('\nEVALUATING MODEL ON DATA')
        opt.root = basefolder + '/denoised'
        opt.out = basefolder + '/output'
        EvaluateModel(opt)

        print('\nASSEMBLING STACKS')
        if opt.region is not None:
            temp = opt.region.split(',')
            opt.region = []
            for idx in temp: opt.region.append(int(idx))
        AssembleStacks(basefolder,opt.region)
