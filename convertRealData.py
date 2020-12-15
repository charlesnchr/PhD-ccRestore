import javabridge
import bioformats
import matplotlib.pyplot as plt
import numpy as np
import glob
import os
from skimage import io

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--root', type=str, default="E:/datasets/Lisa/firstimages/Lisa/11-0_561.tif", help='raw input')
parser.add_argument('--out', type=str, default="athe/test.tif", help='out folder')
parser.add_argument('--nFrames', type=int, default=9, help='desired SIM frames') 

opt = parser.parse_args()

javabridge.start_vm(class_path=bioformats.JARS)

os.makedirs(opt.out,exist_ok=True)

files = glob.glob(opt.root)

for file in files:

    print('PROCESSING file',file)
   
    md = bioformats.get_omexml_metadata(file)
    o = bioformats.OMEXML(md)
    p = o.image().Pixels

    rdr = bioformats.ImageReader(file, perform_init=True)

    # order = 'XYZCT'
    dtype = np.float16 if p.PixelType == 'uint16' else np.float32

    nX = p.SizeX
    nY = p.SizeY
    nC = p.SizeC
    nZ = p.SizeZ
    nT = p.SizeT

    # NCHW
    I = np.zeros((opt.nFrames+3,nX,nY),dtype='uint16')

    for t in range(opt.nFrames):
        frame = rdr.read(t=opt.nFrames-1-t,rescale=False)
        frame = 120 / np.max(frame) * frame
        frame = np.rot90(frame)
        I[t,:,:] = frame


    # wf
    wf = np.mean(I[:opt.nFrames,:,:],axis=0)
    I[opt.nFrames+0,:,:] = wf
    I[opt.nFrames+1,:,:] = wf
    I[opt.nFrames+2,:,:] = wf


    print('SAVING image',I.shape)

    filename = os.path.basename(file)
    ext = os.path.splitext(filename)[1]
    filename = filename.replace(ext,'.tif')
    io.imsave(opt.out + '/' + filename,I)
 
    print('Exported %s' % opt.out)


    meta = {}
    # print(dir(o.image().Pixels))

    # print(img.shape)
    # plt.figure()
    # plt.imshow(img)
    # plt.show()

javabridge.kill_vm()