import numpy as np
from numpy import pi, cos, sin

from skimage import io, transform
from SIMulator_functions import *
import glob
import os
import argparse

from multiprocessing import Pool


# ------------ Options --------------
nrep = 1
outdir = "SIMdata"
os.makedirs(outdir, exist_ok=True)

# for DIV2k
files = glob.glob("G:/My Drive/01datasets/0standard/DIV2K/DIV2K_train_HR/*.png")[:6]

# single test image
# files = glob.glob('TestImage.png')


# ------------ Parameters-------------
def GetParams():
    opt = argparse.Namespace()

    # number of speckels in each pattern
    opt.Nspeckles = 256*50
    # number of speckel patterns
    opt.Nframes  = 50
    # used to adjust PSF/OTF width
    opt.scale = 0.63 + 0.05*(np.random.rand()-0.5)
    # modulation factor
    opt.ModFac = 0.8 + 0.05*(np.random.rand()-0.5)
    # orientation offset
    opt.alpha = pi/3*(np.random.rand()-0.5)
    # orientation error
    opt.angleError = 10*pi/180*(np.random.rand()-0.5)
    # shuffle the order of orientations
    opt.shuffleOrientations = True
    # mean illumination intensity
    opt.meanInten = np.ones(opt.Nspeckles)*0.5
    # amplitude of illumination intensity above mean
    opt.ampInten = np.ones(opt.Nspeckles)*0.5*opt.ModFac
    # in percentage
    opt.NoiseLevel = 2 + 0*8*(np.random.rand()-0.5)
    # 1(to blur using PSF), 0(to blur using OTF)
    opt.UsePSF = 0
    # include OTF and GT in stack
    opt.OTF_and_GT = True

    return opt


# ------------ Main loop --------------
def processImage(file):
    Io = io.imread(file) / 255
    Io = transform.resize(Io, (256, 256), anti_aliasing=True)

    if len(Io.shape) > 2 and Io.shape[2] > 1:
        Io = Io.mean(2)  # if not grayscale

    filename = os.path.basename(file).replace('.png', '')

    print('Generating SIM frames for', file)

    for n in range(nrep):
        opt = GetParams()
        opt.outputname = '%s/%s_%d.tif' % (outdir, filename, n)
        I = Generate_SIM_Image(opt, Io)    

if __name__ == '__main__':
    with Pool(6) as p:
        p.map(processImage,files)
        
