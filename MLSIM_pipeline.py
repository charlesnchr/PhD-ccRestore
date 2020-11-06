import sys
import numpy as np
from numpy import pi, cos, sin
import math
from skimage import io, transform
import glob
import os
import argparse
from multiprocessing import Pool
import subprocess
import MLSIM_datagen.SIMulator_functions
import run
import shutil

# ------------ Options --------------
from options import parser
parser.add_argument('--sourceimages_path', type=str, default='/local/scratch/cnc39/phd/datasets/DIV2K/DIV2K_train_HR')
parser.add_argument('--nrep', type=int, default=1, help='instances of same source image')
parser.add_argument('--datagen_workers', type=int, default=8, help='')
parser.add_argument('--ext', nargs='+', default=['png'], choices=['png','jpg','tif'])

# SIM options to control from command line
parser.add_argument('--Nshifts', type=int, default=3)
parser.add_argument('--Nangles', type=int, default=3)
parser.add_argument('--k2', type=float, default=126.0)
parser.add_argument('--k2_err', type=float, default=30.0)
parser.add_argument('--usePSF', type=int, default=0)


opt = parser.parse_args()
    # return opt


# ------------ Parameters-------------
def GetParams(): # uniform randomisation
    SIMopt = argparse.Namespace()

    # phase shifts for each stripe
    SIMopt.Nshifts = opt.Nshifts
    # number of orientations of stripes
    SIMopt.Nangles = opt.Nangles
    # used to adjust PSF/OTF width
    SIMopt.scale = 0.9 + 0.1*(np.random.rand()-0.5)
    # modulation factor
    SIMopt.ModFac = 0.8 + 0.3*(np.random.rand()-0.5)
    # orientation offset
    SIMopt.alpha = pi/3*(np.random.rand()-0.5)
    # orientation error
    SIMopt.angleError = 10*pi/180*(np.random.rand()-0.5)
    # shuffle the order of orientations
    SIMopt.shuffleOrientations = True
    # random phase shift errors
    SIMopt.phaseError = 1*pi*(0.5-np.random.rand(SIMopt.Nangles, SIMopt.Nshifts))
    # mean illumination intensity
    SIMopt.meanInten = np.ones(SIMopt.Nangles)*0.5
    # amplitude of illumination intensity above mean
    SIMopt.ampInten = np.ones(SIMopt.Nangles)*0.5*SIMopt.ModFac
    # illumination freq
    SIMopt.k2 = opt.k2 + opt.k2_err*(np.random.rand()-0.5)
    # in percentage
    SIMopt.NoiseLevel = 8 + 0*8*(np.random.rand()-0.5)
    # 1(to blur using PSF), 0(to blur using OTF)
    SIMopt.UsePSF = opt.usePSF
    # include OTF and GT in stack
    SIMopt.OTF_and_GT = True

    return SIMopt



# ------------ Main loop --------------
def processImage(file):
    Io = io.imread(file) / 255
    Io = transform.resize(Io, (opt.imageSize, opt.imageSize), anti_aliasing=True)

    if len(Io.shape) > 2 and Io.shape[2] > 1:
        Io = Io.mean(2)  # if not grayscale

    filename = os.path.basename(file).replace('.png', '')

    print('Generating SIM frames for', file)

    for n in range(opt.nrep):
        SIMopt = GetParams()
        SIMopt.outputname = '%s/%s_%d.tif' % (opt.root, filename, n)
        I = MLSIM_datagen.SIMulator_functions.Generate_SIM_Image(SIMopt, Io)


if __name__ == '__main__':

    os.makedirs(opt.root, exist_ok=True)
    os.makedirs(opt.out, exist_ok=True)
    
    shutil.copy2('MLSIM_datagen/SIMulator.py',opt.out)
    shutil.copy2('MLSIM_datagen/SIMulator_functions.py',opt.out)

    files = []
    for ext in opt.ext:
        files.extend(glob.glob(opt.sourceimages_path + "/*." + ext))

    if len(files) == 0:
        print('source images not found')
        sys.exit(0)
    elif opt.ntrain + opt.ntest > opt.nrep*len(files):
        print('ntrain + opt.ntest is too high given nrep and number of source images')
        sys.exit(0)
    elif opt.nch_in > opt.Nangles*opt.Nshifts:
        print('nch_in cannot be greater than Nangles*Nshifts - not enough SIM frames')
        sys.exit(0)
    
    files = files[:math.ceil( (opt.ntrain + opt.ntest) / opt.nrep )]

    with Pool(opt.datagen_workers) as p:
        p.map(processImage,files)


    print('Done generating images,',opt.root)

    print('Now starting training:\n')

    # cmd = '\npython run.py ' + ' '.join(sys.argv[:])
    # print(cmd,end='\n\n')
    # subprocess.Popen(cmd,shell=True)
    run.main(opt)
    