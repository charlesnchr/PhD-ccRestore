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
import MLSIM_datagen.SeqSIMulator_functions
import run
import shutil
import wandb

# ------------ Options --------------
from options import parser

parser.add_argument(
    "--sourceimages_path",
    type=str,
    default="/local/scratch/cnc39/phd/datasets/DIV2K/DIV2K_train_HR",
)
parser.add_argument(
    "--params",
    type=str,
    default="GetParams",
)
parser.add_argument(
    "--nrep", type=int, default=1, help="instances of same source image"
)
parser.add_argument("--datagen_workers", type=int, default=8, help="")
parser.add_argument("--ext", nargs="+", default=["png"])


## SIM options to control from command line

# stripes
parser.add_argument("--Nshifts", type=int, default=3)
parser.add_argument("--Nangles", type=int, default=3)
parser.add_argument("--k2", type=float, default=130.0)
parser.add_argument("--k2_err", type=float, default=15.0)

# spots
parser.add_argument("--Nspots", type=int, default=5)
parser.add_argument("--spotSize", type=int, default=1)

parser.add_argument("--PSFOTFscale", type=float, default=0.6)
parser.add_argument("--ModFac", type=float, default=0.5)
parser.add_argument("--usePSF", type=int, default=0)
parser.add_argument("--NoiseLevel", type=float, default=8)
parser.add_argument("--NoiseLevelRandFac", type=float, default=8)
parser.add_argument(
    "--phaseErrorFac", type=float, default=0.15
)  # pi/3 quite large but still feasible
parser.add_argument(
    "--alphaErrorFac", type=float, default=0.15
)  # pi/3 quite large but still feasible
parser.add_argument(
    "--angleError", type=float, default=5
)  # pi/3 quite large but still feasible
parser.add_argument("--usePoissonNoise", action="store_true")
parser.add_argument("--dontShuffleOrientations", action="store_true")
parser.add_argument("--dataonly", action="store_true")
parser.add_argument("--applyOTFtoGT", action="store_true")
parser.add_argument("--noStripes", action="store_true")
parser.add_argument("--seqSIM", action="store_true")
parser.add_argument("--skip_datagen", action="store_true")
parser.add_argument(
    "--SIMmodality",
    type=str,
    default="stripes",
    help="SIM modality",
    choices=["stripes", "spots", "speckle"],
)
parser.add_argument(
    "--patterns", action="store_true", help="Only illumination patterns"
)

# DMD options
parser.add_argument(
    "--crop_factor", action="store_true", help="Crop factor for DMD coordinates"
)
parser.add_argument(
    "--dmdMapping",
    default=0,
    type=int,
    help="""whether to map image to DMD coordinates. 0:
                    no mapping (tilted coordinate system if using oblique DMD), 1: DMD coordinate
                    transformation for acquisition, 2: predicted physical appeareance on DMD for modelling""",
    choices=[0, 1, 2],
)


opt = parser.parse_args()

if opt.root == "auto":
    opt.root = opt.out + "_SIMdata"

np.random.seed(20221219)


def GetParams_20230625():  # uniform randomisation
    SIMopt = argparse.Namespace()

    # modulation factor
    SIMopt.ModFac = 0.3 + 0.3 * (np.random.rand() - 0.5)

    # ---- stripes
    # phase shifts for each stripe
    SIMopt.Nshifts = opt.Nshifts
    # number of orientations of stripes
    SIMopt.Nangles = opt.Nangles
    # orientation offset
    SIMopt.alpha = opt.alphaErrorFac * pi * (np.random.rand() - 0.5)
    # orientation error
    SIMopt.angleError = opt.angleError * pi / 180 * (np.random.rand() - 0.5)
    # shuffle the order of orientations
    SIMopt.shuffleOrientations = not opt.dontShuffleOrientations
    # random phase shift errors
    SIMopt.phaseError = (
        opt.phaseErrorFac * pi * (0.5 - np.random.rand(SIMopt.Nangles, SIMopt.Nshifts))
    )
    # illumination freq
    SIMopt.k2 = opt.k2 + opt.k2_err * (np.random.rand() - 0.5)

    # --- spots
    SIMopt.Nspots = opt.Nspots
    SIMopt.spotSize = opt.spotSize

    # used to adjust PSF/OTF width
    SIMopt.PSFOTFscale = 0.7 + 0.2 * (np.random.rand() - 0.5)
    # noise type
    SIMopt.usePoissonNoise = opt.usePoissonNoise
    # noise level (percentage for Gaussian)
    SIMopt.NoiseLevel = opt.NoiseLevel + opt.NoiseLevelRandFac * (
        np.random.rand() - 0.5
    )
    # 1(to blur using PSF), 0(to blur using OTF)
    SIMopt.UsePSF = opt.usePSF
    # include OTF and GT in stack
    SIMopt.OTF_and_GT = True
    # use a blurred target (according to theoretical optimal construction)
    SIMopt.applyOTFtoGT = opt.applyOTFtoGT
    # whether to simulate images using just widefield illumination
    SIMopt.noStripes = opt.noStripes

    # function to use for stripes
    SIMopt.func = np.cos

    SIMopt.patterns = opt.patterns
    SIMopt.crop_factor = opt.crop_factor
    SIMopt.SIMmodality = opt.SIMmodality
    SIMopt.dmdMapping = opt.dmdMapping

    # --- Nframes
    if SIMopt.SIMmodality == "stripes":
        SIMopt.Nframes = SIMopt.Nangles * SIMopt.Nshifts
        # mean illumination intensity
        SIMopt.meanInten = np.ones(SIMopt.Nangles)
        # amplitude of illumination intensity above mean
        SIMopt.ampInten = np.ones(SIMopt.Nangles) * SIMopt.ModFac
    else:
        SIMopt.Nframes = (SIMopt.Nspots // SIMopt.spotSize) ** 2
        # amplitude of illumination intensity above mean
        SIMopt.ampInten = SIMopt.ModFac
        # mean illumination intensity
        SIMopt.meanInten = 1 - SIMopt.ampInten
        # resize amount of spots (to imitate effect of cropping from FOV on DMD/camera sensor)
        SIMopt.spotResize = 0.7 + 0.6 * (np.random.rand() - 0.5)

    SIMopt.imageSize = opt.imageSize

    return SIMopt


def GetParams_20230410():  # uniform randomisation
    SIMopt = argparse.Namespace()

    # phase shifts for each stripe
    SIMopt.Nshifts = opt.Nshifts
    # number of orientations of stripes
    SIMopt.Nangles = opt.Nangles
    # used to adjust PSF/OTF width
    SIMopt.PSFOTFscale = opt.PSFOTFscale + 0.1 * (np.random.rand() - 0.5)
    # modulation factor
    SIMopt.ModFac = opt.ModFac + 0.3 * (np.random.rand() - 0.5)
    # orientation offset
    SIMopt.alpha = opt.alphaErrorFac * pi * (np.random.rand() - 0.5)
    # orientation error
    SIMopt.angleError = opt.angleError * pi / 180 * (np.random.rand() - 0.5)
    # shuffle the order of orientations
    SIMopt.shuffleOrientations = not opt.dontShuffleOrientations
    # random phase shift errors
    SIMopt.phaseError = (
        opt.phaseErrorFac * pi * (0.5 - np.random.rand(SIMopt.Nangles, SIMopt.Nshifts))
    )
    # mean illumination intensity
    SIMopt.meanInten = np.ones(SIMopt.Nangles) * 0.5
    # amplitude of illumination intensity above mean
    SIMopt.ampInten = np.ones(SIMopt.Nangles) * 0.5 * SIMopt.ModFac
    # illumination freq
    SIMopt.k2 = opt.k2 + opt.k2_err * (np.random.rand() - 0.5)
    # noise type
    SIMopt.usePoissonNoise = opt.usePoissonNoise
    # noise level (percentage for Gaussian)
    SIMopt.NoiseLevel = opt.NoiseLevel + opt.NoiseLevelRandFac * (
        np.random.rand() - 0.5
    )
    # 1(to blur using PSF), 0(to blur using OTF)
    SIMopt.UsePSF = opt.usePSF
    # include OTF and GT in stack
    SIMopt.OTF_and_GT = True
    # use a blurred target (according to theoretical optimal construction)
    SIMopt.applyOTFtoGT = opt.applyOTFtoGT
    # whether to simulate images using just widefield illumination
    SIMopt.noStripes = opt.noStripes

    # function to use for stripes
    SIMopt.func = (
        np.cos
        if np.random.rand() < 0.5
        else MLSIM_datagen.SIMulator_functions.square_wave_one_third
    )

    return SIMopt


# ------------ Parameters-------------
def GetParams():  # uniform randomisation
    SIMopt = argparse.Namespace()

    # modulation factor
    SIMopt.ModFac = opt.ModFac  # + 0.3*(np.random.rand()-0.5)

    # ---- stripes
    # phase shifts for each stripe
    SIMopt.Nshifts = opt.Nshifts
    # number of orientations of stripes
    SIMopt.Nangles = opt.Nangles
    # orientation offset
    SIMopt.alpha = opt.alphaErrorFac * pi * (np.random.rand() - 0.5)
    # orientation error
    SIMopt.angleError = opt.angleError * pi / 180 * (np.random.rand() - 0.5)
    # shuffle the order of orientations
    SIMopt.shuffleOrientations = not opt.dontShuffleOrientations
    # random phase shift errors
    SIMopt.phaseError = (
        opt.phaseErrorFac * pi * (0.5 - np.random.rand(SIMopt.Nangles, SIMopt.Nshifts))
    )
    # illumination freq
    SIMopt.k2 = opt.k2 + opt.k2_err * (np.random.rand() - 0.5)

    # --- spots
    SIMopt.Nspots = opt.Nspots
    SIMopt.spotSize = opt.spotSize

    # used to adjust PSF/OTF width
    SIMopt.PSFOTFscale = opt.PSFOTFscale  # + 0.1*(np.random.rand()-0.5)
    # noise type
    SIMopt.usePoissonNoise = opt.usePoissonNoise
    # noise level (percentage for Gaussian)
    SIMopt.NoiseLevel = opt.NoiseLevel + opt.NoiseLevelRandFac * (
        np.random.rand() - 0.5
    )
    # 1(to blur using PSF), 0(to blur using OTF)
    SIMopt.UsePSF = opt.usePSF
    # include OTF and GT in stack
    SIMopt.OTF_and_GT = True
    # use a blurred target (according to theoretical optimal construction)
    SIMopt.applyOTFtoGT = opt.applyOTFtoGT
    # whether to simulate images using just widefield illumination
    SIMopt.noStripes = opt.noStripes

    # function to use for stripes
    SIMopt.func = np.cos

    SIMopt.patterns = opt.patterns
    SIMopt.crop_factor = opt.crop_factor
    SIMopt.SIMmodality = opt.SIMmodality
    SIMopt.dmdMapping = opt.dmdMapping

    # --- Nframes
    if SIMopt.SIMmodality == "stripes":
        SIMopt.Nframes = SIMopt.Nangles * SIMopt.Nshifts
        # mean illumination intensity
        SIMopt.meanInten = np.ones(SIMopt.Nangles)
        # amplitude of illumination intensity above mean
        SIMopt.ampInten = np.ones(SIMopt.Nangles) * SIMopt.ModFac
    else:
        SIMopt.Nframes = (SIMopt.Nspots // SIMopt.spotSize) ** 2
        # amplitude of illumination intensity above mean
        SIMopt.ampInten = SIMopt.ModFac
        # mean illumination intensity
        SIMopt.meanInten = 1 - SIMopt.ampInten
        # resize amount of spots (to imitate effect of cropping from FOV on DMD/camera sensor)
        SIMopt.spotResize = 1

    SIMopt.imageSize = opt.imageSize

    return SIMopt


# ------------ Main loop --------------
def processImage(file, SIMopt_override=None):
    if "npy" in opt.ext:
        Io = np.load(file, allow_pickle=True) / 255
        filename = os.path.basename(file).replace(".npy", "")

        if len(Io.shape) > 2 and Io.shape[2] > 3:
            Io = Io[:, :, 8]  # assuming t-stack
        elif Io.shape[2] > 1:
            Io = Io.mean(2)  # if not grayscale
    else:
        Io = io.imread(file) / 255
        # Io = transform.resize(Io, (opt.imageSize, opt.imageSize), anti_aliasing=True)

        if len(Io.shape) > 2 and Io.shape[2] > 1:
            Io = Io.mean(2)  # if not grayscale

        filename = os.path.basename(file).replace(".png", "")

    print("Generating SIM frames for", file)

    gt_dim = opt.imageSize
    if type(gt_dim) is int:
        gt_dim = (gt_dim, gt_dim)

    # multiple by opt.scale
    gt_dim = [int(x * opt.scale) for x in gt_dim]

    for n in range(opt.nrep):
        if SIMopt_override is None:
            SIMopt = eval("%s()" % opt.params)  # GetParams
        else:
            SIMopt = SIMopt_override

        SIMopt.outputname = "%s/%s_%d.tif" % (opt.root, filename, n)

        I = MLSIM_datagen.SIMulator_functions.Generate_SIM_Image(
            SIMopt, Io, opt.imageSize, gt_dim, func=SIMopt.func
        )

    return I


# ------------ Main loop --------------
def processSeqImage(file):
    if "npy" in opt.ext:
        Io = np.load(file, allow_pickle=True) / 255
    else:
        Io = io.imread(file).transpose(1, 2, 0) / 255
    # Io = transform.resize(Io, (256, 256), anti_aliasing=True)

    # if len(Io.shape) > 2 and Io.shape[2] > 1:
    #     Io = Io.mean(2)  # if not grayscale

    filename = os.path.basename(file).replace(".npy", "")

    print("Generating SIM frames for", file)

    gt_dim = opt.imageSize
    if type(gt_dim) is int:
        gt_dim = (gt_dim, gt_dim)

    # multiple by opt.scale
    gt_dim = [int(x * opt.scale) for x in gt_dim]

    for n in range(opt.nrep):
        SIMopt = eval("%s()" % opt.params)
        SIMopt.outputname = "%s/%s_%d.tif" % (opt.root, filename, n)
        I = MLSIM_datagen.SeqSIMulator_functions.Generate_SIM_Image(
            SIMopt, Io, opt.imageSize, gt_dim
        )


def processSeqImageFolder(filepath):
    # Io = np.load(file, allow_pickle=True) / 255
    Io = []
    # for i in range(9):
    # im = io.imread('%s/%02d.jpg' % (filepath,(i+1))) / 255
    # im = im.mean(axis=2)
    # Io.append(im)

    for file in sorted(glob.glob("%s/*" % filepath)):
        im = io.imread(file) / 255
        if len(im.shape) > 2:
            im = im.mean(axis=2)
        Io.append(im)

    Io = np.array(Io).transpose(1, 2, 0) / 255
    # Io = transform.resize(Io, (256, 256), anti_aliasing=True)

    # if len(Io.shape) > 2 and Io.shape[2] > 1:
    #     Io = Io.mean(2)  # if not grayscale
    filename = os.path.basename(filepath)
    pardir = os.path.basename(os.path.abspath(os.path.join(filepath, os.pardir)))

    print("Generating SIM frames for", filepath)

    gt_dim = opt.imageSize
    if type(gt_dim) is int:
        gt_dim = (gt_dim, gt_dim)

    # multiple by opt.scale
    gt_dim = [int(x * opt.scale) for x in gt_dim]

    for n in range(opt.nrep):
        SIMopt = eval("%s()" % opt.params)
        SIMopt.outputname = "%s/%s_%s_%d.tif" % (opt.root, pardir, filename, n)
        I = MLSIM_datagen.SeqSIMulator_functions.Generate_SIM_Image(
            SIMopt, Io, opt.imageSize, gt_dim
        )


if __name__ == "__main__":
    print(opt)

    if not opt.skip_datagen:
        os.makedirs(opt.root, exist_ok=True)
        os.makedirs(opt.out, exist_ok=True)

        shutil.copy2("MLSIM_pipeline.py", opt.out)
        shutil.copy2("MLSIM_datagen/SIMulator_functions.py", opt.out)

        files = []
        if "imagefolder" not in opt.ext:
            for ext in opt.ext:
                files.extend(sorted(glob.glob(opt.sourceimages_path + "/*." + ext)))
        else:
            print("looking in opt", opt.sourceimages_path)
            folders = glob.glob("%s/*" % opt.sourceimages_path)
            for folder in folders:
                subfolders = glob.glob("%s/*" % folder)
                if len(subfolders) > 0:
                    if subfolders[0].endswith((".jpg", ".png")):
                        files.extend(folders)
                        break
                    files.extend(subfolders)

        if len(files) == 0:
            print("source images not found")
            sys.exit(0)
        elif (
            opt.ntrain + opt.ntest > opt.nrep * len(files)
            and opt.ntrain + opt.ntest > 0
        ):
            print(
                "ntrain + opt.ntest is too high given nrep and number of source images"
            )
            sys.exit(0)

        files = files[: math.ceil((opt.ntrain + opt.ntest) / opt.nrep)]

        if opt.ntrain + opt.ntest > 0:  # if == 0, use all
            files = files[: math.ceil((opt.ntrain + opt.ntest) / opt.nrep)]

        for file in files:
            print(file)

        with Pool(opt.datagen_workers) as p:
            if not opt.seqSIM:
                p.map(processImage, files)
            elif "imagefolder" not in opt.ext:
                p.map(processSeqImage, files)
            else:
                p.map(
                    processSeqImageFolder, files
                )  # processSeqImage if using tif files instead of folders of jpgs

        print("Done generating images,", opt.root)

    # cmd = '\npython run.py ' + ' '.join(sys.argv[:])
    # print(cmd,end='\n\n')
    # subprocess.Popen(cmd,shell=True)
    if not opt.dataonly:
        if not opt.disable_wandb:
            wandb.init(project="oni-sim")
            wandb.config.update(opt)
            opt.wandb = wandb

        print("Now starting training:\n")

        run.main(opt)
