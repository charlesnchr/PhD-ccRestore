""" ----------------------------------------
* Creation Time : Mon Dec 19 14:41:48 2022
* Author : Charles N. Christensen
* Github : github.com/charlesnchr
----------------------------------------"""

import numpy as np
from numpy import pi, cos, sin
from numpy.fft import fft2, ifft2, fftshift, ifftshift

from skimage import io, transform
from scipy.signal import convolve2d
import scipy.special


def PsfOtf(w, scale):
    # AIM: To generate PSF and OTF using Bessel function
    # INPUT VARIABLES
    #   w: image size
    #   scale: a parameter used to adjust PSF/OTF width
    # OUTPUT VRAIBLES
    #   yyo: system PSF
    #   OTF2dc: system OTF
    eps = np.finfo(np.float64).eps

    x = np.linspace(0, w - 1, w)
    y = np.linspace(0, w - 1, w)
    X, Y = np.meshgrid(x, y)

    # Generation of the PSF with Besselj.
    R = np.sqrt(np.minimum(X, np.abs(X - w)) ** 2 + np.minimum(Y, np.abs(Y - w)) ** 2)
    yy = np.abs(2 * scipy.special.jv(1, scale * R + eps) / (scale * R + eps)) ** 2
    yy0 = fftshift(yy)

    # Generate 2D OTF.
    OTF2d = fft2(yy)
    OTF2dmax = np.max([np.abs(OTF2d)])
    OTF2d = OTF2d / OTF2dmax
    OTF2dc = np.abs(fftshift(OTF2d))

    return (yy0, OTF2dc)


def conv2(x, y, mode="same"):
    # Make it equivalent to Matlab's conv2 function
    # https://stackoverflow.com/questions/3731093/is-there-a-python-equivalent-of-matlabs-conv2-function
    return np.rot90(convolve2d(np.rot90(x, 2), np.rot90(y, 2), mode=mode), 2)


def SIMimage_patterns(opt, w, PSFo, OTFo):

    # AIM: to generate raw sim images
    # INPUT VARIABLES
    #   k2: illumination frequency
    #   DIo: specimen image
    #   PSFo: system PSF
    #   OTFo: system OTF
    #   UsePSF: 1 (to blur SIM images by convloving with PSF)
    #           0 (to blur SIM images by truncating its fourier content beyond OTF)
    #   NoiseLevel: percentage noise level for generating gaussian noise
    # OUTPUT VARIABLES
    #   frames:  raw sim images
    #   DIoTnoisy: noisy wide field image
    #   DIoT: noise-free wide field image

    wo = w / 2
    x = np.linspace(0, w - 1, w)
    y = np.linspace(0, w - 1, w)
    [X, Y] = np.meshgrid(x, y)

    # Illuminating pattern

    # orientation direction of illumination patterns
    orientation = np.zeros(opt.Nangles)
    for i in range(opt.Nangles):
        orientation[i] = i * pi / opt.Nangles + opt.alpha + opt.angleError

    if opt.shuffleOrientations:
        np.random.shuffle(orientation)

    # illumination frequency vectors
    k2mat = np.zeros((opt.Nangles, 2))
    for i in range(opt.Nangles):
        theta = orientation[i]
        k2mat[i, :] = (opt.k2 / w) * np.array([cos(theta), sin(theta)])

    # illumination phase shifts along directions with errors
    ps = np.zeros((opt.Nangles, opt.Nshifts))
    for i_a in range(opt.Nangles):
        for i_s in range(opt.Nshifts):
            ps[i_a, i_s] = 2 * pi * i_s / opt.Nshifts + opt.phaseError[i_a, i_s]

    # illumination patterns
    frames = []
    for i_a in range(opt.Nangles):
        for i_s in range(opt.Nshifts):
            # illuminated signal
            sig = opt.meanInten[i_a] + opt.ampInten[i_a] * cos(
                2 * pi * (k2mat[i_a, 0] * (X - wo) + k2mat[i_a, 1] * (Y - wo))
                + ps[i_a, i_s]
            )

            frames.append(sig)

    return frames


def ApplyOTF(opt, Io):
    w = Io.shape[0]
    psfGT, otfGT = PsfOtf(w, 1.8 * opt.scale)
    newGT = np.real(ifft2(fft2(Io) * fftshift(otfGT)))
    return newGT


# ------------ Options --------------
import argparse
import json

parser = argparse.ArgumentParser()

# SIM options to control from command line
parser.add_argument("--Nshifts", type=int, default=3)
parser.add_argument("--Nangles", type=int, default=3)
parser.add_argument("--k2", type=float, default=126.0)
parser.add_argument("--k2_err", type=float, default=30.0)
parser.add_argument("--PSFOTFscale", type=float, default=0.9)
parser.add_argument("--ModFac", type=float, default=0.8)
parser.add_argument("--usePSF", type=int, default=0)
parser.add_argument("--NoiseLevel", type=float, default=8)
parser.add_argument("--NoiseLevelRandFac", type=float, default=8)
parser.add_argument(
    "--phaseErrorFac", type=float, default=0
)  # pi/3 quite large but still feasible
parser.add_argument(
    "--alphaErrorFac", type=float, default=0
)  # pi/3 quite large but still feasible
parser.add_argument(
    "--angleError", type=float, default=0
)  # pi/3 quite large but still feasible
parser.add_argument("--usePoissonNoise", action="store_true")
parser.add_argument("--dontShuffleOrientations", action="store_true")
parser.add_argument("--dataonly", action="store_true")
parser.add_argument("--applyOTFtoGT", action="store_true")
parser.add_argument("--noStripes", action="store_true")
parser.add_argument("--seqSIM", action="store_true")
parser.add_argument("--skip_datagen", action="store_true")

opt = parser.parse_args()

np.random.seed(20221219)

# ------------ Parameters-------------
def GetParams():  # uniform randomisation
    SIMopt = argparse.Namespace()

    # phase shifts for each stripe
    SIMopt.Nshifts = opt.Nshifts
    # number of orientations of stripes
    SIMopt.Nangles = opt.Nangles
    # used to adjust PSF/OTF width
    SIMopt.scale = opt.PSFOTFscale  # + 0.1*(np.random.rand()-0.5)
    # modulation factor
    SIMopt.ModFac = opt.ModFac  # + 0.3*(np.random.rand()-0.5)
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

    return SIMopt


# main function where patterns are generated and saved to disk
def main():

    w = 512
    opt = GetParams()

    # Generation of the PSF with Besselj.

    PSFo, OTFo = PsfOtf(w, opt.scale)

    frames = SIMimage_patterns(opt, w, PSFo, OTFo)

    io.imsave("patterns.tif", np.array(frames))


if __name__ == "__main__":
    main()
