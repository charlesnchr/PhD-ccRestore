""" ----------------------------------------
* Creation Time : Mon Dec 19 14:41:48 2022
* Author : Charles N. Christensen
* Github : github.com/charlesnchr
----------------------------------------"""

import numpy as np
from numpy import pi, cos, sin
from numpy.fft import fft2, ifft2, fftshift, ifftshift

from skimage import io, transform, data, img_as_ubyte, exposure
from scipy.signal import convolve2d
import scipy.special
import argparse
from SIMulator_functions import (
    SIMimages,
    SIMimage_patterns,
    PsfOtf,
    square_wave_one_third,
    square_wave,
)
from SIMulator import GetParams
import os


def GetParams_Exact():  # uniform randomisation
    opt = argparse.Namespace()

    # phase shifts for each stripe
    opt.Nshifts = 3
    # number of orientations of stripes
    opt.Nangles = 3
    # used to adjust PSF/OTF width
    opt.scale = 0.9
    # modulation factor
    opt.ModFac = 0.8
    # orientation offset
    opt.alpha = 0
    # orientation error
    opt.angleError = 0
    # shuffle the order of orientations
    opt.shuffleOrientations = False
    # random phase shift errors
    opt.phaseError = 0 * (0.5 - np.random.rand(opt.Nangles, opt.Nshifts))
    # mean illumination intensity
    opt.meanInten = np.ones(opt.Nangles)
    # amplitude of illumination intensity above mean
    opt.ampInten = np.ones(opt.Nangles) * opt.ModFac
    # illumination freq
    opt.k2 = 100
    # noise type
    opt.usePoissonNoise = False
    # noise level (percentage for Gaussian)
    opt.NoiseLevel = 0
    # 1(to blur using PSF), 0(to blur using OTF)
    opt.UsePSF = 0
    # include OTF and GT in stack
    opt.OTF_and_GT = True
    opt.applyOTFtoGT = True
    opt.noStripes = False

    return opt


def gen_sample_images():
    # function that uses data.astronaut to generate a simulated image
    # with stripes and saves it to disk
    w = 512
    opt = GetParams_Exact()

    img = data.astronaut()
    # resize to 684, 428
    img = transform.resize(img, (684, 428))
    img = np.mean(img, axis=2)

    # Generation of the PSF with Besselj.
    PSFo, OTFo = PsfOtf(w, opt.scale)
    frames = SIMimages(opt, img, PSFo, OTFo)

    io.imsave("sim_image.tif", np.array(frames))
    print("generated sample image sim_image.tif")


def gen_sample_pattern():
    w = 512
    opt = GetParams_Exact()

    PSFo, OTFo = PsfOtf(w, opt.scale)

    frames = SIMimage_patterns(opt, w, PSFo, OTFo, func=square_wave_one_third)

    new_frames = []
    for frame in frames:
        frame = exposure.rescale_intensity(frame, out_range="uint8")
        frame = img_as_ubyte(frame)
        new_frames.append(frame)
    frames = np.array(new_frames)

    io.imsave("patterns.tif", frames)
    print("generated sample pattern patterns.tif")


def read_sample_pattern():
    # read sample pattern, calculate mean of each three consecutive frames (1-3, 4-6, 7-9) and plot with streamlit
    import streamlit as st
    import matplotlib.pyplot as plt

    stack = io.imread("patterns.tif")
    for i in range(3):
        st.text(f"min: {np.min(stack[i])}, max: {np.max(stack[i])}")
        meanframe = np.mean(stack[i * 3 : (i + 1) * 3], axis=0)
        st.text(f"min: {np.min(meanframe)}, max: {np.max(meanframe)}")
        fig, ax = plt.subplots()
        ax.imshow(meanframe, cmap="gray", vmin=0, vmax=255)
        st.pyplot(fig)


if __name__ == "__main__":
    # gen_sample_images()
    gen_sample_pattern()
    read_sample_pattern()
