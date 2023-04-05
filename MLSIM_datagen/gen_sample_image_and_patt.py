""" ----------------------------------------
* Creation Time : Mon Dec 19 14:41:48 2022
* Author : Charles N. Christensen
* Github : github.com/charlesnchr
----------------------------------------"""

import numpy as np
from numpy import pi, cos, sin
from numpy.fft import fft2, ifft2, fftshift, ifftshift

from skimage import io, transform, data
from scipy.signal import convolve2d
import scipy.special
from SIMulator_functions import SIMimages, SIMimage_patterns, PsfOtf
from SIMulator import GetParams
import os


def gen_sample_images():
    # function that uses data.astronaut to generate a simulated image
    # with stripes and saves it to disk
    w = 512
    opt = GetParams()

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
    opt = GetParams()

    # Generation of the PSF with Besselj.

    PSFo, OTFo = PsfOtf(w, opt.scale)

    frames = SIMimage_patterns(opt, w, PSFo, OTFo)

    io.imsave("patterns.tif", np.array(frames))
    print("generated sample pattern patterns.tif")


if __name__ == "__main__":
    gen_sample_images()
    gen_sample_pattern()
