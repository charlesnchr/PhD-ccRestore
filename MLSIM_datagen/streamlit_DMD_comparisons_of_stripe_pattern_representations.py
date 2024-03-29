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
    SIMimages_speckle,
    SIMimages_spots,
    PsfOtf,
    square_wave_one_third,
    square_wave,
)
import os
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# not needed for generating (just reading/visualising)
import streamlit as st


def GetParams_20230410():  # uniform randomisation
    opt = argparse.Namespace()

    # phase shifts for each stripe
    opt.Nshifts = 3
    # number of orientations of stripes
    opt.Nangles = 3
    # used to adjust PSF/OTF width
    opt.OTFPSFscale = 0.9 + 0.1 * (np.random.rand() - 0.5)
    # modulation factor
    opt.ModFac = 0.2 + 0.3 * (np.random.rand() - 0.5)
    # orientation offset
    opt.alpha = 0 * pi / 3 * (np.random.rand() - 0.5)
    # orientation error
    opt.angleError = 10 * pi / 180 * (np.random.rand() - 0.5)
    # shuffle the order of orientations
    opt.shuffleOrientations = False
    # random phase shift errors
    opt.phaseError = 10 * pi / 180 * (0.5 - np.random.rand(opt.Nangles, opt.Nshifts))
    # mean illumination intensity
    opt.meanInten = np.ones(opt.Nangles) * 0.5
    # amplitude of illumination intensity above mean
    opt.ampInten = np.ones(opt.Nangles) * 0.5 * opt.ModFac
    # illumination freq
    opt.k2 = 110 + 30 * (np.random.rand() - 0.5)
    # noise type
    opt.usePoissonNoise = False
    # noise level (percentage for Gaussian)
    opt.NoiseLevel = 15 + 15 * (np.random.rand() - 0.5)
    # 1(to blur using PSF), 0(to blur using OTF)
    opt.UsePSF = 0
    # include OTF and GT in stack
    opt.OTF_and_GT = True
    opt.noStripes = False

    return opt


def GetParams_Exact():  # uniform randomisation
    opt = argparse.Namespace()

    # phase shifts for each stripe
    opt.Nshifts = 3
    # number of orientations of stripes
    opt.Nangles = 3
    # used to adjust PSF/OTF width
    opt.PSFOTFscale = 0.9
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
    opt.meanInten = np.ones(opt.Nangles) * 0.5
    # amplitude of illumination intensity above mean
    opt.ampInten = np.ones(opt.Nangles) * 0.5 * opt.ModFac
    # illumination freq
    opt.k2 = 80
    # noise type
    opt.usePoissonNoise = False
    # noise level (percentage for Gaussian)
    opt.NoiseLevel = 0
    # 1(to blur using PSF), 0(to blur using OTF)
    opt.UsePSF = 0
    # include OTF and GT in stack
    opt.OTF_and_GT = True
    opt.noStripes = False

    return opt


def gen_sample_images():
    # function that uses data.astronaut to generate a simulated image
    # with stripes and saves it to disk
    w = 512
    opt = GetParams_20230410()

    img = data.astronaut()
    # resize to 684, 428
    img = transform.resize(img, (684, 428))
    img = np.mean(img, axis=2)

    opt.k2 = 110
    pixelsize_ratio = 1

    # Generation of the PSF with Besselj.
    frames = SIMimages_spots(
        opt,
        img.shape[0],
        func=square_wave_one_third,
        pixelsize_ratio=pixelsize_ratio,
    )

    io.imsave("sim_image.tif", np.array(frames))
    print("generated sample image sim_image.tif")


def read_sample_image():
    # read sample pattern, calculate mean of each three consecutive frames (1-3, 4-6, 7-9) and plot with streamlit

    stack = io.imread("sim_image.tif")
    stack = exposure.rescale_intensity(stack, out_range="uint8")

    # checking pattern cancellation
    # for i in range(3):
    #     st.text(f"min: {np.min(stack[i])}, max: {np.max(stack[i])}")
    #     meanframe = np.mean(stack[i * 3 : (i + 1) * 3], axis=0)
    #     st.text(f"min: {np.min(meanframe)}, max: {np.max(meanframe)}")
    #     fig, ax = plt.subplots()
    #     ax.imshow(meanframe, cmap="gray", vmin=0, vmax=255)
    #     st.pyplot(fig)

    # widefield projection
    meanframe = np.mean(stack, axis=0)
    st.text(f"min: {np.min(meanframe)}, max: {np.max(meanframe)}")
    fig, ax = plt.subplots(figsize=(30, 10))
    ax.imshow(meanframe, cmap="gray", vmin=0, vmax=255)
    st.pyplot(fig)

    fig, ax = plt.subplots(figsize=(30, 10))
    ax.imshow(stack[0], cmap="gray", vmin=0, vmax=255)
    st.pyplot(fig)

    fig, ax = plt.subplots(figsize=(30, 10))
    ax.imshow(stack[3], cmap="gray", vmin=0, vmax=255)
    st.pyplot(fig)

    fig, ax = plt.subplots(figsize=(30, 10))
    ax.imshow(stack[6], cmap="gray", vmin=0, vmax=255)
    st.pyplot(fig)

    curve = np.mean(stack[0], axis=0)

    # count peaks on curve and plot
    peaks = find_peaks(curve, height=0.5, distance=5)
    fig, ax = plt.subplots()
    ax.plot(curve)
    ax.plot(peaks[0], curve[peaks[0]], "x")
    st.pyplot(fig)
    st.text(f"peaks: {len(peaks[0])}")


def read_exp_sample_image():
    # read sample pattern, calculate mean of each three consecutive frames (1-3, 4-6, 7-9) and plot with streamlit

    stack = io.imread(
        "plugins-branch/data/highlighter-april2024/highlighter-k2-80/right/image_10_April_2023_09_38_PM.tif"
    )
    stack = exposure.rescale_intensity(stack, out_range="uint8")

    fig, ax = plt.subplots()
    ax.imshow(stack[0], cmap="gray", vmin=0, vmax=255)
    st.pyplot(fig)

    fig, ax = plt.subplots()
    ax.imshow(stack[3], cmap="gray", vmin=0, vmax=255)
    st.pyplot(fig)

    fig, ax = plt.subplots()
    ax.imshow(stack[6], cmap="gray", vmin=0, vmax=255)
    st.pyplot(fig)

    curve = np.mean(stack[0], axis=0)
    # count peaks on curve and plot

    peaks = find_peaks(curve, height=0.5, distance=10)
    fig, ax = plt.subplots()
    ax.plot(curve)
    ax.plot(peaks[0], curve[peaks[0]], "x")
    st.pyplot(fig)
    st.text(f"peaks: {len(peaks[0])}")


def gen_sample_pattern(opt):
    w = 512


    # opt.k2 = 70
    opt.k2 = 20  # most used value so far
    # opt.k2 = 90
    # opt.k2 = 100
    pixelsize_ratio = 1
    # pixelsize_ratio = 1.6
    # pixelsize_ratio = 1.7
    # pixelsize_ratio = 1.8  # seemingly best value so far
    # func = np.cos
    # func = square_wave
    func = square_wave_one_third  # seems best for DMD

    img = data.astronaut().mean(axis=2)

    # regular stripes
    opt.noStripes = False
    frames = SIMimages(opt, w, func=func, pixelsize_ratio=pixelsize_ratio)

    # speckles
    # opt.Nframes = 100
    # opt.Nspeckles = 100
    # opt.crop_factor = False
    # frames = SIMimages_speckle(opt, img, PSFo, OTFo)

    # frames = SIMimages_spots(opt, img.shape[0])

    # new_frames = []
    # for frame in frames:
    #     frame = exposure.rescale_intensity(frame, out_range="uint8")
    #     frame = img_as_ubyte(frame)
    #     new_frames.append(frame)
    # frames = np.array(new_frames)

    frames = np.array(frames)
    frames = exposure.rescale_intensity(frames, out_range="uint8")

    io.imsave(
        f"patterns.tif",
        frames,
    )
    print("generated sample pattern patterns.tif")


def gen_sample_pattern_loop_stripes(opt):
    w = 512
    # opt = GetParams_Exact()

    opt.patterns = True

    k2_arr = [200]
    pixelsize_ratio_arr = [1.6, 1.7, 1.8]
    func_arr = [np.cos, square_wave, square_wave_one_third]

    for k2 in k2_arr:
        for pixelsize_ratio in pixelsize_ratio_arr:
            for func in func_arr:
                opt.k2 = k2

                frames = SIMimages(
                    opt, w, func=func, pixelsize_ratio=pixelsize_ratio
                )

                new_frames = []
                for frame in frames:
                    frame = exposure.rescale_intensity(frame, out_range="uint8")
                    frame = img_as_ubyte(frame)

                    new_frames.append(frame)
                frames = np.array(new_frames)

                io.imsave(
                    f"../plugins-branch/sim_patterns/new_patterns_pixelsize_ratio_{pixelsize_ratio}_k2_{opt.k2}_func_{func.__name__}.tif",
                    frames,
                )
                print("generated sample pattern patterns.tif")


def gen_sample_pattern_loop_spots(opt):
    w = 512

    opt.patterns = True

    spotSize_arr = [1, 2]
    # Nspots_arr = [3, 5, 8, 10]
    Nspots_arr = [20]

    for spotSize in spotSize_arr:
        for Nspots in Nspots_arr:
            for dmdMapping in [True, False]:
                opt.spotSize = spotSize
                opt.Nspots = Nspots
                opt.Nframes = 2  # opt.Nspots**2
                opt.dmdMapping = dmdMapping
                frames = SIMimages_spots(opt, w)

                new_frames = []
                for frame in frames:
                    frame = exposure.rescale_intensity(frame, out_range="uint8")
                    frame = img_as_ubyte(frame)
                    new_frames.append(frame)
                frames = np.array(new_frames)

                os.makedirs("../plugins-branch/spot_patterns", exist_ok=True)
                io.imsave(
                    f"../plugins-branch/spot_patterns/patterns_spotSize_{spotSize}_Nspots_{Nspots}_dmdMapping_{int(opt.dmdMapping)}.tif",
                    frames,
                )
                print("generated sample pattern patterns.tif")


def read_sample_pattern(opt):
    # read sample pattern, calculate mean of each three consecutive frames (1-3, 4-6, 7-9) and plot with streamlit
    stack = io.imread("patterns.tif")

    nspots = opt.Nspots
    spotSize = opt.spotSize

    cols = st.columns(3)

    with cols[0]:
        fig, ax = plt.subplots(figsize=(10, 10))
        # wf = stack.mean(axis=0)
        wf = stack[0]
        st.text(f"min: {np.min(wf)}, max: {np.max(wf)}")
        ax.imshow(wf, cmap="gray")
        plt.title(f"Spot size {spotSize}x{spotSize}, pattern {nspots}x{nspots}")
        plt.savefig(
            f"{nspots}x{nspots}_spot_size_{spotSize}x{spotSize}.png",
            dpi=300,
            bbox_inches="tight",
            pad_inches=0,
        )
        st.pyplot(fig, dpi=300)

    with cols[1]:
        fig, ax = plt.subplots(figsize=(10, 10))
        # wf = stack.mean(axis=0)
        wf = stack[-1]
        st.text(f"min: {np.min(wf)}, max: {np.max(wf)}")
        ax.imshow(wf, cmap="gray")
        plt.title(f"Spot size {spotSize}x{spotSize}, pattern {nspots}x{nspots}")
        plt.savefig(
            f"{nspots}x{nspots}_spot_size_{spotSize}x{spotSize}.png",
            dpi=300,
            bbox_inches="tight",
            pad_inches=0,
        )
        st.pyplot(fig, dpi=300)

    with cols[2]:
        fig, ax = plt.subplots(figsize=(10, 10))
        wf = stack.mean(axis=0)
        st.text(f"min: {np.min(wf)}, max: {np.max(wf)}")
        ax.imshow(wf, cmap="gray")
        plt.title("Wide-field projection")
        st.pyplot(fig, dpi=300)


if __name__ == "__main__":
    st.set_page_config(layout="wide")

    opt = GetParams_Exact()
    opt.Nspots = 10
    opt.spotSize = 2
    opt.Nframes = opt.Nspots**2
    opt.crop_factor = True
    opt.dmdMapping = True

    # gen_sample_images()
    # read_sample_image()
    # read_exp_sample_image()

    # gen_sample_pattern_loop_stripes(opt)
    # gen_sample_pattern_loop_spots(opt)

    st.header("Tilted pattern (natural coordinate system)")

    opt.dmdMapping = 0
    opt.patterns = True

    gen_sample_pattern(opt)
    read_sample_pattern(opt)

    st.header("Preprocessed for DMD (coordinate mapping)")

    opt.dmdMapping = 1
    opt.patterns = True

    gen_sample_pattern(opt)
    read_sample_pattern(opt)

    st.header("Rotated by -45 degrees for forward modelling")

    opt.dmdMapping = 2
    opt.patterns = True

    gen_sample_pattern(opt)
    read_sample_pattern(opt)

    # gen_sample_pattern_loop()
