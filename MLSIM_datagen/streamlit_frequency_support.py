""" ----------------------------------------
* Creation Time : Wed Jun 21 15:51:31 2023
* Author : Charles N. Christensen
* Github : github.com/charlesnchr
----------------------------------------"""

# Import libraries
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

import sys
import numpy as np
from numpy import pi, cos, sin
import math
from skimage import io, transform, exposure, img_as_ubyte, img_as_float
import glob
import os
import argparse
from multiprocessing import Pool
import subprocess
import MLSIM_datagen.SIMulator_functions
from MLSIM_pipeline import *
import MLSIM_datagen.SeqSIMulator_functions
import run
import shutil
import wandb


def radial_profile(data, center):
    y, x = np.indices((data.shape))
    r = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
    r = r.astype(np.int)

    tbin = np.bincount(r.ravel(), data.ravel())
    nr = np.bincount(r.ravel())
    radialprofile = tbin / nr
    return radialprofile


def GetParams_frequency_support_investigation_20230621():  # uniform randomisation
    SIMopt = argparse.Namespace()

    # modulation factor
    SIMopt.ModFac = opt.ModFac

    # ---- stripes
    # phase shifts for each stripe
    SIMopt.Nshifts = opt.Nshifts
    # number of orientations of stripes
    SIMopt.Nangles = opt.Nangles
    # orientation offset
    SIMopt.alpha = 0
    # orientation error
    SIMopt.angleError = 0
    # shuffle the order of orientations
    SIMopt.shuffleOrientations = False
    # random phase shift errors
    SIMopt.phaseError = np.zeros((SIMopt.Nangles, SIMopt.Nshifts))
    # illumination freq
    SIMopt.k2 = opt.k2

    # --- spots
    SIMopt.Nspots = opt.Nspots
    SIMopt.spotSize = opt.spotSize

    # used to adjust PSF/OTF width
    SIMopt.PSFOTFscale = opt.PSFOTFscale
    # noise type
    SIMopt.usePoissonNoise = opt.usePoissonNoise
    # noise level (percentage for Gaussian)
    SIMopt.NoiseLevel = opt.NoiseLevel
    # 1(to blur using PSF), 0(to blur using OTF)
    SIMopt.UsePSF = opt.usePSF
    # include OTF and GT in stack
    SIMopt.OTF_and_GT = True
    # use a blurred target (according to theoretical optimal construction)
    SIMopt.applyOTFtoGT = opt.applyOTFtoGT
    # whether to simulate images using just widefield illumination
    SIMopt.noStripes = opt.noStripes

    # function to use for stripes
    SIMopt.func = opt.func

    SIMopt.patterns = opt.patterns
    SIMopt.crop_factor = opt.crop_factor
    SIMopt.SIMmodality = opt.SIMmodality
    SIMopt.dmdMapping = opt.dmdMapping

    # --- Nframes
    if SIMopt.SIMmodality == "stripes":
        SIMopt.Nframes = SIMopt.Nangles * SIMopt.Nshifts
        # amplitude of illumination intensity above mean
        SIMopt.ampInten = np.ones(SIMopt.Nangles) * SIMopt.ModFac
        # mean illumination intensity
        SIMopt.meanInten = 1 - SIMopt.ampInten
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


# Define a function to plot the OTF
def calc_otf(img):
    print(f"{img.shape}, {img.dtype}, {img.min()}, {img.max()}")
    img = img_as_float(img)
    print(f"{img.shape}, {img.dtype}, {img.min()}, {img.max()}")

    # Compute the Fourier Transform of the image
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)

    # Compute the magnitude spectrum (log transformed for better visualization)
    magnitude_spectrum = np.abs(fshift)

    return magnitude_spectrum


def proj_otf(imgstack):
    fig = plt.figure()

    ffts = []

    for img in imgstack:
        print(f"{img.shape}, {img.dtype}, {img.min()}, {img.max()}", end=" — ")
        img = img_as_float(img)
        print(f"{img.shape}, {img.dtype}, {img.min()}, {img.max()}")

        # Compute the Fourier Transform of the image
        f = np.fft.fft2(img)
        fshift = np.fft.fftshift(f)

        ffts.append(fshift)

    # max projection
    ffts = np.array(ffts)
    ffts = np.max(ffts, axis=0)
    proj_fft = np.abs(ffts)
    return proj_fft


def estimate_noise_floor(data, percentile=5):
    return np.percentile(data, percentile)


def estimate_max(data, percentile=98):
    return np.percentile(data, percentile)


def plot_otf_and_profile(data, center):
    """Plot OTF with cutoff circle and 1D radial profile."""
    # Compute 1D radial profile
    radial_profile_1d = radial_profile(data, center)

    # Estimate noise floor and compute adjusted profile
    noise_floor = estimate_noise_floor(radial_profile_1d)
    max_value = estimate_max(radial_profile_1d)
    adjusted_profile = np.maximum(radial_profile_1d - noise_floor, 0)

    # Find the radius where the magnitude has decreased to 10% of its maximum
    cutoff = max_value * 0.1
    indices = np.where(adjusted_profile < cutoff)[0]
    cutoff_radius = indices[0] if indices.size > 0 else len(radial_profile_1d) - 1

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1D radial profile of the OTF
    axs[0].semilogy(radial_profile_1d)
    axs[0].axvline(
        x=cutoff_radius, color="r", linestyle="--"
    )  # Indicate the cutoff radius
    axs[0].axhline(
        y=cutoff + noise_floor, color="r", linestyle="--"
    )  # Indicate the cutoff value
    axs[0].axhline(y=noise_floor, color="r", linestyle="--")  # Indicate noise floor
    axs[0].axhline(y=max_value, color="r", linestyle="--")  # Indicate max
    axs[0].set_title("1D Radial Profile of OTF")
    axs[0].set_xlabel("20 * log(OTF)")

    # Add annotations
    axs[0].annotate(
        "10 % Cutoff",
        xy=(0, 1.2 * cutoff + noise_floor),
        xytext=(len(radial_profile_1d) // 2, 1.2 * cutoff + noise_floor),
    )
    axs[0].annotate(
        "Noise floor",
        xy=(0, 1.2 * noise_floor),
        xytext=(len(radial_profile_1d) // 2, 1.2 * noise_floor),
    )
    axs[0].annotate(
        "Max",
        xy=(0, 1.2 * max_value),
        xytext=(len(radial_profile_1d) // 2, 1.2 * max_value),
    )
    axs[0].annotate(
        f"Cutoff Radius={cutoff_radius}",
        xy=(1.2 * cutoff_radius, 1.5 * max_value),
        xytext=(1.2 * cutoff_radius, 1.5 * max_value),
        fontsize=8,
        rotation=90,
    )

    # Plot OTF with cutoff circle
    # normalise before plotting
    p1, p2 = noise_floor, max_value
    # plot_data = data
    # plot_data = np.clip((data - p1) / (p2 - p1), 0, 1) + 1
    plot_data = np.clip(data, p1, p2)

    axs[1].imshow(20 * np.log(plot_data), cmap="gray")
    cutoff_circle = plt.Circle(center, cutoff_radius, color="r", fill=False)
    axs[1].add_artist(cutoff_circle)
    axs[1].set_title(f"OTF with Cutoff (radius={cutoff_radius})")

    return fig


# Define the Streamlit app
def main():
    st.title("Structured Illumination Microscopy (SIM) OTF Visualiser")

    opt.imageSize = [512, 512]
    opt.scale = 2
    opt.sourceimages_path = "MLSIM_datagen"
    opt.root = "MLSIM_datagen"
    opt.out = "MLSIM_datagen"
    opt.ModFac = 0.9
    opt.PSFOTFscale = 0.6
    opt.k2 = 80
    opt.SIMmodality = "stripes"
    opt.Nspots = 10
    opt.Nshifts = 10
    opt.spotSize = 2
    opt.NoiseLevel = 20
    opt.func = MLSIM_datagen.SIMulator_functions.cos

    print(opt)

    projs = []
    wf_spectra = []
    images = glob.glob("MLSIM_datagen/DIV2K_subset/*.png")
    N = 10
    bar = st.progress(0)

    for i in range(N):
        SIMopt = GetParams_frequency_support_investigation_20230621()
        I = processImage(images[i], SIMopt)

        wf = I.mean(axis=0)
        wf_spectrum = calc_otf(wf)
        wf_spectra.append(wf_spectrum)

        proj = proj_otf(I)
        projs.append(proj)

        bar.progress((i + 1) / N)

    cols = st.columns(2)

    with cols[0]:
        fig = plt.figure(figsize=(10, 10))
        # img = io.imread(images[N - 1])
        plt.imshow(wf)
        st.pyplot(fig)
    with cols[1]:
        fig = plt.figure(figsize=(10, 10))
        plt.imshow(I[0])
        st.pyplot(fig)

    wf_spectra = np.array(wf_spectra)
    wf_spectrum = np.mean(wf_spectra, axis=0)

    projs = np.array(projs)
    proj = np.mean(projs, axis=0)
    center = (proj.shape[0] // 2, proj.shape[1] // 2)

    st.pyplot(plot_otf_and_profile(wf_spectrum, center))
    st.pyplot(plot_otf_and_profile(proj, center))


# Run the Streamlit app
if __name__ == "__main__":
    main()
