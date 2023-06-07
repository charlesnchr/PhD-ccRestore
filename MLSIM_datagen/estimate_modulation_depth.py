""" ----------------------------------------
* Creation Time : Wed Apr  5 10:49:06 2023
* Author : Charles N. Christensen
* Github : github.com/charlesnchr
----------------------------------------"""

import numpy as np
from numpy import pi, cos, sin
from numpy.fft import fft2, ifft2, fftshift, ifftshift

import argparse
from skimage import io, transform, data
from scipy.signal import convolve2d
import scipy.special
from SIMulator_functions import SIMimages, SIMimage_patterns, PsfOtf
import os
import streamlit as st
from scipy import signal
import cv2
from matplotlib import pyplot as plt
from scipy.signal import find_peaks
from matplotlib import pyplot as plt
import pandas as pd


def GetParams():  # uniform randomisation
    opt = argparse.Namespace()

    # phase shifts for each stripe
    opt.Nshifts = 3
    # number of orientations of stripes
    opt.Nangles = 3
    # used to adjust PSF/OTF width
    opt.scale = 0.9 + 0.0 * (np.random.rand() - 0.5)
    # modulation factor
    opt.ModFac = 0.8 + 0.0 * (np.random.rand() - 0.5)
    # orientation offset
    opt.alpha = 0 * pi / 3 * (np.random.rand() - 0.5)
    # orientation error
    opt.angleError = 0 * pi / 180 * (np.random.rand() - 0.5)
    # shuffle the order of orientations
    opt.shuffleOrientations = False
    # random phase shift errors
    opt.phaseError = 0 * pi * (0.5 - np.random.rand(opt.Nangles, opt.Nshifts))
    # mean illumination intensity
    opt.meanInten = np.ones(opt.Nangles)
    # amplitude of illumination intensity above mean
    opt.ampInten = np.ones(opt.Nangles) * opt.ModFac
    # illumination freq
    opt.k2 = 100 + 0 * (np.random.rand() - 0.5)
    # noise type
    opt.usePoissonNoise = False
    # noise level (percentage for Gaussian)
    opt.NoiseLevel = 8 + 0 * 8 * (np.random.rand() - 0.5)
    # 1(to blur using PSF), 0(to blur using OTF)
    opt.UsePSF = 0
    # include OTF and GT in stack
    opt.OTF_and_GT = True
    opt.noStripes = False

    return opt


def fourier_transform(img):
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    return fshift


def power_spectrum(fshift):
    magnitude_spectrum = 20 * np.log(np.abs(fshift))
    normalized_spectrum = (magnitude_spectrum - np.min(magnitude_spectrum)) / (
        np.max(magnitude_spectrum) - np.min(magnitude_spectrum)
    )
    return normalized_spectrum


def find_peaks_spectrum(normalized_spectrum, threshold):
    peaks, _ = find_peaks(normalized_spectrum.flatten(), height=threshold)
    peak_intensities = normalized_spectrum.flatten()[peaks]
    return peaks, peak_intensities


def modulation_depth(peak_intensities):
    max_intensity = np.max(peak_intensities)
    min_intensity = np.min(peak_intensities)
    modulation_depth = (max_intensity - min_intensity) / (max_intensity + min_intensity)
    return modulation_depth


def estimate_modulation_depth(img):
    fshift = fourier_transform(img)
    normalized_spectrum = power_spectrum(fshift)

    threshold = 0.5  # Set an appropriate threshold value
    peaks, peak_intensities = find_peaks_spectrum(normalized_spectrum, threshold)
    mod_depth = modulation_depth(peak_intensities)

    return mod_depth


def calculate_modulation_depth(images):
    num_orientations = 3
    num_phases = 3
    imgs_fft = []

    for i, img in enumerate(images):
        img_fft = np.abs(fftshift(fft2(img)))
        imgs_fft.append(img_fft)

    modulation_depths = np.zeros(num_orientations)

    for i in range(num_orientations):
        phase_imgs_fft = imgs_fft[i * num_phases : (i + 1) * num_phases]
        st.text(len(phase_imgs_fft))

        # Sum the Fourier transformed images for each phase
        sum_img_fft = np.sum(phase_imgs_fft, axis=0)

        # Get the coordinates of the first harmonic peak
        peak_y, peak_x = np.unravel_index(np.argmax(sum_img_fft), sum_img_fft.shape)

        # Calculate modulation depth for each orientation
        I_max = np.max(sum_img_fft[peak_y, peak_x])
        I_min = np.min(sum_img_fft[peak_y, peak_x])
        modulation_depths[i] = (I_max - I_min) / (I_max + I_min)

    # Average the modulation depths for all orientations
    overall_modulation_depth = np.mean(modulation_depths)
    return overall_modulation_depth


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
    # print("generated sample image sim_image.tif")

    fig, ax = plt.subplots()
    ax.imshow(frames[0])
    ax.plot([260, 310], [100, 100], color="red")

    st.pyplot(fig)

    return frames


def gen_sample_pattern():
    w = 512
    opt = GetParams()

    # Generation of the PSF with Besselj.

    PSFo, OTFo = PsfOtf(w, opt.scale)

    frames = SIMimage_patterns(opt, w, PSFo, OTFo)

    io.imsave("patterns.tif", np.array(frames))
    print("generated sample pattern patterns.tif")


def extract_amplitude(data=None):
    if type(data) == str:
        # Read the CSV file
        data = pd.read_csv(data)

        # Extract the sinusoidal signal
        data = data["Gray_Value"].values

    else:
        # assuming img
        data = data[100:101, 260:310].flatten()

    # plot
    fig, ax = plt.subplots()
    ax.plot(data)
    st.pyplot(fig)

    data = signal.detrend(data)

    # plot
    fig, ax = plt.subplots()
    ax.plot(data)
    st.pyplot(fig)

    # Calculate the maximum and minimum values of the signal
    max_signal = np.max(data)
    min_signal = np.min(data)

    st.text("Max Signal:" + str(max_signal))
    st.text("Min Signal:" + str(min_signal))

    # Compute the amplitude
    amplitude = (max_signal - min_signal) / 2

    return amplitude


if __name__ == "__main__":
    img = gen_sample_images()
    # gen_sample_pattern()

    ampl = extract_amplitude(img[0])
    st.text("Amplitude of the sinusoidal signal:" + str(ampl))
