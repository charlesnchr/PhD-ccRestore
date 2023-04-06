""" ----------------------------------------
* Creation Time : Thu Apr  6 12:45:13 2023
* Author : Charles N. Christensen
* Github : github.com/charlesnchr
----------------------------------------"""


import numpy as np
from numpy import pi, cos, sin
from numpy.fft import fft2, ifft2, fftshift, ifftshift

from skimage import io, transform
from scipy.signal import convolve2d
import scipy.special
import matplotlib.pyplot as plt
import streamlit as st


def square_wave(x):
    return 2 * np.heaviside(np.cos(x), 0) - 1


def square_wave_one_third(x):
    # sums to 0
    return 2 * (np.heaviside(np.cos(x) - np.cos(1 * np.pi / 3), 0) - 1 / 3)


def discretized_sine(x):
    return np.where(
        np.sin(x) >= np.sin(2 * np.pi / 3),
        1,
        np.where(np.sin(x) <= np.sin(4 * np.pi / 3), -1, 0),
    )


def symmetric_sawtooth_wave(x):
    return 4 * (0.5 - np.abs(((x / (2 * np.pi)) % 1) - 0.5)) - 1


def plot_summed_wave(func=np.cos, plotting_displacement=0):
    k2 = 50
    w = 512
    wo = w / 2
    theta = 0
    k2val = (k2 / w) * np.array([cos(theta), sin(theta)])

    # illumination phase shifts along directions with errors
    ps = np.zeros((1, 3))
    for i_s in range(3):
        ps[0, i_s] = 2 * pi * i_s / 3

    # x vec
    X = np.linspace(0, w / 10, w)

    fig, ax = plt.subplots()

    # add three square waves
    sum_wave = np.zeros((1, w))
    for i_s in range(3):
        wave = func(2 * pi * (k2val[0] * (X - wo)) + ps[0, i_s])
        sum_wave += wave

        # plot with transparency
        ax.plot(X, wave + plotting_displacement * i_s / 10, alpha=0.5)

    st.title(f"{func.__name__}")

    st.pyplot(fig)
    st.text(f"individual waves, {func.__name__}")

    fig, ax = plt.subplots()
    ax.plot(X, sum_wave[0, :])
    plt.ylim([-1.1, 1.1])
    st.pyplot(fig)
    st.text(f"summed wave, {func.__name__}")


plot_summed_wave()
plot_summed_wave(square_wave, plotting_displacement=1)
plot_summed_wave(square_wave_one_third, plotting_displacement=1)
plot_summed_wave(discretized_sine, plotting_displacement=1)
plot_summed_wave(symmetric_sawtooth_wave, plotting_displacement=0)
