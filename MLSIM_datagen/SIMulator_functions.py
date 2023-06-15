import numpy as np
from numpy import pi, cos, sin
from numpy.fft import fft2, ifft2, fftshift, ifftshift

from skimage import io, draw, transform, img_as_ubyte, img_as_float
from scipy.signal import convolve2d
import scipy.special

from numba import jit

import time


@jit(nopython=True)
def DMDPixelTransform(input_img, dmdMapping, xoffset=0, yoffset=0):
    # Initialize an array of zeros with same size as the input image
    transformed_img = np.zeros_like(input_img)

    # Get the dimensions of the input image
    rows, cols = input_img.shape

    # Iterate over the pixels of the input image
    for i in range(rows):
        for j in range(cols):
            # Calculate the new coordinates for the pixel
            ip = i + yoffset
            jp = j + xoffset

            # Apply the dmdMapping transformation if set
            if dmdMapping > 0:
                transformed_i = jp + ip - 2
                transformed_j = (jp - ip + 4) // 2
            else:
                transformed_i = ip
                transformed_j = jp

            # If the new coordinates are within the bounds of the image, copy the pixel value
            if 0 <= transformed_i < rows and 0 <= transformed_j < cols:
                transformed_img[transformed_i, transformed_j] = input_img[i, j]

    # Return the transformed image
    return transformed_img


def Get_X_Y_MeshGrids(w, opt, forPSF=False):
    # TODO: these hard-coded values are not ideal
    #  and this way of scaling the patterns is
    #  likely going to lead to undesired behaviour

    if opt.crop_factor:
        if opt.patterns == True:  # assuming DMD resolution
            crop_factor_x = 1
            crop_factor_y = 1
        else:
            crop_factor_x = 428 / 912
            crop_factor_y = 684 / 1140

        # data from dec 2022 acquired with DMD patterns with the below factors
        # crop_factor_x = 1
        # crop_factor_y = 1

        # first version, december 2022
        # wo = w / 2
        # x = np.linspace(0, w - 1, 912)
        # y = np.linspace(0, w - 1, 1140)
        # [X, Y] = np.meshgrid(x, y)

        if (opt.dmdMapping == 2 or (opt.dmdMapping == 1 and opt.SIMmodality == 'stripes'))  and not forPSF:
            padding = 4
        else:
            padding = 1

        x = np.linspace(
            0, padding * crop_factor_x * 512 - 1, padding * int(crop_factor_x * 912)
        )
        y = np.linspace(
            0, padding * crop_factor_y * 512 - 1, padding * int(crop_factor_y * 1140)
        )
        [X, Y] = np.meshgrid(x, y)
    else:
        x = np.linspace(0, w - 1, w)
        y = np.linspace(0, w - 1, w)
        X, Y = np.meshgrid(x, y)

    return X, Y


def PsfOtf(w, opt):
    # AIM: To generate PSF and OTF using Bessel function
    # INPUT VARIABLES
    #   w: image size
    #   scale: a parameter used to adjust PSF/OTF width
    # OUTPUT VRAIBLES
    #   yyo: system PSF
    #   OTF2dc: system OTF
    eps = np.finfo(np.float64).eps

    X, Y = Get_X_Y_MeshGrids(w, opt, forPSF=True)

    scale = opt.PSFOTFscale

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


def SIMimages(opt, DIo, func=np.cos, pixelsize_ratio=1):
    # AIM: to generate raw sim images
    # INPUT VARIABLES
    #   k2: illumination frequency
    #   DIo: specimen image or integer (dimension) if only patterns are wanted
    #   PSFo: system PSF
    #   OTFo: system OTF
    #   UsePSF: 1 (to blur SIM images by convloving with PSF)
    #           0 (to blur SIM images by truncating its fourier content beyond OTF)
    #   NoiseLevel: percentage noise level for generating gaussian noise
    # OUTPUT VARIABLES
    #   frames:  raw sim images
    #   DIoTnoisy: noisy wide field image
    #   DIoT: noise-free wide field image

    if type(DIo) == int:
        opt.patterns = True
        w = DIo
        wo = w / 2
    else:
        opt.patterns = False
        w = DIo.shape[0]
        wo = w / 2

    X, Y = Get_X_Y_MeshGrids(w, opt)

    PSFo, OTFo = PsfOtf(w, opt)

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
        k2mat[i, :] = np.array(
            [(opt.k2 * pixelsize_ratio / w) * cos(theta), (opt.k2 / w) * sin(theta)]
        )

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
            if not opt.noStripes:
                sig = opt.meanInten[i_a] + opt.ampInten[i_a] * func(
                    2 * pi * (k2mat[i_a, 0] * (X - wo) + k2mat[i_a, 1] * (Y - wo))
                    + ps[i_a, i_s]
                )
            else:
                sig = 1  # simulating widefield

            # whether to transform sig for dmd
            if opt.dmdMapping > 0:
            # crop to upper left quadrant if padding was added
                if opt.dmdMapping == 1:
                    sig = DMDPixelTransform(
                        sig,
                        opt.dmdMapping,
                        xoffset=-sig.shape[1] // 2,
                        yoffset=-sig.shape[0] // 2,
                    )
                    sig = sig[: sig.shape[0] // 4, : sig.shape[1] // 4]
                elif opt.dmdMapping == 2:
                    # rotate image by 45 degrees
                    rotated_image = transform.rotate(sig, -45)

                    rows, cols = rotated_image.shape[0], rotated_image.shape[1]

                    # crop centre to avoid black corners
                    row_start = rows // 4 + rows // 8
                    row_end = row_start + rows // 4
                    col_start = cols // 4 + cols // 8
                    col_end = col_start + cols // 4

                    # Crop the center of the image
                    sig = rotated_image[row_start:row_end, col_start:col_end]


            if opt.patterns:
                frame = sig
            else:
                sup_sig = DIo * sig  # superposed signal

                # superposed (noise-free) Images
                if opt.UsePSF == 1:
                    ST = conv2(sup_sig, PSFo, "same")
                else:
                    ST = np.real(ifft2(fft2(sup_sig) * fftshift(OTFo)))

                # Noise generation
                if opt.usePoissonNoise:
                    # Poisson
                    vals = 2 ** np.ceil(
                        np.log2(opt.NoiseLevel)
                    )  # NoiseLevel could be 200 for Poisson: degradation seems similar to Noiselevel 20 for Gaussian
                    STnoisy = np.random.poisson(ST * vals) / float(vals)
                else:
                    # Gaussian
                    aNoise = opt.NoiseLevel / 100  # noise
                    # SNR = 1/aNoise
                    # SNRdb = 20*log10(1/aNoise)

                    nST = np.random.normal(0, aNoise * np.std(ST, ddof=1), (ST.shape))
                    NoiseFrac = 1  # may be set to 0 to avoid noise addition
                    # noise added raw SIM images
                    STnoisy = ST + NoiseFrac * nST

                frame = STnoisy.clip(0, 1)


            frames.append(frame)

    return frames


def GenSpeckle(dim, opt):
    N = opt.Nspeckles
    I = np.zeros((dim, dim))
    randx = np.random.choice(
        list(range(dim)) * np.ceil(N / dim).astype("int"), size=N, replace=False
    )
    randy = np.random.choice(
        list(range(dim)) * np.ceil(N / dim).astype("int"), size=N, replace=False
    )

    for i in range(N):
        x = randx[i]
        y = randy[i]

        r = np.random.randint(3, 5)
        cr, cc = draw.ellipse(x, y, r, r, (dim, dim))
        I[cr, cc] += 0.1
    return I


def SIMimages_speckle(opt, DIo):
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

    w = DIo.shape[0]
    X, Y = Get_X_Y_MeshGrids(w, opt)

    PSFo, OTFo = PsfOtf(w, opt)

    # illumination patterns
    frames = []
    for i_a in range(opt.Nframes):
        # illuminated signal
        sig = GenSpeckle(
            w, opt
        )  # opt.meanInten[i_a] + opt.ampInten[i_a] * GenSpeckle(w)

        sup_sig = DIo * sig  # superposed signal

        # superposed (noise-free) Images
        if opt.UsePSF == 1:
            ST = conv2(sup_sig, PSFo, "same")
        else:
            ST = np.real(ifft2(fft2(sup_sig) * fftshift(OTFo)))

        # Gaussian noise generation
        aNoise = opt.NoiseLevel / 100  # noise
        # SNR = 1/aNoise
        # SNRdb = 20*log10(1/aNoise)

        nST = np.random.normal(0, aNoise * np.std(ST, ddof=1), (w, w))
        NoiseFrac = 1  # may be set to 0 to avoid noise addition
        # noise added raw SIM images
        STnoisy = ST + NoiseFrac * nST
        frames.append(STnoisy)

    return frames


@jit(nopython=True)
def GenSpots(rows, cols, Nspots, spotSize, dmdMapping, xoffset, yoffset):
    N = Nspots
    I = np.zeros((rows, cols))

    # ortholinear grid
    # fill in spots in partitions of NxN
    # for row in range(0, rows, N):
    #     for col in range(0, cols, N):
    #         for spot_x in range(spotSize):
    #             for spot_y in range(spotSize):
    #                 # prevent index out of bounds
    #                 if row + yoffset + spot_y < rows and col + xoffset + spot_x < cols:
    #                     I[row + yoffset + spot_y, col + xoffset + spot_x] = 1

    # staggered grid
    for row in range(-2 * rows, 2 * rows, N):
        for col in range(-2 * cols, 2 * cols, N):
            for spot_x in range(spotSize):
                for spot_y in range(spotSize):
                    # prevent index out of bounds
                    # if row + yoffset + spot_y < rows and col + xoffset + spot_x < cols:
                    #     I[row + yoffset + spot_y, col + xoffset + spot_x] = 1

                    ip = row + yoffset + spot_y
                    jp = col + xoffset + spot_x

                    if dmdMapping == 1:
                        i = jp + ip - 2
                        j = (jp - ip + 4) // 2
                    else:
                        # ip = (i - 2 * j + 6) // 2
                        # jp = (2 * j + i + 1) // 2 - 1
                        # use tilted coordinates
                        i = ip
                        j = jp

                    if i < rows and j < cols and i >= 0 and j >= 0:
                        I[i, j] = 1

    return I


def SIMimages_spots(opt, DIo):
    # AIM: to generate raw sim images
    # INPUT VARIABLES
    #   k2: illumination frequency
    #   DIo: specimen image or integer (dimension) if only patterns are wanted
    #   PSFo: system PSF
    #   OTFo: system OTF
    #   UsePSF: 1 (to blur SIM images by convloving with PSF)
    #           0 (to blur SIM images by truncating its fourier content beyond OTF)
    #   NoiseLevel: percentage noise level for generating gaussian noise
    # OUTPUT VARIABLES
    #   frames:  raw sim images
    #   DIoTnoisy: noisy wide field image
    #   DIoT: noise-free wide field image

    if type(DIo) == int:
        w = DIo
    else:
        w = DIo.shape[0]

    X, Y = Get_X_Y_MeshGrids(w, opt)

    PSFo, OTFo = PsfOtf(w, opt)

    N = opt.Nspots

    # offsets depending on spot size
    offsets = [(x, y) for x in range(0, N, opt.spotSize) for y in range(0, N, opt.spotSize)]

    t0 = time.perf_counter()

    # illumination patterns
    frames = []
    for i_a in range(opt.Nframes):
        # illuminated signal

        sig = GenSpots(
            X.shape[0],
            X.shape[1],
            opt.Nspots,
            opt.spotSize,
            opt.dmdMapping,
            *offsets[i_a],
        )

        if not opt.patterns: # pure patterns for reference and DMD control
            sig = opt.meanInten + opt.ampInten * sig

        # modify sig if padding was added
        if opt.dmdMapping == 2:
            # rotate image by 45 degrees
            rotated_image = transform.rotate(sig, -45)

            rotated_image = img_as_float(rotated_image)

            rows, cols = rotated_image.shape[0], rotated_image.shape[1]

            # crop centre to avoid black corners
            row_start = rows // 4 + rows // 8
            row_end = row_start + rows // 4
            col_start = cols // 4 + cols // 8
            col_end = col_start + cols // 4

            # Crop the center of the image
            sig = rotated_image[row_start:row_end, col_start:col_end]

            # don't think this is needed
            # # clip to 0-1
            # sig = sig.clip(0, 1)
            # sig = img_as_ubyte(sig)

        if opt.patterns:
            frame = sig
        else:
            sup_sig = DIo * sig  # superposed signal

            # superposed (noise-free) Images
            if opt.UsePSF == 1:
                ST = conv2(sup_sig, PSFo, "same")
            else:
                ST = np.real(ifft2(fft2(sup_sig) * fftshift(OTFo)))

            # Noise generation
            if opt.usePoissonNoise:
                # Poisson
                vals = 2 ** np.ceil(
                    np.log2(opt.NoiseLevel)
                )  # NoiseLevel could be 200 for Poisson: degradation seems similar to Noiselevel 20 for Gaussian
                STnoisy = np.random.poisson(ST * vals) / float(vals)
            else:
                # Gaussian
                aNoise = opt.NoiseLevel / 100  # noise
                # SNR = 1/aNoise
                # SNRdb = 20*log10(1/aNoise)

                nST = np.random.normal(0, aNoise * np.std(ST, ddof=1), (ST.shape))
                NoiseFrac = 1  # may be set to 0 to avoid noise addition
                # noise added raw SIM images
                STnoisy = ST + NoiseFrac * nST


            frame = STnoisy.clip(0, 1)


        frames.append(frame)

    print(f"Time taken: {time.perf_counter() - t0}")
    return frames


def square_wave(x):
    return np.heaviside(np.cos(x), 0)
    # return np.where(np.cos(x) >= 0, 1, 0)


def square_wave_one_third(x):
    # sums to 0
    return 2 * (np.heaviside(np.cos(x) - np.cos(1 * np.pi / 3), 0) - 1 / 3)


def Generate_SIM_Image(opt, Io, in_dim=512, gt_dim=1024, func=np.cos):
    DIo = Io.astype("float")

    if in_dim is not None:
        if type(in_dim) is int:
            DIo = transform.resize(Io, (in_dim, in_dim), anti_aliasing=True, order=3)
        else:
            DIo = transform.resize(Io, in_dim, anti_aliasing=True, order=3)

    w = DIo.shape[0]

    # Generation of the PSF with Besselj.
    PSFo, OTFo = PsfOtf(w, opt)


    if opt.SIMmodality == "stripes":
        frames = SIMimages(opt, DIo, func=func)
    elif opt.SIMmodality == "spots":
        frames = SIMimages_spots(opt, DIo)
    elif opt.SIMmodality == "speckle":
        frames = SIMimages_speckle(opt, DIo)

    if opt.OTF_and_GT and not opt.patterns:
        frames.append(OTFo)

        if type(gt_dim) is int:
            gt_img = transform.resize(Io, (gt_dim, gt_dim), anti_aliasing=True, order=3)
        else:
            gt_img = transform.resize(Io, gt_dim, anti_aliasing=True, order=3)

        if gt_dim > in_dim:  # assumes a upscale factor of 2 is given
            # gt_img = skimage.transform.resize(gt_img, (gt_dim,gt_dim), order=3)
            gt11 = gt_img[:in_dim[0], :in_dim[1]]
            gt21 = gt_img[in_dim[0]:, :in_dim[1]]
            gt12 = gt_img[:in_dim[0], in_dim[1]:]
            gt22 = gt_img[in_dim[0]:, in_dim[1]:]
            # frames.extend([gt11,gt21,gt12,gt22])
            frames.append(gt11)
            frames.append(gt21)
            frames.append(gt12)
            frames.append(gt22)
        else:
            frames.append(gt_img)
    stack = np.array(frames)

    # NORMALIZE

    # does not work well with partitioned GT
    # for i in range(len(stack)):
    # stack[i] = (stack[i] - np.min(stack[i])) / \
    # (np.max(stack[i]) - np.min(stack[i]))

    # normalised SIM stack
    simstack = stack[: opt.Nframes]
    stack[: opt.Nframes] = (simstack - np.min(simstack)) / ( np.max(simstack) - np.min(simstack))

    # normalised gt
    if gt_dim > in_dim:
        gtstack = stack[-4:]
        stack[-4:] = (gtstack - np.min(gtstack)) / (np.max(gtstack) - np.min(gtstack))
        # normalised OTF
        stack[-5] = (stack[-5] - np.min(stack[-5])) / (
            np.max(stack[-5] - np.min(stack[-5]))
        )
    else:
        stack[-1] = (stack[-1] - np.min(stack[-1])) / (
            np.max(stack[-1] - np.min(stack[-1]))
        )
        # normalised OTF
        stack[-2] = (stack[-2] - np.min(stack[-2])) / (
            np.max(stack[-2] - np.min(stack[-2]))
        )

    stack = (stack * 255).astype("uint8")

    if opt.outputname is not None:
        io.imsave(opt.outputname, stack)

    return stack
