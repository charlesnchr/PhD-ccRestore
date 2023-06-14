import numpy as np
from numpy import pi, cos, sin
from numpy.fft import fft2, ifft2, fftshift, ifftshift

from skimage import io, draw, transform
from scipy.signal import convolve2d
import scipy.special


def Get_X_Y_MeshGrids(w, opt):
    # TODO: these hard-coded values are not ideal
    #  and this way of scaling the patterns is
    #  likely going to lead to undesired behaviour

    if opt.crop_factor:
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

        x = np.linspace(0, crop_factor_x * 512 - 1, int(crop_factor_x * 912))
        y = np.linspace(0, crop_factor_y * 512 - 1, int(crop_factor_y * 1140))
        [X, Y] = np.meshgrid(x, y)
    else:
        x = np.linspace(0, w - 1, w)
        y = np.linspace(0, w - 1, w)
        X, Y = np.meshgrid(x, y)

    return X, Y


def PsfOtf(w, scale, opt):
    # AIM: To generate PSF and OTF using Bessel function
    # INPUT VARIABLES
    #   w: image size
    #   scale: a parameter used to adjust PSF/OTF width
    # OUTPUT VRAIBLES
    #   yyo: system PSF
    #   OTF2dc: system OTF
    eps = np.finfo(np.float64).eps

    X, Y = Get_X_Y_MeshGrids(w, opt)

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


def SIMimages(opt, DIo, PSFo, OTFo, func=np.cos, pixelsize_ratio=1):
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
        patterns = True
        w = DIo
        wo = w / 2
    else:
        patterns = False
        w = DIo.shape[0]
        wo = w / 2

    X, Y = Get_X_Y_MeshGrids(w, opt)

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

            if patterns:
                frames.append(sig)
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

                frames.append(STnoisy.clip(0, 1))

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


def SIMimages_speckle(opt, DIo, PSFo, OTFo):
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
    wo = w / 2
    X, Y = Get_X_Y_MeshGrids(w, opt)

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


def GenSpots(dim, opt, xoffset, yoffset):
    N = opt.Nspots
    spotSize = opt.spotSize
    I = np.zeros((dim, dim))

    # fill in spots in partitions of NxN
    for row in range(0, dim - N, N):
        for col in range(0, dim - N, N):
            for spot_x in range(spotSize):
                for spot_y in range(spotSize):
                    # prevent index out of bounds
                    if row + xoffset + spot_x < dim and col + yoffset + spot_y < dim:
                        I[row + xoffset + spot_x, col + yoffset + spot_y] = 1
    return I


def SIMimages_spots(opt, DIo, PSFo, OTFo):
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
        patterns = True
        w = DIo
        wo = w / 2
    else:
        patterns = False
        w = DIo.shape[0]
        wo = w / 2

    X, Y = Get_X_Y_MeshGrids(w, opt)

    N = opt.Nspots
    offsets = [(x, y) for x in range(0, N) for y in range(0, N)]

    # illumination patterns
    frames = []
    for i_a in range(opt.Nframes):
        # illuminated signal
        sig = GenSpots(w, opt, *offsets[i_a])

        if patterns:
            frames.append(sig)
        else:
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

    PSFo, OTFo = PsfOtf(w, opt.scale, opt)

    frames = SIMimages(opt, DIo, PSFo, OTFo, func=func)

    if opt.OTF_and_GT:
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
    simstack = stack[: opt.Nangles * opt.Nshifts]
    stack[: opt.Nangles * opt.Nshifts] = (simstack - np.min(simstack)) / (
        np.max(simstack) - np.min(simstack)
    )

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
