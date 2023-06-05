import numpy as np
from numpy import pi, cos, sin
from numpy.fft import fft2, ifft2, fftshift, ifftshift

from skimage import io, draw
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

    x = np.linspace(0, w-1, w)
    y = np.linspace(0, w-1, w)
    X, Y = np.meshgrid(x, y)

    # Generation of the PSF with Besselj.
    R = np.sqrt(np.minimum(X, np.abs(X-w))**2+np.minimum(Y, np.abs(Y-w))**2)
    yy = np.abs(2*scipy.special.jv(1, scale*R+eps) / (scale*R+eps)
                )**2  # 0.5 is introduced to make PSF wider
    yy0 = fftshift(yy)

    # Generate 2D OTF.
    OTF2d = fft2(yy)
    OTF2dmax = np.max([np.abs(OTF2d)])
    OTF2d = OTF2d/OTF2dmax
    OTF2dc = np.abs(fftshift(OTF2d))

    return (yy0, OTF2dc)


def conv2(x, y, mode='same'):
    # Make it equivalent to Matlab's conv2 function
    # https://stackoverflow.com/questions/3731093/is-there-a-python-equivalent-of-matlabs-conv2-function
    return np.rot90(convolve2d(np.rot90(x, 2), np.rot90(y, 2), mode=mode), 2)

def GenSpeckle(dim,opt):
    N = opt.Nspeckles
    I = np.zeros((dim,dim))
    randx = np.random.choice(list(range(dim))*np.ceil(N/dim).astype('int'),size=N,replace=False)
    randy = np.random.choice(list(range(dim))*np.ceil(N/dim).astype('int'),size=N,replace=False)
    
    for i in range(N):
        x = randx[i]
        y = randy[i]
        
        r = np.random.randint(3,5)
        cr,cc = draw.circle(x,y,r,(dim,dim))
        I[cr,cc] += 0.1    
    return I

def SIMimages(opt, DIo, PSFo, OTFo):

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
    wo = w/2
    x = np.linspace(0, w-1, w)
    y = np.linspace(0, w-1, w)
    [X, Y] = np.meshgrid(x, y)

    # illumination patterns
    frames = []
    for i_a in range(opt.Nframes):
        # illuminated signal
        sig = GenSpeckle(w,opt) # opt.meanInten[i_a] + opt.ampInten[i_a] * GenSpeckle(w)

        sup_sig = DIo*sig  # superposed signal

        # superposed (noise-free) Images
        if opt.UsePSF == 1:
            ST = conv2(sup_sig, PSFo, 'same')
        else:
            ST = np.real(ifft2(fft2(sup_sig)*fftshift(OTFo)))

        # Gaussian noise generation
        aNoise = opt.NoiseLevel/100  # noise
        # SNR = 1/aNoise
        # SNRdb = 20*log10(1/aNoise)

        nST = np.random.normal(0, aNoise*np.std(ST, ddof=1), (w, w))
        NoiseFrac = 1  # may be set to 0 to avoid noise addition
        # noise added raw SIM images
        STnoisy = ST + NoiseFrac*nST
        frames.append(STnoisy)

    return frames


# %%
def Generate_SIM_Image(opt, Io):

    w = Io.shape[0]

    # Generation of the PSF with Besselj.

    PSFo, OTFo = PsfOtf(w, opt.scale)

    DIo = Io.astype('float')

    frames = SIMimages(opt, DIo, PSFo, OTFo)

    if opt.OTF_and_GT:
        frames.append(OTFo)
        frames.append(Io)
    stack = np.array(frames)

    # normalise
    for i in range(len(stack)):
        stack[i] = (stack[i] - np.min(stack[i])) / \
            (np.max(stack[i]) - np.min(stack[i]))

    stack = (stack * 255).astype('uint8')

    if opt.outputname is not None:
        io.imsave(opt.outputname, stack)

    return stack
