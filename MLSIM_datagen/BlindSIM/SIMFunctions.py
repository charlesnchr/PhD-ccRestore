
import numpy as np 
import math

def calcPSF(xysize,pixelSize,NA,emission,rindexObj,rindexSp,depth):
    """
    Generate the aberrated emission PSF using P.A.Stokseth model.
    
    Parameters
    ----------
    xysize: number of pixels

    pixelSize: size of pixels (in nm)

    emission: emission wavelength (in nm)

    rindexObj: refractive index of objective lens

    rindexSp: refractive index of the sample

    depth: imaging height above coverslip (in nm)
    
    Returns
    -------
    psf:
        2D array of PSF normalised between 0 and 1
    
    References
    ----------
    [1] P. A. Stokseth (1969), "Properties of a defocused optical system," J. Opt. Soc. Am. A 59:1314-1321. 
    """

    #Calculated the wavelength of light inside the objective lens and specimen
    lambdaObj = emission/rindexObj
    lambdaSp = emission/rindexSp

    #Calculate the wave vectors in vaccuum, objective and specimens
    k0 = 2*np.pi/emission
    kObj = 2*np.pi/lambdaObj
    kSp = 2*np.pi/lambdaSp

    #pixel size in frequency space
    dkxy = 2*np.pi/(pixelSize*xysize)

    #Radius of pupil
    kMax = (2*np.pi*NA)/(emission*dkxy)

    klims = np.linspace(-xysize/2,xysize/2,xysize)
    kx, ky = np.meshgrid(klims,klims)
    k = np.hypot(kx,ky)
    pupil = k
    pupil[pupil<kMax]=1
    pupil[pupil>=kMax]=0

    #sin of objective semi-angle
    sinthetaObj = (k*(dkxy))/kObj
    sinthetaObj[sinthetaObj>1] = 1

    #cosin of objective semi-angle
    costhetaObj = np.finfo(float).eps+np.sqrt(1-(sinthetaObj**2))

    #sin of sample semi-angle
    sinthetaSp = (k*(dkxy))/kSp
    sinthetaSp[sinthetaSp>1] = 1

    #cosin of sample semi-angle
    costhetaSp = np.finfo(float).eps+np.sqrt(1-(sinthetaSp**2))

    #Spherical aberration phase calculation
    phisa = (1j*k0*depth)*((rindexSp*costhetaSp)-(rindexObj*costhetaObj))
    #Calculate the optical path difference due to spherical aberrations
    OPDSA = np.exp(phisa)

    #apodize the emission pupil
    pupil = (pupil/np.sqrt(costhetaObj))


    #calculate the spherically aberrated pupil
    pupilSA = pupil*OPDSA

    #calculate the coherent PSF
    psf = np.fft.ifft2(pupilSA)

    #calculate the incoherent PSF
    psf = np.fft.fftshift(abs(psf)**2)
    psf = psf/np.amax(psf)

    return psf

def edgeTaper(I,PSF):
    """
    Taper the edge of an image with the provided point-spread function. This 
    helps to remove edging artefacts when performing deconvolution operations in 
    frequency space. The output is a weighted sum of a blurred and original version of
    the image with the weighting matrix determined in terms of the tapering PSF
    
    Parameters
    ----------
    I: Image to be tapered

    PSF: Point-spread function to be used for taper

        
    Returns
    -------
    tapered: Image with tapered edges
    
    """

    PSFproj=np.sum(PSF, axis=0) # Calculate the 1D projection of the PSF
    # Generate 2 1D arrays with the tapered PSF at the leading edge
    beta1 = np.pad(PSFproj,(0,(I.shape[1]-1-PSFproj.shape[0])),'constant',constant_values=(0))
    beta2 = np.pad(PSFproj,(0,(I.shape[0]-1-PSFproj.shape[0])),'constant',constant_values=(0))
    
    # In frequency space replicate the tapered edge at both ends of each 1D array
    z1 = np.fft.fftn(beta1) # 1D Fourier transform 
    z1 = abs(np.multiply(z1,z1)) # Absolute value of the square of the Fourier transform
    z1=np.real(np.fft.ifftn(z1)) # Real value of the inverse Fourier transform
    z1 = np.append(z1,z1[0]) # Ensure the edges of the matrix are symetric 
    z1 = 1-(z1/np.amax(z1)) # Normalise

    z2 = np.fft.fftn(beta2)
    z2 = abs(np.multiply(z2,z2))
    z2=np.real(np.fft.ifftn(z2))
    z2 = np.append(z2,z2[0])
    z2 = 1-(z2/np.amax(z2))

    # Use matrix multiplication to generate a 2D edge filter
    q=np.matmul(z2[:,None],z1[None,:])

    # Generate a blured version of the image
    padx = int(np.floor((I.shape[0]-PSF.shape[0])/2))
    pady = int(np.floor((I.shape[1]-PSF.shape[1])/2))
    PSFbig =np.pad(PSF, ((padx,padx),(pady,pady)), 'constant', constant_values=0)
    PSFbig = np.resize(PSFbig,(I.shape[0],I.shape[1]))
    OTF = np.real(np.fft.fft2(PSFbig))
    Iblur = np.multiply(np.fft.fft2(I),OTF)
    Iblur = np.real(np.fft.ifft2(Iblur))

    #calculate the tapered image as the weighted sum of the blured and raw image
    tapered = np.multiply(I,q)+np.multiply((1-q),Iblur)
    Imax = np.amax(I)
    Imin = np.amin(I)

    # Bound the output by the min and max values of the oroginal image
    tapered[tapered < Imin] = Imin
    tapered[tapered > Imax] = Imax

    return tapered

def drawGauss (std,width):
    """
    Generate a 2D Gaussian. Output is always square
    
    Parameters
    ----------
    std: Standard deviation of gaussian

    width: Width of square output

    Returns
    -------
    arg: Square array with 2D Gaussian function centred about the middle
    """

    width = np.linspace(-width/2,width/2,width) # Genate array of values around centre
    kx, ky = np.meshgrid(width,width) # Generate square arrays with co-ordinates about centre
    kx = np.multiply(kx,kx) # Calculate 2D Gaussian function
    ky = np.multiply(ky,ky)
    arg = np.add(kx,ky)
    arg = np.exp(arg/(2*std*std))
    arg = arg/np.sum(arg)

    return arg;

def quasiWiener (OTF, image,factor):
    """
    Perform 2D Wiener deconvolution of an image. The OTF must
    match the dimensions of the image to be filtered.
    
    Parameters
    ----------
    OTF: Optical transfer function of imaging system

    images: Image input

    factor: Wiener factor
    
    Returns
    -------
    image: filtered image

    """
     # Take the absolute values of the OTF
    OTFconj = np.conj(OTF) # Calculate the complex conjugate of the OTF
    OTF = abs(OTF)
   
    FT = np.fft.fftshift(np.fft.fft2(image)) # Fourier transform the image
    filtered = np.multiply(FT,OTFconj) # Element-wise multiplication with the cojugate OTF
    filtered = np.divide(filtered,(OTF+factor)) # Element-wise division by the OTF
    image = np.fft.ifft2(np.fft.ifftshift(filtered)) # Inverse Fourier transform to real space

    return image;

def estimateFrequency(sepFT,suppress_noise_factor,fmask,cutoff):

    dims = np.shape(sepFT)
    norm_ft = np.zeros(dims,dtype=complex)
    im = norm_ft
    re_f = norm_ft
    reference = np.zeros((dims[0],dims[1],dims[2]),dtype=complex)

    for a in range(dims[2]): # Iterate over the angles
        for p in range(dims[3]): # Iterate over the phases
            ft_max = np.amax(abs(sepFT[:,:,a,p]))
            sepFT[:,:,a,p] = sepFT[:,:,a,p]/ft_max
            norm_ft[:,:,a,p] = sepFT[:,:,a,p]//(abs(sepFT[:,:,a,p])+suppress_noise_factor)
            im[:,:,a,p] = np.fft.fftshift(np.fft.fft2(norm_ft[:,:,a,p]))
        
        temp = np.fft.fft2(np.fft.ifftshift(norm_ft[:,:,a,0]))
        reference[:,:,a] = np.fft.fftshift(temp)

    for a in range(dims[2]):
        for p in range(dims[3]):
            temp = np.multiply(np.conj(reference[:,:,a]),im[:,:,a,p])
            re_f[:,:,a,p] = np.fft.ifft2(np.fft.ifftshift(temp))

    shiftvalue = np.zeros((dims[2],3,2)) #Rough estimate of modulation frequency
    for a in range(dims[2]):
        for p in range(dims[3]):
            if a == 0:
                temp = abs(re_f[:,:,a,p])
                ind = np.unravel_index(np.argmax(temp, axis=None), temp.shape)
                shiftvalue[a,p,0] = ind[0]
                shiftvalue[a,p,1] = ind[1]
            else:
                temp = np.multiply(fmask,abs(re_f[:,:,a,p]))
                ind = np.unravel_index(np.argmax(temp, axis=None), temp.shape)
                shiftvalue[a,p,0] = ind[0]
                shiftvalue[a,p,1] = ind[1]

    mask = np.zeros((dims[0],dims[1]))
    background = np.zeros((dims[2],dims[3]),dtype=complex)
    centre_value = np.zeros((dims[2],dims[3]),dtype=complex)

    for a in range(dims[2]):
        for p in range(dims[3]):
            xx = int(shiftvalue[a,p,0])
            yy = int(shiftvalue[a,p,1])
            centre_value[a,p] = abs(re_f[xx,yy,a,p])
            mask[xx-3:xx+3,yy-3:yy+3] = 1
            mask[xx-1:xx+1,yy-1:yy+1] = 0
            temp = np.multiply(abs(re_f[:,:,a,p]),mask)
            background[a,p] = np.sum(temp)/40

            mask[xx-3:xx+3,yy-3:yy+3] = 0

    background = background   

    for a in range(a):
        for p in range(2):
            if background[a,p+1]>0.5:
                background[a,p+1] = 1.7*background[a,p+1]



    return shiftvalue;
