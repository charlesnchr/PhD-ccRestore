
import numpy as np
from numpy import pi,cos,sin
from numpy.fft import fft2,ifft2,fftshift

import skimage
from scipy.signal import convolve2d
import scipy.special
import scipy.optimize
from SIMFunctions import edgeTaper


def Reconstruct_Image(I, outputname=None):

    def TripletSNR0(OBJpara,k2fa,OTFo,fDIp):

        # AIM: To obtain signal spectrums corresponding to central and off-center
        #       frequency components
        # INPUT VARIABLES
        #   OBJpara: object power parameters
        #   k2fa: illumination frequency vector
        #   OTFo: system OTF
        #   fDIp: one of the off-center frequency component (utilized here only for
        #           visual verification of computation)
        # OUTPUT VARIABLES
        #   SIGao: signal spectrum corresponding to central frequency component
        #   SIGap2,SIGam2: signal spectrums corresponding to off-center frequency components


        w = OTFo.shape[0]
        wo = w//2
        x = np.linspace(0,w-1,w)
        y = np.linspace(0,w-1,w)
        X,Y = np.meshgrid(x,y)
        Cv = (X-wo) + 1j*(Y-wo)
        Ro = np.abs(Cv)
        Ro[wo,wo] = 1 # to avoid nan

        # object spectrum parameters
        Aobj = OBJpara[0]
        Bobj = OBJpara[1]

        # object spectrums (default)
        kv = k2fa[1] + 1j*k2fa[0] # vector along illumination direction
        Rp = np.abs(Cv-kv)
        Rm = np.abs(Cv+kv)
        OBJo = Aobj*(Ro**Bobj)
        OBJp = Aobj*(Rp**Bobj)
        OBJm = Aobj*(Rm**Bobj)

        # FIX: pythonise indixces?
        OBJo[wo+0,wo+0] = 0.25*OBJo[wo+1,wo+0] + 0.25*OBJo[wo+0,wo+1] \
        + 0.25*OBJo[wo-1,wo+0] + 0.25*OBJo[wo+0,wo-1]

        k3 = np.round(k2fa)
        OBJp[int(wo+0+k3[0]),int(wo+0+k3[1])] = 0.25*OBJp[int(wo+1+k3[0]),int(wo+0+k3[1])] \
            + 0.25*OBJp[int(wo+0+k3[0]),int(wo+1+k3[1])] \
            + 0.25*OBJp[int(wo-1+k3[0]),int(wo+0+k3[1])] \
            + 0.25*OBJp[int(wo+0+k3[0]),int(wo-1+k3[1])] 
        OBJm[int(wo+0-k3[0]),int(wo+0-k3[1])] = 0.25*OBJm[int(wo+1-k3[0]),int(wo+0-k3[1])] \
            + 0.25*OBJm[int(wo+0-k3[0]),int(wo+1-k3[1])] \
            + 0.25*OBJm[int(wo-1-k3[0]),int(wo+0-k3[1])] \
            + 0.25*OBJm[int(wo+0-k3[0]),int(wo-1-k3[1])]

        # signal spectrums
        SIGao = OBJo*OTFo
        SIGap = OBJp*OTFo
        SIGam = OBJm*OTFo

        # FIX: is roll equivalenet to circshift ??
        k3 = k3.astype('int')
        SIGap2 = np.roll(SIGap,(-k3[0],-k3[1]), (0, 1))
        SIGam2 = np.roll(SIGam,(k3[0],k3[1]), (0, 1))

        return SIGao,SIGap2,SIGam2

    def MergingHeptaletsF(fDIo,fDIp,fDIm,fDBo,fDBp,fDBm,
        fDCo,fDCp,fDCm,Ma,Mb,Mc,npDIo,npDIp,npDIm,
        npDBo,npDBp,npDBm,npDCo,npDCp,npDCm,k2fa,k2fb,k2fc,OBJpara,OTFo):

        # AIM: To merge all 9 frequency components into one using generalized
        #       Wiener Filter
        # INPUT VARIABLES
        #   [fDIo fDIp fDIm
        #    fDBo fDBp fDBm
        #    fDCo fDCp fDCm]: nine frequency components
        #   Ma,Mb,Mc: modulation factors for the three illumination orientations
        #   [npDIo npDIp npDIm 
        #    npDBo npDBp npDBm 
        #    npDCo,npDCp,npDCm]: noise powers corresponding to nine frequency
        #                       components
        #   k2fa,k2fb,k2fc: illumination frequency vectors for the three
        #                   illumination orientations
        #   OBJpara: Object spectrum parameters
        #   OTFo: system OTF
        # OUTPUT VARIABLES
        #   Fsum: all nine frequency components merged into one using 
        #           generalised Wiener Filter
        #   Fperi: six off-center frequency components merged into one using 
        #           generalised Wiener Filter
        #   Fcent: averaged of the three central frequency components


        # obtain signal spectrums corresponding to central and off-center
        # frequency components
        SIGao,SIGam2,SIGap2 = TripletSNR0(OBJpara,k2fa,OTFo,fDIm)
        SIGbo,SIGbm2,SIGbp2 = TripletSNR0(OBJpara,k2fb,OTFo,fDBm)
        SIGco,SIGcm2,SIGcp2 = TripletSNR0(OBJpara,k2fc,OTFo,fDCm)

        SIGap2 = Ma*SIGap2
        SIGam2 = Ma*SIGam2
        SIGbp2 = Mb*SIGbp2
        SIGbm2 = Mb*SIGbm2
        SIGcp2 = Mc*SIGcp2
        SIGcm2 = Mc*SIGcm2

        ## Generalized Wiener-Filter computation
        SNRao = SIGao*SIGao.conj()/npDIo
        SNRap = SIGap2*SIGap2.conj()/npDIp
        SNRam = SIGam2*SIGam2.conj()/npDIm

        SNRbo = SIGbo*SIGbo.conj()/npDBo
        SNRbp = SIGbp2*SIGbp2.conj()/npDBp
        SNRbm = SIGbm2*SIGbm2.conj()/npDBm

        SNRco = SIGco*SIGco.conj()/npDCo
        SNRcp = SIGcp2*SIGcp2.conj()/npDCp
        SNRcm = SIGcm2*SIGcm2.conj()/npDCm

        ComDeno = 0.01 + ( SNRao + SNRap + SNRam + SNRbo + SNRbp + SNRbm + SNRco + SNRcp + SNRcm )
        Fsum = fDIo*SNRao + fDIp*SNRap + fDIm*SNRam \
            + fDBo*SNRbo + fDBp*SNRbp + fDBm*SNRbm \
            + fDCo*SNRco + fDCp*SNRcp + fDCm*SNRcm
        Fsum = Fsum/ComDeno

        ComPeri = 0.01 + ( SNRap + SNRam + SNRbp + SNRbm + SNRcp + SNRcm )
        Fperi = fDIp*SNRap + fDIm*SNRam + fDBp*SNRbp + fDBm*SNRbm \
            + fDCp*SNRcp + fDCm*SNRcm
        Fperi = Fperi/ComPeri

        # averaged central frequency component
        Fcent = (fDIo+fDBo+fDCo)/3

        return Fsum,Fperi,Fcent





    def OTFedgeF(OTFo):

        w = OTFo.shape[0]
        wo = w//2

        OTF1 = OTFo[wo,:]
        OTFmax = np.max(np.abs(OTFo))
        OTFtruncate = 0.01
        i = 1
        while np.abs(OTF1[i])<OTFtruncate*OTFmax:
            Kotf = wo+1-i
            i = i + 1

        return Kotf


    def OBJparaOpt(OBJpara0,fDIoTnoisy,OTFo):
        # AIM: Determined Sum of Squared Errors (SSE) between `actual signal power' and
        #   `approximated signal power'
        # INPUT VARIABLES
        #   OBJpara0: [Aobj Bobj], Object power parameters
        #   fDIoTnoisy: FT of central frequency component
        #   OTFo: system OTF
        # OUTPUT VARIABLE
        #   Esum: SSE between `actual signal spectrum' and `approximated signal spectrum'

        w = OTFo.shape[0]
        wo = w//2
        x = np.linspace(0,w-1,w)
        y = np.linspace(0,w-1,w)
        [X,Y] = np.meshgrid(x,y)
        Cv = (X-wo) + 1j*(Y-wo)
        Ro = np.abs(Cv)
        Ro[wo,wo] = 1 # to avoid nan

        # approximated signal power calculation
        Aobj = OBJpara0[0]
        Bobj = OBJpara0[1]
        OBJpower = Aobj*(Ro**Bobj)
        SIGpower = OBJpower*OTFo

        # OTF cut-off frequency
        Kotf = OTFedgeF(OTFo)

        # range of frequency over which SSE is computed
        Zloop = (Ro<0.75*Kotf)*(Ro>0.25*Kotf)

        # frequency beyond which NoisePower estimate to be computed
        NoiseFreq = Kotf + 20

        # NoisePower determination
        Zo = Ro>NoiseFreq
        nNoise = fDIoTnoisy*Zo
        NoisePower = np.sum( nNoise*nNoise.conj() )/np.sum(Zo)

        # Noise free object power computation 
        Fpower = fDIoTnoisy*fDIoTnoisy.conj() - NoisePower
        fDIoTnoisy = np.sqrt(np.abs(Fpower))

        # SSE computation
        Error = fDIoTnoisy - SIGpower
        Esum = np.sum((Error**2/Ro)*Zloop)

        return Esum


    def PhaseKai2opt(k2fa,fS1aTnoisy,OTFo,OPT):
        # Aim: Compute autocorrelation of FT of raw SIM images
        #   k2fa: illumination frequency vector
        #   fS1aTnoisy: FT of raw SIM image
        #   OTFo: system OTF
        #   OPT: acronym for `OPTIMIZE' to be set to 1 when this function is used
        #       for optimization, or else to 0
        #   CCop: autocorrelation of fS1aTnoisy

        w = fS1aTnoisy.shape[0]
        wo = w//2

        fS1aTnoisy = fS1aTnoisy*(1-1*OTFo**10)
        fS1aT = fS1aTnoisy*OTFo.conj()

        Kotf = OTFedgeF(OTFo)
        DoubleMatSize = 0 

        if 2*Kotf > wo:
            DoubleMatSize = 1 # 1 for doubling fourier domain size, 0 for keeping it unchanged
        if DoubleMatSize>0:
            t = 2*w
            fS1aT_temp = np.zeros((t,t),dtype=np.complex128)
            fS1aT_temp[wo:w+wo,wo:w+wo] = fS1aT
            fS1aT = fS1aT_temp
        else:
            t = w

        to = t/2
        u = np.linspace(0,t-1,t)
        v = np.linspace(0,t-1,t)
        [U,V] = np.meshgrid(u,v)
        S1aT = np.exp( -1j*2*pi*( k2fa[1]/t*(U-to)+k2fa[0]/t*(V-to) ) )*ifft2(fS1aT)
        fS1aT0 = fft2( S1aT )

        mA = np.sum( fS1aT*fS1aT0.conj() )
        mA = mA/np.sum( fS1aT0*fS1aT0.conj() )

        if OPT > 0:
            CCop = -np.abs(mA)
        else:
            CCop = mA    

        return CCop


    def ApproxFreqDuplex(FiSMap,Kotf):
        # AIM: approx. illumination frequency vector determination
        # INPUT VARIABLES
        #   FiSMap: FT of raw SIM image
        #   Kotf: OTF cut-off frequency
        # OUTPUT VARIABLES
        #   maxK2: illumination frequency vector (approx)
        #   Ix,Iy: coordinates of illumination frequency peaks

        FiSMap = np.abs(FiSMap)

        w = FiSMap.shape[0]
        wo = w//2
        x = np.linspace(0,w-1,w)
        y = np.linspace(0,w-1,w)
        [X,Y] = np.meshgrid(x,y)

        Ro = np.sqrt( (X-wo)**2 + (Y-wo)**2 )
        Z0 = Ro > np.round(0.5*Kotf)
        Z1 = X > wo

        FiSMap = FiSMap*Z0*Z1
        dumY = np.max( FiSMap,0 )  # possible pitfal ? syntax is different
        Iy = np.argmax(dumY)
        dumX = np.max( FiSMap,1 )
        Ix = np.argmax(dumX)

        maxK2 = [Ix-wo, Iy-wo]

        return maxK2



    def IlluminationFreqF(S1aTnoisy,OTFo):
        # AIM: illumination frequency vector determination
        # INPUT VARIABLES
        #   S1aTnoisy: raw SIM image
        #   OTFo: system OTF
        # OUTPUT VARIABLE
        #   k2fa: illumination frequency vector

        w = OTFo.shape[0]
        wo = w//2

        # computing PSFe for edge tapering SIM images
        PSFd = np.real(fftshift( ifft2(fftshift(OTFo**10)) ))
        PSFd = PSFd/np.max(PSFd)
        PSFd = PSFd/np.sum(PSFd)
        h = 30
        PSFe = PSFd[int(wo-h):int(wo+h),int(wo-h):int(wo+h)]

        # edge tapering raw SIM image
        S1aTnoisy_et = edgeTaper(S1aTnoisy,PSFe)
        # S1aTnoisy_et = S1aTnoisy
        fS1aTnoisy_et = fftshift(fft2(S1aTnoisy_et))

        # OTF cut-off freq
        Kotf = OTFedgeF(OTFo)

        # Approx illumination frequency vector
        k2fa = ApproxFreqDuplex(fS1aTnoisy_et,Kotf)

        fS1aTnoisy = fftshift(fft2(S1aTnoisy))
        # illumination frequency vector determination by optimizing
        # autocorrelation of fS1aTnoisy
        OPT = 1
        PhaseKai2opt0 = lambda k2fa0: PhaseKai2opt(k2fa0,fS1aTnoisy,OTFo,OPT)

        k2fa0 = k2fa
        k2fa = scipy.optimize.fmin_bfgs(PhaseKai2opt0, k2fa0, maxiter=200)
        # k2a = sqrt(k2fa*k2fa')

        return k2fa

    def PatternPhaseOpt(phaseA,S1aTnoisy,k2fa):

        w = S1aTnoisy.shape[0]
        wo = w//2

        x = np.linspace(0,w-1,w)
        y = np.linspace(0,w-1,w)
        X,Y = np.meshgrid(x,y)

        sAo = cos( 2*pi*(k2fa[1]*(X-wo)+k2fa[0]*(Y-wo))/w + phaseA )
        S1aTnoisy = S1aTnoisy - np.mean(S1aTnoisy)
        CCop = -np.sum(S1aTnoisy*sAo)

        return CCop

    def IlluminationPhaseF(S1aTnoisy,k2fa):
        # AIM: illumination phase shift determination
        # INPUT VARIABLES
        #   S1aTnoisy: raw SIM image
        #   k2fa: illumination frequency vector
        # OUTPUT VARIABLE
        #   phaseA1: illumination phase shift determined

        PatternPhaseOpt0 = lambda phaseA0: PatternPhaseOpt(phaseA0,S1aTnoisy,k2fa)
        phaseA0 = 0
        phaseA1 = scipy.optimize.fmin_bfgs(PatternPhaseOpt0,  phaseA0, maxiter=200)
        return phaseA1

    def SeparatedComponents2D(phaseShift,phaseShift0,FcS1aT,FcS2aT,FcS3aT):
        # Aim: Unmixing the frequency components of raw SIM images
        #   phaseShift,phaseShift0: illumination phase shifts
        #   FcS1aT,FcS2aT,FcS3aT: FT of raw SIM images
        #   FiSMao,FiSMap,FiSMam: unmixed frequency components of raw SIM images

        phaseShift1 = phaseShift[0]
        phaseShift2 = phaseShift[1]
        MF = 1.0
        ## Transformation Matrix
        M = 0.5*np.array([[1+0j, 0.5*MF*np.exp(-1j*phaseShift0), 0.5*MF*np.exp(+1j*phaseShift0)],
                [1+0j, 0.5*MF*np.exp(-1j*phaseShift1), 0.5*MF*np.exp(+1j*phaseShift1)],
                [1+0j, 0.5*MF*np.exp(-1j*phaseShift2), 0.5*MF*np.exp(+1j*phaseShift2)]]).astype('complex128')

        ## Separting the components
        #===========================================================
        Minv = np.linalg.inv(M)

        FiSMao = Minv[0,0]*FcS1aT + Minv[0,1]*FcS2aT + Minv[0,2]*FcS3aT
        FiSMap = Minv[1,0]*FcS1aT + Minv[1,1]*FcS2aT + Minv[1,2]*FcS3aT
        FiSMam = Minv[2,0]*FcS1aT + Minv[2,1]*FcS2aT + Minv[2,2]*FcS3aT

        return FiSMao,FiSMap,FiSMam



    ##
    def PCMseparateF(S1aTnoisy,S2aTnoisy,S3aTnoisy,OTFo):

        # Determination of illumination frequency vectors
        k1a = IlluminationFreqF(S1aTnoisy,OTFo)
        k2a = IlluminationFreqF(S2aTnoisy,OTFo)
        k3a = IlluminationFreqF(S3aTnoisy,OTFo)

        # mean illumination frequency vector
        kA = (k1a + k2a + k3a)/3

        # determination of illumination phase shifts
        phase1A = IlluminationPhaseF(S1aTnoisy,kA)
        phase2A = IlluminationPhaseF(S2aTnoisy,kA)
        phase3A = IlluminationPhaseF(S3aTnoisy,kA)

        # # edge tapering raw SIM images
        S1aTnoisy = edgeTaper(S1aTnoisy,PSFe)
        S2aTnoisy = edgeTaper(S2aTnoisy,PSFe)
        S3aTnoisy = edgeTaper(S3aTnoisy,PSFe)

        fS1aTnoisy = fftshift(fft2(S1aTnoisy))
        fS2aTnoisy = fftshift(fft2(S2aTnoisy))
        fS3aTnoisy = fftshift(fft2(S3aTnoisy))

        # Separating the three frequency components
        phaseShift0 = phase1A
        phaseShift = [phase2A, phase3A]
        fAo,fAp,fAm = SeparatedComponents2D(phaseShift,phaseShift0,fS1aTnoisy,fS2aTnoisy,fS3aTnoisy)

        return fAo,fAp,fAm,kA


    def OTFdoubling(OTFo,DoubleMatSize):

        ## embeds OTFo in doubled range of frequencies  

        w = OTFo.shape[0]
        wo = w/2
        if DoubleMatSize>0:
            t = 2*w
            u = np.linspace(0,t-1,t)
            v = np.linspace(0,t-1,t)
            OTF2 = np.zeros((2*w,2*w))
            OTF2[int(wo):int(w+wo),int(wo):int(w+wo)] = OTFo
        else:
            OTF2 = OTFo

        return OTF2

    ##
    def PCMfilteringF(fDo,fDp,fDm,OTFo,OBJparaA,kA):

        # AIM: obtaining Wiener Filtered estimates of noisy frequency components
        # INPUT VARIABLES
        #   fDo,fDp,fDm: noisy estimates of separated frequency components
        #   OTFo: system OTF
        #   OBJparaA: object power parameters
        #   kA: illumination vector
        # OUTPUT VARIABLES
        #   fDof,fDp2,fDm2: Wiener Filtered estimates of fDo,fDp,fDm
        #   npDo,npDp,npDm: avg. noise power in fDo,fDp,fDm
        #   Mm: illumination modulation factor
        #   DoubleMatSize: parameter for doubling FT size if necessary

        # object power parameters
        Aobj = OBJparaA[0]
        Bobj = OBJparaA[1]

        ## Wiener Filtering central frequency component
        # fDof,npDo] = WoFilterCenter(fDo,OTFo,co,OBJparaA,SFo)

        SFo = 1.0
        co = 1.0 

        OTFpower = OTFo*OTFo.conj()

        # frequency beyond which NoisePower estimate to be computed
        NoiseFreq = Kotf + 20

        # NoisePower determination
        Zo = Ro>NoiseFreq
        nNoise = fDo*Zo
        NoisePower = np.sum( nNoise*nNoise.conj() )/np.sum(Zo)

        # Object Power determination
        Ro[wo,wo] = 1  # to avoid nan
        OBJpower = Aobj*(Ro**Bobj)
        OBJpower = OBJpower**2

        # Wiener Filtering
        fDof = fDo*(SFo*OTFo.conj()/NoisePower)/((SFo**2)*OTFpower/NoisePower + co/OBJpower)
        npDo = NoisePower

        ## Modulation factor determination
        # Mm = ModulationFactor(fDp,kA,OBJparaA,OTFo)

        # magnitude of illumination vector
        k2 = np.sqrt(kA.dot(kA.conj().T))
        print(k2)

        # vector along illumination direction
        kv = kA[1] + 1j*kA[0]
        Rp = np.abs(Cv+kv)

        # Object spectrum
        OBJp = Aobj*(Rp+0)**Bobj

        # illumination vector rounded to nearest pixel
        k3 = -np.round(kA)

        ## FIX: should indices be pythonised?
        OBJp[int(wo+0+k3[0]),int(wo+0+k3[1])] = 0.25*OBJp[int(wo+1+k3[0]),int(wo+0+k3[1])] \
            + 0.25*OBJp[int(wo+0+k3[0]),int(wo+1+k3[1])] \
            + 0.25*OBJp[int(wo-1+k3[0]),int(wo+0+k3[1])] \
            + 0.25*OBJp[int(wo+0+k3[0]),int(wo-1+k3[1])]

        # signal spectrum
        SIGap = OBJp*OTFo

        # Noise free object power computation
        Fpower = fDp*fDp.conj() - NoisePower
        fDpsqrt = np.sqrt(np.abs(Fpower))

        # frequency range over which signal power matching is done to estimate
        # modulation factor
        Zmask = (Ro > 0.2*k2)*(Ro < 0.8*k2)*(Rp > 0.2*k2)

        # least square approximation for modulation factor
        Mm = np.sum(SIGap*np.abs(fDpsqrt)*Zmask)
        Mm = Mm/np.sum(SIGap**2*Zmask)

        
        ## Duplex power (default) # inverted signs in Rp and Rm?
        Rp = np.abs(Cv-kv)
        Rm = np.abs(Cv+kv)
        OBJp = Aobj*(Rp**Bobj)
        OBJm = Aobj*(Rm**Bobj)
        k3 = np.round(kA)

        #  FIX: should indices be pythonised?
        OBJp[int(wo+0+k3[0]),int(wo+0+k3[1])] = 0.25*OBJp[int(wo+1+k3[0]),int(wo+0+k3[1])] \
            + 0.25*OBJp[int(wo+0+k3[0]),int(wo+1+k3[1])] \
            + 0.25*OBJp[int(wo-1+k3[0]),int(wo+0+k3[1])] \
            + 0.25*OBJp[int(wo+0+k3[0]),int(wo-1+k3[1])]
        OBJm[int(wo+0-k3[0]),int(wo+0-k3[1])] = 0.25*OBJm[int(wo+1-k3[0]),int(wo+0-k3[1])] \
            + 0.25*OBJm[int(wo+0-k3[0]),int(wo+1-k3[1])] \
            + 0.25*OBJm[int(wo-1-k3[0]),int(wo+0-k3[1])] \
            + 0.25*OBJm[int(wo+0-k3[0]),int(wo-1-k3[1])]

        # Filtering side lobes (off-center frequency components)
        SFo = Mm

        nNoise = fDp*Zo
        npDp = np.sum( nNoise*nNoise.conj() )/np.sum(Zo)
        OBJpower = OBJp**2
        fDpf = fDp*(SFo*OTFo.conj().T/npDp)/((SFo**2)*OTFpower/npDp + co/OBJpower)

        nNoise = fDm*Zo
        npDm = np.sum( nNoise*nNoise.conj() )/np.sum(Zo)
        OBJpower = OBJm**2
        fDmf = fDm*(SFo*OTFo.conj()/npDp)/((SFo**2)*OTFpower/npDp + co/OBJpower)



        ## doubling Fourier domain size if necessary
        DoubleMatSize = 0

        if 2*Kotf > wo:
            DoubleMatSize = 1 # 1 for doubling fourier domain size, 0 for keeping it unchanged
        if DoubleMatSize>0:
            t = 2*w
            to = t/2
            u = np.linspace(0,t-1,t)
            v = np.linspace(0,t-1,t)
            U, V = np.meshgrid(u,v)
            fDoTemp = np.zeros((2*w,2*w),dtype=np.complex128)
            fDpTemp = np.zeros((2*w,2*w),dtype=np.complex128)
            fDmTemp = np.zeros((2*w,2*w),dtype=np.complex128)
            OTFtemp = np.zeros((2*w,2*w),dtype=np.complex128)
            fDoTemp[int(wo+0):int(w+wo),int(wo+0):int(w+wo)] = fDof
            fDpTemp[int(wo+0):int(w+wo),int(wo+0):int(w+wo)] = fDpf
            fDmTemp[int(wo+0):int(w+wo),int(wo+0):int(w+wo)] = fDmf
            OTFtemp[int(wo+0):int(w+wo),int(wo+0):int(w+wo)] = OTFo
            fDof = fDoTemp
            fDpf = fDpTemp
            fDmf = fDmTemp
            OTFo = OTFtemp
        else:
            t = w
            to = t/2
            u = np.linspace(0,t-1,t)
            v = np.linspace(0,t-1,t)
            U, V = np.meshgrid(u,v)


        # Shifting the off-center frequency components to their correct location
        fDp1 = fft2(ifft2(fDpf)*np.exp( +1j*2*pi*(kA[1]/t*(U-to) + kA[0]/t*(V-to)) ))
        fDm1 = fft2(ifft2(fDmf)*np.exp( -1j*2*pi*(kA[1]/t*(U-to) + kA[0]/t*(V-to)) ))


        ## Shift induced phase error correction
        Cv2 = (U-to) + 1j*(V-to)
        Ro2 = np.abs(Cv2)
        Rp = np.abs(Cv2-kv)
        k2 = np.sqrt(kA.dot(kA.conj().T))

        # frequency range over which corrective phase is determined
        Zmask = (Ro2 < 0.8*k2)*(Rp < 0.8*k2)

        # corrective phase
        Angle0 = np.angle( np.sum( fDof*fDp1.conj()*Zmask ))

        # phase correction
        fDp2 = np.exp(+1j*Angle0)*fDp1
        fDm2 = np.exp(-1j*Angle0)*fDm1

        return fDof,fDp2,fDm2,npDo,npDp,npDm,Mm,DoubleMatSize



    S1aTnoisy = I[0]/255
    S2aTnoisy = I[1]/255
    S3aTnoisy = I[2]/255
    S1bTnoisy = I[3]/255
    S2bTnoisy = I[4]/255
    S3bTnoisy = I[5]/255    
    S1cTnoisy = I[6]/255
    S2cTnoisy = I[7]/255
    S3cTnoisy = I[8]/255
    OTFo = I[9] / 1000


    w = S1aTnoisy.shape[0]
    wo = w//2

    # computing PSFe for edge tapering SIM images
    PSFd = np.real(fftshift( ifft2(fftshift(OTFo**3)) ))
    PSFd = PSFd/np.max(PSFd)
    PSFd = PSFd/np.sum(PSFd)
    h = 30
    PSFe = PSFd[int(wo-h):int(wo+h),int(wo-h):int(wo+h)]


    import time

    t0 = time.perf_counter()

    ## obtaining the noisy estimates of three frequency components

    fAo,fAp,fAm,kA = PCMseparateF(S1aTnoisy,S2aTnoisy,S3aTnoisy,OTFo)
    fBo,fBp,fBm,kB = PCMseparateF(S1bTnoisy,S2bTnoisy,S3bTnoisy,OTFo)
    fCo,fCp,fCm,kC = PCMseparateF(S1cTnoisy,S2cTnoisy,S3cTnoisy,OTFo)

    t1 = time.perf_counter()
    print(t1-t0)

    # averaging the central frequency components
    fCent = (fAo + fBo + fCo)/3

    # Object power parameters determination
    x = np.linspace(0,w-1,w)
    y = np.linspace(0,w-1,w)
    X, Y = np.meshgrid(x,y)
    Cv = (X-wo) + 1j*(Y-wo)
    Ro = np.abs(Cv)
    Ro[wo,wo] = 1 # to avoid nan

    # OTF cut-off frequency
    Kotf = OTFedgeF(OTFo)

    ## object power parameters through optimization
    # OBJparaA = OBJpowerPara(fCent,OTFo)

    OBJparaOpt0 = lambda OBJpara0: OBJparaOpt(OBJpara0,fCent,OTFo)
    # options = optimset('LargeScale','off','Algorithm',...
    # 	'active-set','MaxFunEvals',500,'MaxIter'10500,'Display','notify')

    # obtaining crude initial guesses for Aobj and Bobj 
    Zm = (Ro>0.3*Kotf)*(Ro<0.4*Kotf)
    Aobj = np.sum(np.abs(fCent*Zm))/np.sum(Zm)
    Bobj = -0.5
    OBJpara0 = [Aobj, Bobj]


    # optimization step
    OBJparaA = scipy.optimize.fmin_bfgs(OBJparaOpt0, OBJpara0,maxiter=200)

    t2 = time.perf_counter()
    print(t2 - t1)  

    ## Wiener Filtering the noisy frequency components
    fAof,fApf,fAmf,Nao,Nap,Nam,Ma,DoubleMatSize = PCMfilteringF(fAo,fAp,fAm,OTFo,OBJparaA,kA)
    fBof,fBpf,fBmf,Nbo,Nbp,Nbm,Mb,DoubleMatSize = PCMfilteringF(fBo,fBp,fBm,OTFo,OBJparaA,kB)
    fCof,fCpf,fCmf,Nco,Ncp,Ncm,Mc,DoubleMatSize = PCMfilteringF(fCo,fCp,fCm,OTFo,OBJparaA,kC)

    ## doubling Fourier domain size if necessary
    OTFo = OTFdoubling(OTFo,DoubleMatSize)

    ## merging all 9 frequency components using generalized Wiener Filter
    Fsum,Fperi,Fcent = MergingHeptaletsF(fAof,fApf,fAmf,
        fBof,fBpf,fBmf,fCof,fCpf,fCmf,
        Ma,Mb,Mc,Nao,Nap,Nam,Nbo,Nbp,Nbm,
        Nco,Ncp,Ncm,kA,kB,kC,OBJparaA,OTFo)

    # Plotting SIM results2
    # SIMplot(Fsum,Fperi,Fcent,OTFo,kA,kB,kC,S1aTnoisy)

    t = OTFo.shape[0]
    h = 1*30 # for removing the image edges

    Dcent = np.real( ifft2(fftshift(Fcent)) )
    Dsum = np.real( ifft2(fftshift(Fsum)) )

    # Dcent.astype('uint16')
    # Dsum.astype('uint16')

    stack = np.array([Dcent,Dsum])

    if outputname is not None:
        io.imsave('recon.tif',stack)

    return stack
