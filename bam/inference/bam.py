import os
import numpy as np
import ehtim as eh
import matplotlib.pyplot as plt
import random
import pickle as pkl
from bam.inference.model_helpers import Gpercsq, M87_ra, M87_dec, M87_mass, M87_dist, M87_inc, isiterable
from numpy import arctan2, sin, cos, exp, log, clip, sqrt,sign
import dynesty
from dynesty import plotting as dyplot
from dynesty import utils as dyfunc
from bam.inference.schwarzschildexact import getphi, rinvert, getalphan, getpsin
# from bam.inference.schwarzschildexact import getscreencoords, getwindangle, getpsin, getalphan
# from bam.inference.gradients import LogLikeGrad, LogLikeWithGrad, exact_vis_loglike
# theano.config.exception_verbosity='high'
# theano.config.compute_test_value = 'ignore'


def example_jfunc(r, phi, jargs):
    peak_r = jargs[0]
    thickness = jargs[1]
    return exp(-4.*log(2)*((r-peak_r)/thickness)**2)

# def example_model_jfunc(r, phi, jargs):
#     peak_r = jargs[0]
#     thickness = jargs[1]
#     return pm.math.exp(-4.*np.log(2)*((r-peak_r)/thickness)**2)

def get_uniform_transform(lower, upper):
    return lambda x: (upper-lower)*x + lower

def beloborodov_r(psi, b):
    psi = np.complex128(psi)
    b = np.complex128(b)
    return np.sqrt((1-np.cos(psi)**2)/(1+np.cos(psi)**2) + b**2 / np.sin(psi)**2)-(1-np.cos(psi))/(1+np.cos(psi))

def getrt(rho):
    rho = np.complex128(rho)
    det = (-9*rho**2 + np.sqrt(3)*np.sqrt(27*rho**4 - (2+0j) * rho**6))**(1/3)
    rt = 2.**(2/3)*rho**2 / (3**(1/3)*det) + 2.**(1/3)*det / 3**(2/3)
    return rt

def getpsit(rt):
    rt = np.complex128(rt)
    return np.arccos(-2 / (rt-2))
#psit - np.abs(psit-psivecs[n))
def betterborodov_v1(rho, varphi, inc, nmax):
    """
    Given rho and varphi vecs, inclination, and nmax,
    compute r and phi vecs for each n
    """
    phivecs = [getphi(varphi, inc, n).real for n in range(nmax+1)]
    rt = getrt(rho)
    psit = getpsit(rt)
    psivecs = [getpsin(inc, phivecs[n], n) for n in range(nmax+1)]
    rvecs = [beloborodov_r(psit - np.sqrt((psit-psivecs[n])**2), rho).real for n in range(nmax+1)]
    
    signprvecs = [(np.abs(psivecs[n])<psit).astype(int) for n in range(nmax+1)]
    for n in range(nmax+1):
        signprvecs[n][np.abs(psivecs[n])<psit] = -1
        signprvecs[n][rho <= np.sqrt(27)] = 1
    arctannumvecs = [np.arctan(1/np.sqrt(rvecs[n]**2 / rho**2 / (1-2/rvecs[n])-1)) for n in range(nmax+1)]
    alphavecs = [np.sign(psivecs[n]) * (np.pi-arctannumvecs[n]) for n in range(nmax+1)]
    for n in range(nmax+1):
        mask = (signprvecs[n]==1)*(0<psivecs[n])*(psivecs[n]<np.pi)
        alphavecs[n][mask] = (np.sign(psivecs[n])*arctannumvecs[n])[mask]

    return rvecs, phivecs, psivecs, alphavecs

def beloborodov(rho, varphi, inc, nmax):

    phivec = arctan2(sin(varphi),cos(varphi)*cos(inc))

    sinprod = sin(inc)*sin(phivec)
    numerator = 1.+rho**2 - (-3.+rho**2.)*sinprod+3.*sinprod**2. + sinprod**3. 
    denomenator = (-1.+sinprod)**2 * (1+sinprod)
    sqq = sqrt(numerator/denomenator)
    rvec = np.maximum((1.-sqq + sinprod*(1.+sqq))/(sinprod-1.),2.+1.e-5)
    cospsi = -sinprod#sin(inc) * sin(phivec)
    psivec = np.arccos(cospsi)
    # sinpsi = sqrt(1. - cospsi**2)
    cosalpha = 1. - (1. - cospsi) * (1. - 2./rvec)
    # sinalpha =sqrt(1. - cosalpha**2)
    alphavec = np.arccos(cosalpha)
    rvecs = [rvec]
    phivecs = [phivec]
    alphavecs = [alphavec]
    psivecs = [psivec]
    return rvecs, phivecs, psivecs, alphavecs

# def getsignpr(b, r, theta, psin):
#     # if b <= np.sqrt(27):
#     #     return 1
#     psit = getpsit(b, theta)
#     out = (np.abs(psin) < psit).astype(int)
#     out[np.abs(psin)<psit] = -1
#     out[b <= np.sqrt(27)]=1
#     return out

# #now compute alpha_n
# def approxalphan(b, r, theta, psin):
#     signpr = getsignpr(b, r, theta, psin)
#     arctannum = np.arctan(1 / np.sqrt(r**2/b**2/(1-2/r)-1))
#     signpsin = np.sign(psin)
#     out = signpsin * (np.pi-arctannum)
#     mask = (signpr == 1)*(0<psin)*(psin<np.pi)
#     out[mask] = (signpsin*arctannum)[mask]
#     return out
# rho = np.linspace(0,10,100)
# varphi = np.zeros_like(rho)
# rvecs, phivecs = betterborodov_v1(rho, varphi, 0, 0)
# plt.plot(rho, rvecs[0])
# plt.show()

class Bam:
    '''The Bam class is a collection of accretion flow and black hole parameters.
    jfunc: a callable that takes (r, phi, jargs)
    if Bam is in modeling mode, jfunc should use pm functions
    '''
    #class contains knowledge of a grid in Boyer-Lindquist coordinates, priors on each pixel, and the machinery to fit them
    def __init__(self, fov, npix, jfunc, jarg_names, jargs, M, D, inc, zbl, PA=0.,  nmax=0, beta=0., chi=0., thetabz=np.pi/2, spec=1., f=0., e=0., calctype='approx',approxtype='belo', Mscale = 2.e30*1.e9):
        self.approxtype = approxtype
        self.fov = fov
        self.npix = npix
        self.recent_sampler = None
        self.recent_results = None
        # self.MAP_values = None
        self.jfunc = jfunc
        self.jarg_names = jarg_names
        self.jargs = jargs
        self.M = M
        self.D = D
        self.inc = inc
        self.PA = PA
        self.beta = beta
        self.chi=chi
        self.thetabz=thetabz
        self.spec=spec
        self.f = f
        self.e = e
        self.nmax = nmax
        self.zbl = zbl
        self.calctype = calctype
        self.rho_c = np.sqrt(27)
        self.Mscale = Mscale
        pxi = (np.arange(npix)-0.01)/npix-0.5
        pxj = np.arange(npix)/npix-0.5
        # get angles measured north of west
        PXI,PXJ = np.meshgrid(pxi,pxj)
        varphi = np.arctan2(-PXJ,PXI)# - np.pi/2
        self.varphivec = varphi.flatten()
        
        #get grid of angular radii
        mui = pxi*fov
        muj = pxj*fov
        MUI,MUJ = np.meshgrid(mui,muj)
        MUDISTS = np.sqrt(np.power(MUI,2.)+np.power(MUJ,2.))
        self.MUDISTS = MUDISTS.flatten()

        #while we're at it, get x and y
        self.imxvec = -self.MUDISTS*np.cos(self.varphivec)
       
        self.imyvec = self.MUDISTS*np.sin(self.varphivec)
        if any([isiterable(i) for i in [M, D, inc, zbl, PA, f, beta, chi, thetabz, e, spec]+jargs]):
            mode = 'model'
        else:
            mode = 'fixed' 
        self.mode = mode


        self.all_params = [M, D, inc, zbl, PA, beta, chi, thetabz, spec, f, e]+jargs
        self.all_names = ['M','D','inc','zbl','PA','beta','chi','thetabz','spec','f','e']+jarg_names
        self.modeled_indices = [i for i in range(len(self.all_params)) if isiterable(self.all_params[i])]
        self.modeled_names = [self.all_names[i] for i in self.modeled_indices]
        self.modeled_params = [i for i in self.all_params if isiterable(i)]
        # print(len(self.modeled_indices))
        # print(len(self.modeled_names))
        # print(len(self.modeled_params))
        self.periodic_names = []
        self.periodic_indices=[]
        #if PA and chi are being modeled, check if they are allowing a full 2pi wrap
        #if so, let dynesty know they are periodic later
        for i in ['PA','chi']:
            if i in self.modeled_names:
                bounds = self.modeled_params[self.modeled_names.index(i)]
                if np.isclose(np.exp(1j*bounds[0]),np.exp(1j*bounds[1]),rtol=1e-12):
                    print("Found periodic prior on "+str(i))
                    self.periodic_names.append(i)
                    self.periodic_indices.append(self.modeled_names.index(i))
        self.model_dim = len(self.modeled_names)
        # self.periodic_names = [i for i in ['PA','chi'] if i in self.modeled_names]
        # self.periodic_indices = [self.modeled_names.index(i) for i in self.periodic_names]


        print("Finished building Bam! in "+ self.mode +" mode with calctype " +self.calctype)  
        

    def test(self, i):
        if len(i) == self.npix**2:
            i = i.reshape((self.npix, self.npix))
        plt.imshow(i)
        plt.colorbar()
        plt.show()

    def compute_image(self, imparams):
        """
        Given a list of values of modeled parameters in imparams,
        compute the resulting qvec, uvec, ivec, rotimxvec, and rotimyvec.
        """
        M, D, inc, zbl, PA, beta, chi, thetabz, spec, jargs = imparams

        # self.M = M
        #this is the end of the problem setup;
        #everything after this point should bifurcate depending on calctype

         # def rho_conv(r, phi, D, theta0, r_g):
        #     rho2 = (((r/D)**2.0)*(1.0 - ((sin(theta0)**2.0)*(sin(phi)**2.0)))) + ((2.0*r*r_g/(D**2.0))*((1.0 + (sin(theta0)*sin(phi)))**2.0))
        #     rho = sqrt(rho2)
        #     return rho

        # def emission_coordinates(rho, varphi):
        #convert mudists to gravitational units
        rhovec = D / (M*self.Mscale*Gpercsq) * self.MUDISTS
        

        if self.calctype == 'approx':

            if self.approxtype == 'better':
                rvecs, phivecs, psivecs, alphavecs = betterborodov_v1(rhovec, self.varphivec, inc, self.nmax)
            elif self.approxtype == 'belo':
                rvecs, phivecs, psivecs, alphavecs = beloborodov(rhovec, self.varphivec, inc, self.nmax)
            # rvec, phivec = emission_coordinates(self.rhovec, self.varphivec)

        elif self.calctype == 'exact':
            
            rvecs = [np.maximum(rinvert(rhovec,self.varphivec, n, inc),2.+1.e-5) for n in range(self.nmax+1)]
            phivecs = [getphi(self.varphivec, inc, n) for n in range(self.nmax+1)]
            psivecs = [getpsin(inc, phivecs[n], n) for n in range(self.nmax+1)]
            alphavecs = [getalphan(rhovec, rvecs[n], inc, psis[n]) for n in range(self.nmax+1)]
            # cosalphas = [np.cos(alpha) for alpha in alphas]

        eta = chi+np.pi
        beq = sin(thetabz)
        bz = cos(thetabz)
        br = beq*cos(eta)
        bphi = beq*sin(eta)
        coschi = cos(chi)
        sinchi = sin(chi)
        betax = beta*coschi
        betay = beta*sinchi
        bx = br
        by = bphi
        bmag = sqrt(bx**2 + by**2 + bz**2)
        sintheta = sin(inc)
        # print(inc)
        # print(sintheta)
        costheta = cos(inc)    
        #now do the rotation

        rotimxvec = cos(PA)*self.imxvec - sin(PA)*self.imyvec
        rotimyvec = sin(PA)*self.imxvec + cos(PA)*self.imyvec

        ivecs = []
        qvecs = []
        uvecs = []        

        for n in range(self.nmax+1):
            rvec = np.maximum(rvecs[n],2.0001)
            phivec = phivecs[n]
            # print(phivec)
            psivec = psivecs[n]
            # print(psivec)
            alphavec = alphavecs[n]
            # print(alphavec)
            cospsi = np.cos(psivec)
            sinpsi = np.sin(psivec)
            cosalpha = np.cos(alphavec)
            sinalpha = np.sin(alphavec)
            # if n == 1:
            #     plt.imshow(rvec.reshape((self.npix, self.npix)))
            #     plt.show()
            # cosalpha = cosalphas[n]
            # psi = psis[n]

            gfac = sqrt(1. - 2./rvec)
            gfacinv = 1. / gfac
            gamma = 1. / sqrt(1. - beta**2)

            sinphi = sin(phivec)
            cosphi = cos(phivec)

            # self.test(sinpsi)


            # # cosalpha = np.cos(alpha)
            # if self.calctype == 'exact':
            #     psivec = psivecs[n]
            #     cospsi = np.cos(psivec)
            #     sinpsi = np.sin(psivec)
            #     alpha = alphas[n]
            #     cosalpha = np.cos(alpha)
            #     sinalpha = np.sin(alpha)
            # else:                    
            #     psivec = psivecs[n]
            #     cospsi = np.cos(psivec)
            #     sinpsi = np.sin(psivec)
            #     # cospsi = -sintheta * sinphi
            #     # sinpsi = sqrt(1. - cospsi**2)
            #     cosalpha = 1. - (1. - cospsi) * (1. - 2./rvec)
            #     sinalpha =sqrt(1. - cosalpha**2)
                
            
            sinxi = sintheta * cosphi / sinpsi
            cosxi = costheta / sinpsi
            
            kPthat = gfacinv
            kPxhat = cosalpha * gfacinv
            kPyhat = -sinxi * sinalpha * gfacinv
            kPzhat = cosxi * sinalpha * gfacinv
            
            kFthat = gamma * (kPthat - betax * kPxhat - betay * kPyhat)
            kFxhat = -gamma * betax * kPthat + (1. + (gamma-1.) * coschi**2) * kPxhat + (gamma-1.) * coschi * sinchi * kPyhat
            kFyhat = -gamma * betay * kPthat + (gamma-1.) * sinchi * coschi * kPxhat + (1. + (gamma-1.) * sinchi**2) * kPyhat
            kFzhat = kPzhat


            delta = 1. / kFthat
            
            kcrossbx = kFyhat * bz - kFzhat * by
            kcrossby = kFzhat * bx - kFxhat * bz
            kcrossbz = kFxhat * by - kFyhat * bx

            # polfac = np.sqrt(kcrossbx**2 + kcrossby**2 + kcrossbz**2) / (kFthat * bmag)
            sinzeta = sqrt(kcrossbx**2 + kcrossby**2 + kcrossbz**2) / (kFthat * bmag)
            # self.test(sinzeta)
            profile = self.jfunc(rvec, phivec, jargs)
            # profile=1.
            # 
            # print(self.profile)

            polarizedintensity = sinzeta**(1.+spec) * delta**(3. + spec) * profile
            
            # if INTENSITYISOTROPIC:
            #     intensity = delta**(3. + SPECTRALINDEX)
            # else:
            #     intensity = polarizedintensity
            
                
            pathlength = kFthat/kFzhat
            mag = polarizedintensity*pathlength
            
            # self.test(bmag)
            # self.test(kFthat)

            fFthat = 0.
            fFxhat = kcrossbx / (kFthat * bmag)
            fFyhat = kcrossby / (kFthat * bmag)
            fFzhat = kcrossbz / (kFthat * bmag)
            
            # fPthat = gamma * (fFthat + beta * fFyhat)
            # fPxhat = fFxhat
            # fPyhat = gamma * (beta *fFthat + fFyhat)
            # fPzhat = fFzhat
            
            fPthat = gamma * (fFthat + betax * fFxhat + betay * fFyhat)
            fPxhat = gamma * betax * fFthat + (1. + (gamma-1.) * coschi**2) * fFxhat + (gamma-1.) * coschi * sinchi * fFyhat
            fPyhat = gamma * betay * fFthat + (gamma-1.) * sinchi * coschi * fFxhat + (1. + (gamma-1.) * sinchi**2) * fFyhat
            fPzhat = fFzhat

            kPrhat = kPxhat
            kPthhat = -kPzhat
            kPphat = kPyhat
            fPrhat = fPxhat
            fPthhat = -fPzhat
            fPphat = fPyhat
               
            k1 = rvec * (kPthat * fPrhat - kPrhat * fPthat)
            k2 = -rvec * (kPphat * fPthhat - kPthhat * fPphat)
                
            kOlp = rvec * kPphat


            radicand = kPthhat**2 - kPphat**2 * costheta**2 / sintheta**2
            radicand = np.maximum(radicand,0.)
            # due to machine precision, some small negative values are present. We clip these here.
            # radicand[radicand<0] = 0
            radical = sqrt(radicand)
            # plt.imshow(radicand)
            # plt.show()
            kOlth = rvec * radical * np.sign(sinphi)
            #sinphi / (sqrt(sinphi**2)+1.e-10)

            xalpha = -kOlp / sintheta
            ybeta = kOlth
            nu = -xalpha
            den = sqrt((k1**2 + k2**2) * (ybeta**2 + nu**2))

            ealpha = (ybeta * k2 - nu * k1) / den
            ebeta = (ybeta * k1 + nu * k2) / den

            qvec = -mag*(ealpha**2 - ebeta**2)
            uvec = -mag*(2*ealpha*ebeta)
            qvec[np.isnan(qvec)]=0
            uvec[np.isnan(uvec)]=0
            ivec = sqrt(qvec**2 + uvec**2)
            qvecs.append(qvec)
            uvecs.append(uvec)
            ivecs.append(ivec)
            # tf = np.sum(ivec)
            # qvec = qvec * zbl/tf
            # uvec = uvec * zbl/tf
            # ivec = ivec * zbl/tf
        tf = np.sum(ivecs)
        ivecs = [ivec*zbl/tf for ivec in ivecs]
        qvecs = [qvec*zbl/tf for qvec in qvecs]
        uvecs = [uvec*zbl/tf for uvec in uvecs]
        return ivecs, qvecs, uvecs, rotimxvec, rotimyvec

    def vis(self, vec, rotimxvec, rotimyvec, u, v):#, vis_types=list('i')):

        u = np.array(u)
        v = np.array(v)

        matrix = np.outer(u, rotimxvec)+np.outer(v, rotimyvec)
        A_real = np.cos(2.0*np.pi*matrix)
        A_imag = -np.sin(2.0*np.pi*matrix)
        visreal_model = np.dot(A_real,vec)
        visimag_model = np.dot(A_imag,vec)
        return visreal_model+1j*visimag_model

    def vis_fixed(self, u, v):
        if self.mode=='model':
            print("Can't compute fixed visibilities in model mode!")
            return
        imparams = [self.M, self.D, self.inc, self.zbl, self.PA, self.beta, self.chi, self.thetabz, self.spec, self.jargs]
        ivecs, qvecs, uvecs, rotimxvec, rotimyvec = self.compute_image(imparams)
        ivec = np.sum(ivecs, axis=0)
        qvec = np.sum(qvecs, axis=0)
        uvec = np.sum(uvecs, axis=0)
        return self.vis(ivec, rotimxvec, rotimyvec, u, v)

    def observe_same(self, obs):
        if self.mode=='model':
            print("Can't observe_same in model mode!")
            return
        im = self.make_image(ra=obs.ra, dec=obs.dec, rf=obs.rf, mjd = obs.mjd, source=obs.source)
        return im.observe_same(obs)

    def cphase(self, vec, rotimxvec, rotimyvec, u1, u2, u3, v1, v2, v3):
        
        vis12 = self.vis(vec, rotimxvec, rotimyvec, u1, v1)
        vis23 = self.vis(vec, rotimxvec, rotimyvec, u2, v2)
        vis31 = self.vis(vec, rotimxvec, rotimyvec, u3, v3)
        phase12 = np.angle(vis12)
        phase23 = np.angle(vis23)
        phase31 = np.angle(vis31)
        cphase_model = phase12+phase23+phase31
        return cphase_model

    def logcamp(self, vec, rotimxvec, rotimyvec, u1, u2, u3, u4, v1, v2, v3, v4):
        
        # print("Building direct image FT matrices.")
        vis12 = self.vis(vec, rotimxvec, rotimyvec,u1,v1)
        vis34 = self.vis(vec, rotimxvec, rotimyvec,u2,v2)
        vis23 = self.vis(vec, rotimxvec, rotimyvec,u3,v3)
        vis14 = self.vis(vec, rotimxvec, rotimyvec,u4,v4)
        amp12 = np.abs(vis12)
        amp34 = np.abs(vis34)
        amp23 = np.abs(vis23)
        amp14 = np.abs(vis14)
        logcamp_model = np.log(amp12)+np.log(amp34)-np.log(amp23)-np.log(amp14)
        return logcamp_model



    def build_likelihood(self, obs, data_types=['vis']):
        """
        Given an observation and a list of data product names, 
        return a likelihood function that accounts for each contribution. 
        """

        
        
        if 'vis' in data_types:
            vis = obs.data['vis']
            sigma = obs.data['sigma']
            amp = obs.unpack('amp')['amp']
            u = obs.data['u']
            v = obs.data['v']
            Nvis = len(vis)
            print("Building vis likelihood!")
        if 'logcamp' in data_types:
            logcamp_data = obs.c_amplitudes(ctype='logcamp')
            logcamp = logcamp_data['camp']
            logcamp_sigma = logcamp_data['sigmaca']
            campu1 = logcamp_data['u1']
            campu2 = logcamp_data['u2']
            campu3 = logcamp_data['u3']
            campu4 = logcamp_data['u4']
            campv1 = logcamp_data['v1']
            campv2 = logcamp_data['v2']
            campv3 = logcamp_data['v3']
            campv4 = logcamp_data['v4']
            Ncamp = len(logcamp)
            print("Building logcamp likelihood!")
        if 'cphase' in data_types:
            cphase_data = obs.c_phases(ang_unit='rad')
            cphaseu1 = cphase_data['u1']
            cphaseu2 = cphase_data['u2']
            cphaseu3 = cphase_data['u3']
            cphasev1 = cphase_data['v1']
            cphasev2 = cphase_data['v2']
            cphasev3 = cphase_data['v3']
            cphase = cphase_data['cphase']
            cphase_sigma = cphase_data['sigmacp']
            Ncphase = len(cphase)
            print("Building cphase likelihood!")

        def loglike(params):
            to_eval = []
            for name in self.all_names:
                if not(name in self.modeled_names):
                    to_eval.append(self.all_params[self.all_names.index(name)])
                else:
                    to_eval.append(params[self.modeled_names.index(name)])
            #at this point, to_eval contains the full model description,
            #so it should have 11+N parameters where N is the number of jargs
            #M, D, inc, zbl, PA, beta, chi, thetabz, spec, f, e + jargs
            #f and e are not used in image computation, so slice around them for now
            imparams = to_eval[:9] + [to_eval[11:]]
            ivecs, qvecs, uvecs, rotimxvec, rotimyvec = self.compute_image(imparams)
            out = 0.
            ivec = np.sum(ivecs,axis=0)
            qvec = np.sum(qvecs,axis=0)
            uvec = np.sum(uvecs,axis=0)
            if 'vis' in data_types:
                sd = sqrt(sigma**2.0 + (to_eval[9]*amp)**2.0+to_eval[10]**2.0)
                model_vis = self.vis(ivec, rotimxvec, rotimyvec, u, v)
                vislike = -1./(2.*Nvis) * np.sum(np.abs(model_vis-vis)**2 / sd**2)
                out+=vislike
            if 'logcamp' in data_types:
                model_logcamp = self.logcamp(ivec, rotimxvec, rotimyvec, campu1, campu2, campu3, campu4, campv1, campv2, campv3, campv4)
                logcamplike = -1./Ncamp * np.sum((logcamp-model_logcamp)**2 / logcamp_sigma**2)
                out += logcamplike
            if 'cphase' in data_types:
                model_cphase = self.cphase(ivec, rotimxvec, rotimyvec, cphaseu1, cphaseu2, cphaseu3, cphasev1, cphasev2, cphasev3)
                cphaselike = -2/Ncphase * np.sum((1-np.cos(cphase-model_cphase))/cphase_sigma)
                out += cphaselike
            return out
        print("Built combined likelihood function!")
        return loglike

    def build_prior_transform(self):
        functions = [get_uniform_transform(bounds[0],bounds[1]) for bounds in self.modeled_params]

        def ptform(hypercube):
            scaledcube = np.copy(hypercube)
            for i in range(len(scaledcube)):
                scaledcube[i] = functions[i](scaledcube[i])
            return scaledcube
        return ptform

    
    def build_sampler(self, loglike, ptform, dynamic=False, nlive=1000, bound='multi'):#, pool=None, queue_size=None):
        
        if dynamic:
            sampler = dynesty.DynamicNestedSampler(loglike, ptform,self.model_dim, periodic=self.periodic_indices, bound=bound, nlive=nlive)#, pool=pool, queue_size=queue_size)
        else:
            sampler = dynesty.NestedSampler(loglike, ptform, self.model_dim, periodic=self.periodic_indices, bound=bound, nlive=nlive)#, pool=pool, queue_size=queue_size)
        return sampler

    def setup(self, obs, data_types=['vis'],dynamic=False, nlive=1000, bound='multi'):#, pool=None, queue_size=None):
        ptform = self.build_prior_transform()
        loglike = self.build_likelihood(obs, data_types=data_types)
        sampler = self.build_sampler(loglike,ptform,dynamic=dynamic, nlive=nlive, bound=bound)#, pool=pool, queue_size=queue_size)

        self.recent_sampler = sampler
        print("Ready to model with this BAM's recent_sampler! Call run_nested!")

    def run_nested(self):
        self.recent_sampler.run_nested()
        self.recent_results = self.recent_sampler.results
        return self.recent_results

        
    def runplot(self, save='', show=True):
        fig, axes = dyplot.runplot(self.recent_results)
        if len(save)>0:
            plt.savefig(save,bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close('all')


    def traceplot(self, save='', show=True):
        fig, axes = dyplot.traceplot(self.recent_results, labels=self.modeled_names)
        if len(save)>0:
            plt.savefig(save,bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close('all')


    def cornerplot(self, save='',show=True):
        fig, axes = dyplot.cornerplot(self.recent_results, labels=self.modeled_names)
        if len(save)>0:
            plt.savefig(save,bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close('all')

    def mean_and_cov(self):
        samples = self.recent_results.samples
        weights = np.exp(self.recent_results.logwt - self.recent_results.logz[-1])
        return dyfunc.mean_and_cov(samples, weights)

    def MOP_Bam(self):
        mean, cov = self.mean_and_cov()
        to_eval = []
        for name in self.all_names:
            if not(name in self.modeled_names):
                to_eval.append(self.all_params[self.all_names.index(name)])
            else:
                to_eval.append(mean[self.modeled_names.index(name)])
        return Bam(self.fov, self.npix, self.jfunc, self.jarg_names, to_eval[11:], to_eval[0], to_eval[1], to_eval[2], to_eval[3], PA=to_eval[4],  nmax=self.nmax, beta=to_eval[5], chi=to_eval[6], thetabz=to_eval[7], spec=to_eval[8], f=to_eval[9], e=to_eval[10], calctype=self.calctype,approxtype=self.approxtype, Mscale = self.Mscale)

    def make_image(self, ra=M87_ra, dec=M87_dec, rf= 230e9, mjd = 57854, source='M87',n='all'):
        """
        Returns an ehtim Image object corresponding to the Blimage n0 emission
        """

        if self.mode == 'model':
            print("Cannot directly make images in model mode! Call sample_blimage or MAP_blimage and display that!")
            return
        imparams = self.all_params[:9] + [self.all_params[11:]]
        ivecs, qvecs, uvecs, rotimxvec, rotimyvec = self.compute_image(imparams)
        if n =='all':
            ivec = np.sum(ivecs,axis=0)
            qvec = np.sum(qvecs,axis=0)
            uvec = np.sum(uvecs,axis=0)
        elif type(n) is int:
            ivec = ivecs[n]
            qvec = qvecs[n]
            uvec = uvecs[n]

        im = eh.image.make_empty(self.npix,self.fov, ra=ra, dec=dec, rf= rf, mjd = mjd, source='M87')
        im.ivec = ivec
        im.qvec = qvec
        im.uvec = uvec

        im = im.rotate(self.PA)
        # im.ivec *= self.tf / im.total_flux()
        return im



    # def logcamp_chisq(self,obs):
    #     if self.mode != 'fixed':
    #         print("Can only compute chisqs to fixed model!")
    #         return
    #     logcamp_data = obs.c_amplitudes(ctype='logcamp')
    #     sigmaca = logcamp_data['sigmaca']
    #     logcamp = logcamp_data['camp']
    #     model_logcamps = self.logcamp(logcamp_data['u1'],logcamp_data['u2'],logcamp_data['u3'],logcamp_data['u4'],logcamp_data['v1'],logcamp_data['v2'],logcamp_data['v3'],logcamp_data['v4'])
    #     logcamp_chisq = 1/len(sigmaca) * np.sum(np.abs((logcamp-model_logcamps)/sigmaca)**2)
    #     return logcamp_chisq

    # def cphase_chisq(self,obs):
    #     if self.mode != 'fixed':
    #         print("Can only compute chisqs to fixed model!")
    #         return
    #     cphase_data = obs.c_phases(ang_unit='rad')
    #     cphase = cphase_data['cphase']
    #     sigmacp = cphase_data['sigmacp']
    #     model_cphases = self.cphase(cphase_data['u1'],cphase_data['u2'],cphase_data['u3'],cphase_data['v1'],cphase_data['v2'],cphase_data['v3'])
    #     cphase_chisq = (2.0/len(sigmacp)) * np.sum((1.0 - np.cos(cphase-model_cphases))/(sigmacp**2))
    #     return cphase_chisq

    # def vis_chisq(self,obs):
    #     if self.mode !='fixed':
    #         print("Can only compute chisqs to fixed model!")
    #         return

    #     u = obs.data['u']
    #     v = obs.data['v']
    #     sigma = obs.data['sigma']  
    #     amp = obs.unpack('amp')['amp']
    #     vis = obs.data['vis']
    #     sd = np.sqrt(sigma**2.0 + (self.sys_err*amp)**2.0 + self.abs_err**2.0)

    #     model_vis = self.vis(u,v)
    #     absdelta = np.abs(model_vis-vis)
    #     vis_chisq = np.sum((absdelta/sd)**2)/(2*len(vis))
    #     return vis_chisq

    # def all_chisqs(self, obs):
    #     if self.mode !='fixed':
    #         print("Can only compute chisqs to fixed model!")
    #         return
    #     logcamp_chisq = self.logcamp_chisq(obs)
    #     cphase_chisq = self.cphase_chisq(obs)
    #     vis_chisq = self.vis_chisq(obs)
    #     return {'logcamp':logcamp_chisq,'cphase':cphase_chisq,'vis':vis_chisq}



def load_blimage(blimage_name):
    """
    Loads a dictionary of blimage parameters and returns the created blimage.
    """
    with open(blimage_name, 'rb') as file:
        in_dict = pkl.load(file)
    r_lims = in_dict['r_lims']
    phi_lims = in_dict['phi_lims']
    nr = in_dict['nr']
    nphi = in_dict['nphi']
    M = in_dict['M']
    D = in_dict['D']
    inc = in_dict['inc']
    j = in_dict['j']
    PA = in_dict['j']
    beta = in_dict['beta']
    chi = in_dict['chi']
    spec = in_dict['spec']
    f = in_dict['f']
    nmax = in_dict['nmax']
    zbl = in_dict['zbl']
    
    new_blim =  Blimage(r_lims, phi_lims, nr, nphi, M, D, inc, j, zbl, PA=PA, nmax=nmax, beta=beta, chi=chi, spec=spec, f=f)
    if not(isiterable(j)):
        new_blim.blixels = in_dict['blixels']
        new_blim.j = 0
    return new_blim

