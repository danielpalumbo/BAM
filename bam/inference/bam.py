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



def get_uniform_transform(lower, upper):
    return lambda x: (upper-lower)*x + lower

def beloborodov_r(psi, b):
    psi = np.complex128(psi)
    b = np.complex128(b)
    return np.sqrt(((1-np.cos(psi))**2)/((1+np.cos(psi))**2) + b**2 / np.sin(psi)**2)-(1-np.cos(psi))/(1+np.cos(psi))

def getrt(rho):
    rho = np.complex128(rho)
    det = (-9*rho**2 + np.sqrt(3)*np.sqrt(27*rho**4 - rho**6))**(1/3)
    rt = rho**2 / (3**(1/3)*det) + det / 3**(2/3)
    return rt

def getpsit(rt):
    # rt = np.complex128(rt)
    return np.arccos(-2 / (rt-2))

def piecewise_better(rho, varphi, inc, nmax):
    phivec = arctan2(sin(varphi), cos(varphi)*cos(inc))
    psivec = getpsin(inc, phivec, 0)
    psivecs = [psivec]+[psivec+n*np.pi for n in range(1, nmax+1)]

    sinprod = sin(inc)*sin(phivec)
    numerator = 1.+rho**2 - (-3.+rho**2.)*sinprod+3.*sinprod**2. + sinprod**3. 
    denomenator = (-1.+sinprod)**2 * (1+sinprod)
    sqq = sqrt(numerator/denomenator)
    rvec = (1.-sqq + sinprod*(1.+sqq))/(sinprod-1.)

    rt = getrt(rho)
    psit = getpsit(rt)
    nosubim_mask = 2*psit < psivec+np.pi

    rvecs = [rvec]+[beloborodov_r(psit - np.sqrt((psit-psivecs[n])**2), rho).real for n in range(1, nmax+1)]
    for vec in rvecs[1:]:
        vec[nosubim_mask] = np.nan

    phivecs = [phivec]+[phivec+n*np.pi for n in range(1,nmax+1)]

    # cosalphavec = 1. - (1. - cos(psivecs[0])) * (1. - 2./rvecs[0])
    # alphavecs = [np.arccos(cosalphavec)]+[np.arcsin(np.sqrt(1-2/rvecs[n])*np.sqrt(27)/rvecs[n]) for n in range(1, nmax+1)]
    
    cosalphavecs = [1. - (1. - cos(psivecs[i])) * (1. - 2./rvecs[i]) for i in range(nmax+1)]
    alphavecs = [np.arccos(cosalphavec) for cosalphavec in cosalphavecs]#+[np.arcsin(np.sqrt(1-2/rvecs[n])*np.sqrt(27)/rvecs[n]) for n in range(1, nmax+1)]
    

    for i in range(1,len(alphavecs)):
        alphavecs[i] = np.sign(alphavecs[i])*(np.pi-np.abs(alphavecs[i]))

    return rvecs, phivecs, psivecs, alphavecs

def cphase_uvpairs(cphase_data):
    cphaseu1 = cphase_data['u1']
    cphaseu2 = cphase_data['u2']
    cphaseu3 = cphase_data['u3']
    cphasev1 = cphase_data['v1']
    cphasev2 = cphase_data['v2']
    cphasev3 = cphase_data['v3']
    cphaseuv1 = np.vstack([cphaseu1,cphasev1]).T
    cphaseuv2 = np.vstack([cphaseu2,cphasev2]).T
    cphaseuv3 = np.vstack([cphaseu3,cphasev3]).T
    return cphaseuv1, cphaseuv2, cphaseuv3

def logcamp_uvpairs(logcamp_data):
    campu1 = logcamp_data['u1']
    campu2 = logcamp_data['u2']
    campu3 = logcamp_data['u3']
    campu4 = logcamp_data['u4']
    campv1 = logcamp_data['v1']
    campv2 = logcamp_data['v2']
    campv3 = logcamp_data['v3']
    campv4 = logcamp_data['v4']
    campuv1 = np.vstack([campu1,campv1]).T
    campuv2 = np.vstack([campu2,campv2]).T
    campuv3 = np.vstack([campu3,campv3]).T
    campuv4 = np.vstack([campu4,campv4]).T
    return campuv1, campuv2, campuv3, campuv4


class Bam:
    '''The Bam class is a collection of accretion flow and black hole parameters.
    jfunc: a callable that takes (r, phi, jargs)
    if Bam is in modeling mode, jfunc should use pm functions
    '''
    #class contains knowledge of a grid in Boyer-Lindquist coordinates, priors on each pixel, and the machinery to fit them
    def __init__(self, fov, npix, jfunc, jarg_names, jargs, M, D, inc, zbl, PA=0.,  nmax=0, beta=0., chi=0., thetabz=np.pi/2, spec=1., f=0., e=0., calctype='approx',approxtype='better', Mscale = 2.e30*1.e9, polflux=True, source='', periodic=False):
        self.periodic=periodic
        self.dynamic=False
        self.source = source
        self.polflux = polflux
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
        if self.periodic:
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

        if self.mode == 'fixed':
            print("Fixed Bam: precomputing all subimages.")
            self.imparams = [self.M, self.D, self.inc, self.zbl, self.PA, self.beta, self.chi, self.thetabz, self.spec, self.jargs]
            self.ivecs, self.qvecs, self.uvecs= self.compute_image(self.imparams)
            # ivecs, qvecs, uvecs, rotimxvec, rotimyvec = self.compute_image(imparams), self.rotimxvec, self.rotimyvec 

        self.modelim = None
        print("Finished building Bam! in "+ self.mode +" mode with calctype " +self.calctype)

        if self.calctype == 'approx':  
            print("Using approxtype", approxtype)
        

    def test(self, i, out):
        if len(i) == self.npix**2:
            i = i.reshape((self.npix, self.npix))
        plt.imshow(i)
        plt.colorbar()
        plt.savefig(out+'.png',bbox_inches='tight')
        plt.close('all')
        # plt.show()

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
                rvecs, phivecs, psivecs, alphavecs = piecewise_better(rhovec, self.varphivec, inc, self.nmax)
            elif self.approxtype == 'belo':
                print("No! Use better!")
                return
                # rvecs, phivecs, psivecs, alphavecs = beloborodov(rhovec, self.varphivec, inc, self.nmax)
            # rvec, phivec = emission_coordinates(self.rhovec, self.varphivec)

        elif self.calctype == 'exact':
            
            rvecs = [np.maximum(rinvert(rhovec,self.varphivec, n, inc),2.+1.e-5) for n in range(self.nmax+1)]
            phivecs = [getphi(self.varphivec, inc, n) for n in range(self.nmax+1)]
            psivecs = [getpsin(inc, phivecs[n], n) for n in range(self.nmax+1)]
            alphavecs = [getalphan(rhovec, rvecs[n], inc, psivecs[n]) for n in range(self.nmax+1)]
            # cosalphas = [np.cos(alpha) for alpha in alphas]
        # if len(rvecs)>1:
        #     self.test(rvecs[1])
        #     self.test(phivecs[1])
        #     self.test(alphavecs[0])
        #     self.test(alphavecs[1])
            # self.test(self.jfunc(rvecs[1], phivecs[1],jargs))

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
        #note that this is essentially an inverse mapping, saying where
        #previously unrotated points can be found on the new image, after rotation
        #hence, the minus sign
        # rotimxvec = cos(-PA)*self.imxvec - sin(-PA)*self.imyvec
        # rotimyvec = sin(-PA)*self.imxvec + cos(-PA)*self.imyvec
        # self.test(rotimxvec)
        ivecs = []
        qvecs = []
        uvecs = []        

        for n in range(self.nmax+1):
            # rvec = np.maximum(rvecs[n],2.0001)
            rvec = rvecs[n]
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
            # self.test(alphavec, out='alphavec'+str(n))
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
            # if n%2 == 1:
            #     sinxi = np.sin(np.pi-np.arcsin(sinxi))
            #     cosxi = np.cos(np.pi-np.arccos(cosxi))

            
            kPthat = gfacinv
            kPxhat = cosalpha * gfacinv
            kPyhat = -sinxi * sinalpha * gfacinv
            kPzhat = cosxi * sinalpha * gfacinv

            # self.test(alphavec,'alpha'+str(n))
            # self.test(psivec,'psi'+str(n))
            # self.test(sinxi, 'sinxi'+str(n))
            # self.test(cosxi, 'cosxi'+str(n))
            # self.test(kPthat, 'kPthat'+str(n))
            # self.test(kPxhat, 'kPxhat'+str(n))
            # self.test(kPyhat, 'kPyhat'+str(n))
            # self.test(kPzhat, 'kPzhat'+str(n))
            kFthat = gamma * (kPthat - betax * kPxhat - betay * kPyhat)
            kFxhat = -gamma * betax * kPthat + (1. + (gamma-1.) * coschi**2) * kPxhat + (gamma-1.) * coschi * sinchi * kPyhat
            kFyhat = -gamma * betay * kPthat + (gamma-1.) * sinchi * coschi * kPxhat + (1. + (gamma-1.) * sinchi**2) * kPyhat
            kFzhat = kPzhat


            # kFzhat *= (-1)**n


            delta = 1. / kFthat
            
            # self.test(delta)

            kcrossbx = kFyhat * bz - kFzhat * by
            kcrossby = kFzhat * bx - kFxhat * bz
            kcrossbz = kFxhat * by - kFyhat * bx

            # self.test(kcrossbx,out='kcrossbx'+str(n))
            # self.test(kcrossby,out='kcrossby'+str(n))
            # self.test(kcrossbz,out='kcrossbz'+str(n))

            # kcrossbx *= (-1)**n
            # kcrossby *= (-1)**n
            # kcrossbz *= (-1)**n

            # polfac = np.sqrt(kcrossbx**2 + kcrossby**2 + kcrossbz**2) / (kFthat * bmag)
            sinzeta = sqrt(kcrossbx**2 + kcrossby**2 + kcrossbz**2) / (kFthat * bmag)
            # self.test(sinzeta)
            profile = self.jfunc(rvec, phivec, jargs)
            # profile=1.
            # 
            # print(self.profile)
            if self.polflux:
                polarizedintensity = sinzeta**(1.+spec) * delta**(3. + spec) * profile
            else:
                polarizedintensity = delta**(3+spec)*profile        
            # if INTENSITYISOTROPIC:
            #     intensity = delta**(3. + SPECTRALINDEX)
            # else:
            #     intensity = polarizedintensity
            
                
            pathlength = np.abs(kFthat/kFzhat)
            mag = polarizedintensity*pathlength
            # self.test(pathlength, out='pathlength'+str(n))
            # self.test(mag, out='mag'+str(n))
            # self.test(bmag)
            # self.test(kFthat)

            fFthat = 0.
            fFxhat = kcrossbx / (kFthat * bmag)
            fFyhat = kcrossby / (kFthat * bmag)
            fFzhat = kcrossbz / (kFthat * bmag)# * (-1)**n
            
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
            kOlth = rvec * radical * np.sign(sinphi) * (-1)**n
            #sinphi / (sqrt(sinphi**2)+1.e-10)

            xalpha = -kOlp / sintheta
            ybeta = kOlth
            nu = -xalpha
            den = sqrt((k1**2 + k2**2) * (ybeta**2 + nu**2))

            ealpha = (ybeta * k2 - nu * k1) / den
            ebeta = (ybeta * k1 + nu * k2) / den
            if self.polflux:
                qvec = -mag*(ealpha**2 - ebeta**2)# * (-1)**n
                uvec = -mag*(2*ealpha*ebeta)
                qvec[np.isnan(qvec)]=0
                uvec[np.isnan(uvec)]=0
                ivec = sqrt(qvec**2 + uvec**2)
            else:
                ivec = mag
                ivec[np.isnan(ivec)] = 0
                qvec = 0*ivec
                uvec = 0*ivec
            # self.test(ivec, 'ivec'+str(n))

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
        return ivecs, qvecs, uvecs#, rotimxvec, rotimyvec



    def observe_same(self, obs):
        if self.mode=='model':
            print("Can't observe_same in model mode!")
            return
        im = self.make_image(ra=obs.ra, dec=obs.dec, rf=obs.rf, mjd = obs.mjd, source=obs.source)
        return im.observe_same(obs)

    def modelim_ivis(self, uv, ttype='nfft'):
        return self.modelim.sample_uv(uv,ttype=ttype)[0]

    def modelim_logcamp(self, uv1, uv2, uv3, uv4, ttype='nfft'):
        vis12 = self.modelim_ivis(uv1,ttype=ttype)
        vis34 = self.modelim_ivis(uv2,ttype=ttype)
        vis23 = self.modelim_ivis(uv3,ttype=ttype)
        vis14 = self.modelim_ivis(uv4,ttype=ttype)
        amp12 = np.abs(vis12)
        amp34 = np.abs(vis34)
        amp23 = np.abs(vis23)
        amp14 = np.abs(vis14)
        logcamp_model = np.log(amp12)+np.log(amp34)-np.log(amp23)-np.log(amp14)
        return logcamp_model

    def modelim_cphase(self, uv1, uv2, uv3, ttype='nfft'):
        vis12 = self.modelim_ivis(uv1,ttype=ttype)
        vis23 = self.modelim_ivis(uv2,ttype=ttype)
        vis31 = self.modelim_ivis(uv3,ttype=ttype)
        phase12 = np.angle(vis12)
        phase23 = np.angle(vis23)
        phase31 = np.angle(vis31)
        cphase_model = phase12+phase23+phase31
        return cphase_model
    

    def build_likelihood(self, obs, data_types=['vis'], ttype='nfft'):
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
            visuv = np.vstack([u,v]).T
            Nvis = len(vis)
            print("Building vis likelihood!")
        if 'amp' in data_types:
            sigma = obs.data['sigma']
            amp = obs.unpack('amp', debias=True)['amp']
            u = obs.data['u']
            v = obs.data['v']
            ampuv = np.vstack([u,v]).T
            Namp = len(amp)
            print("Building amp likelihood!")
        if 'logcamp' in data_types:
            logcamp_data = obs.c_amplitudes(ctype='logcamp', debias=True)
            logcamp = logcamp_data['camp']
            logcamp_sigma = logcamp_data['sigmaca']
            campuv1, campuv2, campuv3, campuv4 = logcamp_uvpairs(logcamp_data)
            Ncamp = len(logcamp)
            print("Building logcamp likelihood!")
        if 'cphase' in data_types:
            cphase_data = obs.c_phases(ang_unit='rad')
            cphaseuv1, cphaseuv2, cphaseuv3 = cphase_uvpairs(cphase_data)
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
            ivecs, qvecs, uvecs = self.compute_image(imparams)#, rotimxvec, rotimyvec 
            out = 0.
            ivec = np.sum(ivecs,axis=0)
            qvec = np.sum(qvecs,axis=0)
            uvec = np.sum(uvecs,axis=0)

            self.modelim.ivec = ivec
            self.modelim.qvec = qvec
            self.modelim.uvec = uvec
            self.modelim.pa = to_eval[self.all_names.index('PA')]

            if 'vis' in data_types:
                sd = sqrt(sigma**2.0 + (to_eval[9]*amp)**2.0+to_eval[10]**2.0)
                model_vis = self.modelim_ivis(visuv, ttype=ttype)
                # model_vis = self.vis(ivec, rotimxvec, rotimyvec, u, v)
                # vislike = -1./(2.*Nvis) * np.sum(np.abs(model_vis-vis)**2 / sd**2)
                vislike = -np.sum(np.abs(model_vis-vis)**2 / sd**2)
                ln_norm = vislike-2*np.sum(np.log((2.0*np.pi)**0.5 * sd)) 
                out+=ln_norm
            if 'amp' in data_types:
                sd = sqrt(sigma**2.0 + (to_eval[9]*amp)**2.0+to_eval[10]**2.0)
                model_amp = np.abs(self.modelim_ivis(ampuv, ttype=ttype))
                # model_amp = np.abs(self.vis(ivec, rotimxvec, rotimyvec, u, v))
                # amplike = -1/Namp * np.sum(np.abs(model_amp-amp)**2 / sd**2)
                amplike = -0.5*np.sum((model_amp-amp)**2 / sd**2)
                ln_norm = amplike-np.sum(np.log((2.0*np.pi)**0.5 * sd)) 
                out+=ln_norm
            if 'logcamp' in data_types:
                model_logcamp = self.modelim_logcamp(campuv1, campuv2, campuv3, campuv4, ttype=ttype)
                # model_logcamp = self.logcamp(ivec, rotimxvec, rotimyvec, campu1, campu2, campu3, campu4, campv1, campv2, campv3, campv4)
                # logcamplike = -1./Ncamp * np.sum((logcamp-model_logcamp)**2 / logcamp_sigma**2)
                logcamplike = -0.5*np.sum((logcamp-model_logcamp)**2 / logcamp_sigma**2)
                ln_norm = logcamplike-np.sum(np.log((2.0*np.pi)**0.5 * logcamp_sigma)) 
                out += ln_norm
            if 'cphase' in data_types:
                model_cphase = self.modelim_cphase(cphaseuv1, cphaseuv2, cphaseuv3, ttype=ttype)
                # model_cphase = self.cphase(ivec, rotimxvec, rotimyvec, cphaseu1, cphaseu2, cphaseu3, cphasev1, cphasev2, cphasev3)
                # cphaselike = -2/Ncphase * np.sum((1-np.cos(cphase-model_cphase))/cphase_sigma)
                cphaselike = -0.5*np.sum((1-np.cos(cphase-model_cphase))/cphase_sigma)
                ln_norm = cphaselike -np.sum(np.log((2.0*np.pi)**0.5 * cphase_sigma))
                out += ln_norm
            return out/2
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
        self.dynamic=dynamic
        if dynamic:
            sampler = dynesty.DynamicNestedSampler(loglike, ptform,self.model_dim, periodic=self.periodic_indices, bound=bound, nlive=nlive)#, pool=pool, queue_size=queue_size)
        else:
            sampler = dynesty.NestedSampler(loglike, ptform, self.model_dim, periodic=self.periodic_indices, bound=bound, nlive=nlive)#, pool=pool, queue_size=queue_size)
        self.recent_sampler=sampler
        return sampler

    def setup(self, obs, data_types=['vis'],dynamic=False, nlive=1000, bound='multi', ttype='nfft'):#, pool=None, queue_size=None):
        self.source = obs.source
        self.modelim = eh.image.make_empty(self.npix,self.fov, ra=obs.ra, dec=obs.dec, rf= obs.rf, mjd = obs.mjd, source=obs.source)
        ptform = self.build_prior_transform()
        loglike = self.build_likelihood(obs, data_types=data_types, ttype=ttype)
        sampler = self.build_sampler(loglike,ptform,dynamic=dynamic, nlive=nlive, bound=bound)#, pool=pool, queue_size=queue_size)
        print("Ready to model with this BAM's recent_sampler! Call run_nested!")
        return sampler

    def run_nested(self, maxiter=None, maxcall=None, dlogz=None, logl_max=np.inf, n_effective=None, add_live=True, print_progress=True, print_func=None, save_bounds=True):
        self.recent_sampler.run_nested(maxiter=maxiter,maxcall=maxcall,dlogz=dlogz,logl_max=logl_max, n_effective=n_effective,add_live=add_live, print_progress=print_progress, print_func=None, save_bounds=True)
        self.recent_results = self.recent_sampler.results
        return self.recent_results

    def run_nested_default(self):
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

    def save_posterior(self, outname='Bam_posterior'):
        samples = self.recent_results.samples
        weights = np.exp(self.recent_results.logwt - self.recent_results.logz[-1])
        np.savetxt(outname+'_samples.txt',samples)
        np.savetxt(outname+'_weights.txt',weights)

    def MOP_Bam(self):
        mean, cov = self.mean_and_cov()
        to_eval = []
        for name in self.all_names:
            if not(name in self.modeled_names):
                to_eval.append(self.all_params[self.all_names.index(name)])
            else:
                to_eval.append(mean[self.modeled_names.index(name)])
        new = Bam(self.fov, self.npix, self.jfunc, self.jarg_names, to_eval[11:], to_eval[0], to_eval[1], to_eval[2], to_eval[3], PA=to_eval[4],  nmax=self.nmax, beta=to_eval[5], chi=to_eval[6], thetabz=to_eval[7], spec=to_eval[8], f=to_eval[9], e=to_eval[10], calctype=self.calctype,approxtype=self.approxtype, Mscale = self.Mscale, polflux=self.polflux,source=self.source)
        new.modelim = new.make_image(modelim=True)
        return new

    def resample_equal(self):
        samples = self.recent_results.samples
        weights = np.exp(self.recent_results.logwt - self.recent_results.logz[-1])
        resampled = dyfunc.resample_equal(samples,weights)
        return resampled

    def random_sample_Bam(self, samples=None, weights=None):
        if samples is None:
            samples = self.resample_equal()
        sample = samples[random.randint(0,len(samples)-1)]
        to_eval = []
        for name in self.all_names:
            if not(name in self.modeled_names):
                to_eval.append(self.all_params[self.all_names.index(name)])
            else:
                to_eval.append(sample[self.modeled_names.index(name)])
        new = Bam(self.fov, self.npix, self.jfunc, self.jarg_names, to_eval[11:], to_eval[0], to_eval[1], to_eval[2], to_eval[3], PA=to_eval[4],  nmax=self.nmax, beta=to_eval[5], chi=to_eval[6], thetabz=to_eval[7], spec=to_eval[8], f=to_eval[9], e=to_eval[10], calctype=self.calctype,approxtype=self.approxtype, Mscale = self.Mscale, polflux=self.polflux,source=self.source)
        new.modelim = new.make_image(modelim=True)
        return new


    def make_image(self, ra=M87_ra, dec=M87_dec, rf= 230e9, mjd = 57854, n='all', source = '', modelim=False):
        if source == '':
            source = self.source
        """
        Returns an ehtim Image object corresponding to the Blimage n0 emission
        """

        if self.mode == 'model':
            print("Cannot directly make images in model mode! Call sample_blimage or MAP_blimage and display that!")
            return
        # imparams = self.all_params[:9] + [self.all_params[11:]]
        # ivecs, qvecs, uvecs, rotimxvec, rotimyvec = self.compute_image(imparams)
        if n =='all':
            ivec = np.sum(self.ivecs,axis=0)
            qvec = np.sum(self.qvecs,axis=0)
            uvec = np.sum(self.uvecs,axis=0)
        elif type(n) is int:
            ivec = self.ivecs[n]
            qvec = self.qvecs[n]
            uvec = self.uvecs[n]

        im = eh.image.make_empty(self.npix,self.fov, ra=ra, dec=dec, rf= rf, mjd = mjd, source=source)
        im.ivec = ivec
        im.qvec = qvec
        im.uvec = uvec

        if modelim:
            im.pa = self.PA
        else:
            im = im.rotate(self.PA)
            mask = im.ivec<0
            im.ivec[mask]=0.
            im.qvec[mask]=0.
            im.uvec[mask]=0.

        # im.ivec *= self.tf / im.total_flux()
        return im



    def logcamp_chisq(self,obs):
        if self.mode != 'fixed':
            print("Can only compute chisqs to fixed model!")
            return
        if self.modelim is None:
            self.modelim = self.make_image(modelim=True)
        logcamp_data = obs.c_amplitudes(ctype='logcamp')
        sigmaca = logcamp_data['sigmaca']
        logcamp = logcamp_data['camp']
        campuv1, campuv2, campuv3, campuv4 = logcamp_uvpairs(logcamp_data)
        model_logcamp = self.modelim_logcamp(campuv1, campuv2, campuv3, campuv4)
        # model_logcamps = self.logcamp_fixed(logcamp_data['u1'],logcamp_data['u2'],logcamp_data['u3'],logcamp_data['u4'],logcamp_data['v1'],logcamp_data['v2'],logcamp_data['v3'],logcamp_data['v4'])
        logcamp_chisq = 1/len(sigmaca) * np.sum(np.abs((logcamp-model_logcamp)/sigmaca)**2)
        return logcamp_chisq

    def cphase_chisq(self,obs):
        if self.mode != 'fixed':
            print("Can only compute chisqs to fixed model!")
            return
        if self.modelim is None:
            self.modelim = self.make_image(modelim=True)
        cphase_data = obs.c_phases(ang_unit='rad')
        cphase = cphase_data['cphase']
        sigmacp = cphase_data['sigmacp']
        cphaseuv1, cphaseuv2, cphaseuv3 = cphase_uvpairs(cphase_data)
        model_cphase = self.modelim_cphase(cphaseuv1, cphaseuv2, cphaseuv3)
        # model_cphases = self.cphase_fixed(cphase_data['u1'],cphase_data['u2'],cphase_data['u3'],cphase_data['v1'],cphase_data['v2'],cphase_data['v3'])
        cphase_chisq = (2.0/len(sigmacp)) * np.sum((1.0 - np.cos(cphase-model_cphase))/(sigmacp**2))
        return cphase_chisq

    def vis_chisq(self,obs):
        if self.mode !='fixed':
            print("Can only compute chisqs to fixed model!")
            return
        if self.modelim is None:
            self.modelim = self.make_image(modelim=True)
        u = obs.data['u']
        v = obs.data['v']
        sigma = obs.data['sigma']  
        amp = obs.unpack('amp')['amp']
        vis = obs.data['vis']
        sd = np.sqrt(sigma**2.0 + (self.f*amp)**2.0 + self.e**2.0)

        uv = np.vstack([u,v]).T
        model_vis = self.modelim_ivis(uv)
        # model_vis = self.vis_fixed(u,v)
        absdelta = np.abs(model_vis-vis)
        vis_chisq = np.sum((absdelta/sd)**2)/(2*len(vis))
        return vis_chisq

    def amp_chisq(self,obs):
        if self.mode !='fixed':
            print("Can only compute chisqs to fixed model!")
            return
        if self.modelim is None:
            self.modelim = self.make_image(modelim=True)
        u = obs.data['u']
        v = obs.data['v']
        sigma = obs.data['sigma']  
        amp = obs.unpack('amp')['amp']
        # vis = obs.data['vis']
        sd = np.sqrt(sigma**2.0 + (self.f*amp)**2.0 + self.e**2.0)
        uv = np.vstack([u,v]).T
        model_amp = np.abs(self.modelim_ivis(uv))
        # model_amp = np.abs(self.vis_fixed(u,v))
        absdelta = np.abs(model_amp-amp)
        amp_chisq = np.sum((absdelta/sd)**2)/(len(amp))
        return amp_chisq


    def all_chisqs(self, obs):
        if self.mode !='fixed':
            print("Can only compute chisqs to fixed model!")
            return
        logcamp_chisq = self.logcamp_chisq(obs)
        cphase_chisq = self.cphase_chisq(obs)
        amp_chisq = self.amp_chisq(obs)
        vis_chisq = self.vis_chisq(obs)
        return {'logcamp':logcamp_chisq,'cphase':cphase_chisq,'vis':vis_chisq,'amp':amp_chisq}



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

