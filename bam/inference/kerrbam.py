import os
import sys
import numpy as np
import ehtim as eh
import matplotlib.pyplot as plt
import random
import dill as pkl
from bam.inference.model_helpers import Gpercsq, M87_ra, M87_dec, M87_mass, M87_dist, M87_inc, isiterable
from numpy import arctan2, sin, cos, exp, log, clip, sqrt,sign
import dynesty
from dynesty import plotting as dyplot
from dynesty import utils as dyfunc
from scipy.optimize import dual_annealing
from bam.inference.kerrexact import kerr_exact#, build_all_interpolators, Delta, R, Xi, omega, Sigma, getlorentzboost
# from bam.inference.schwarzschildexact import getscreencoords, getwindangle, getpsin, getalphan
# from bam.inference.gradients import LogLikeGrad, LogLikeWithGrad, exact_vis_loglike
# from ehtim.observing.pulses import deltaPulse2D


def get_uniform_transform(lower, upper):
    return lambda x: (upper-lower)*x + lower

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


class KerrBam:
    '''The Bam class is a collection of accretion flow and black hole parameters.
    jfunc: a callable that takes (r, phi, jargs)
    if Bam is in modeling mode, jfunc should use pm functions
    '''
    #class contains knowledge of a grid in Boyer-Lindquist coordinates, priors on each pixel, and the machinery to fit them
    def __init__(self, fov, npix, jfunc, jarg_names, jargs, M, D, a, inc, zbl, PA=0.,  nmax=0, beta=0., chi=0., eta = None, thetabz=np.pi/2, spec=1., f=0., e=0., Mscale = 2.e30*1.e9, polflux=True, source='', periodic=False):
        self.periodic=periodic
        self.dynamic=False
        self.source = source
        self.polflux = polflux
        # self.exacttype = exacttype
        # self.K_int = K_int
        # self.Fobs_int = Fobs_int
        # self.fobs_outer_int = fobs_outer_int
        # self.fobs_inner_ints = fobs_inner_ints
        # self.sn_outer_int = sn_outer_int
        # self.sn_inner_ints = sn_inner_ints
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
        self.a = a
        self.inc = inc
        self.PA = PA
        self.beta = beta
        self.chi = chi
        self.eta = eta
        self.thetabz = thetabz
        self.spec = spec
        self.f = f
        self.e = e
        self.nmax = nmax
        self.zbl = zbl
        self.rho_c = np.sqrt(27)
        self.Mscale = Mscale
        pxi = (np.arange(npix)-0.01)/npix-0.5
        pxj = np.arange(npix)/npix-0.5
        # get angles measured north of west
        PXI,PXJ = np.meshgrid(pxi,pxj)
        varphi = np.arctan2(-PXJ,PXI)# - np.pi/2
        varphi[varphi==0]=np.min(varphi[varphi>0])/10
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
        if any([isiterable(i) for i in [M, D, a, inc, zbl, PA, f, beta, chi, thetabz, e, spec]+jargs]):
            mode = 'model'
        else:
            mode = 'fixed' 
        self.mode = mode

        # self.all_interps = [K_int, Fobs_int, fobs_outer_int, fobs_inner_ints, sn_outer_int, sn_inner_ints]
        # self.all_interp_names = ['K','Fobs','fobs_outer','fobs_inner','sn_outer','sn_inner']
        self.all_params = [M, D, a, inc, zbl, PA, beta, chi,eta, thetabz, spec, f, e]+jargs
        self.all_names = ['M','D','a', 'inc','zbl','PA','beta','chi','eta','thetabz','spec','f','e']+jarg_names
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
            self.imparams = [self.M, self.D, self.a, self.inc, self.zbl, self.PA, self.beta, self.chi, self.eta, self.thetabz, self.spec, self.jargs]
            self.rhovec = D/(M*Mscale*Gpercsq*self.MUDISTS)
            # if self.exacttype=='interp' and all([not(self.all_interps[i] is None) for i in range(len(self.all_interps))]):
            print("Fixed Bam: precomputing all subimages.")
            self.ivecs, self.qvecs, self.uvecs, self.vvecs = self.compute_image(self.imparams)
                # ivecs, qvecs, uvecs, rotimxvec, rotimyvec = self.compute_image(imparams), self.rotimxvec, self.rotimyvec 
            # elif self.exacttype =='interp':
            #     print("Can't precompute subimages without all interpolators!")
        self.modelim = None
        # if self.exacttype=='interp':
        #     for i in range(len(self.all_interp_names)):
        #         if self.all_interps[i] is None:
        #             print(self.all_interp_names[i]+' does not have a specified interpolator!')
            
        print("Finished building KerrBam! in "+ self.mode +" mode!")#" with exacttype " +self.exacttype)

    # def guess_new_interpolators(self, ngrid=50):
    #     """
    #     Given known model parameters or prior ranges,
    #     specify a reasonable range of rho values and compute
    #     new interpolators.
    #     """
    #     print("Building all new interpolators from reasonable numbers (PLACEHOLDER FUNCTIONALITY)")
    #     K_int, Fobs_int, fobs_outer_int, fobs_inner_ints, sn_outer_int, sn_inner_ints = build_all_interpolators(ngrid=ngrid)
    #     self.K_int = K_int
    #     self.Fobs_int = Fobs_int
    #     self.fobs_outer_int = fobs_outer_int
    #     self.fobs_inner_ints = fobs_inner_ints
    #     self.sn_outer_int = sn_outer_int
    #     self.sn_inner_ints = sn_inner_ints
    #     self.all_interps = [K_int, Fobs_int, fobs_outer_int, fobs_inner_ints, sn_outer_int, sn_inner_ints]
    #     print("Finished building all interpolators.")
                            
    # def save_interpolators(self, outname=''):
    #     for i in range(len(self.all_interp_names)):
    #         with open(outname+self.all_interp_names[i]+'.pkl','wb') as myfile:
    #             pkl.dump(self.all_interps[i], myfile)
    #         # if type(self.all_interps[i]) is list:
    #         #     for j in range(len(self.all_interps[i])):
    #         #         with open(outname+all_interp_names[i]+'_'+str(j)+'.pkl','wb') as myfile:
    #         #             pkl.dump(self.all_interps[i][j], myfile)
    #         # else:
    #         #     with open(outname+all_interp_names[i]+'.pkl','wb') as myfile:
    #         #         pkl.dump(self.all_interps[i], myfile)

    # def load_interpolators(self, outname=''):
    #     for i in range(len(self.all_interp_names)):
    #         with open(outname+self.all_interp_names[i]+'.pkl','rb') as myfile:
    #             self.all_interps[i] = pkl.load(myfile)
    #     self.K_int, self.Fobs_int, self.fobs_outer_int, self.fobs_inner_ints, self.sn_outer_int, self.sn_inner_ints = self.all_interps
    #         # if 'inner' in all_interp_names[i]:
    #         #     self.all_interps[i] = []
    #         #     if 'fobs' in all_interp_names[i]:
    #         #         for j in range(2):
    #         #             with open(outname+all_interp_names[i]+'_'+str(j)+'.pkl','rb') as myfile:
    #         #                 self.all_interps[i].append(pkl.load(myfile))
    #         # else:
    #         #     with open(outname+all_interp_names[i]+'.pkl','wb') as myfile:
    #         #         pkl.dump(self.all_interps[i], myfile)

    def test(self, i, out):
        plt.close('all')
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
        M, D, a, inc, zbl, PA, beta, chi, eta, thetabz, spec, jargs = imparams

        
        #convert mudists to gravitational units
        rhovec = D / (M*self.Mscale*Gpercsq) * self.MUDISTS
        rvecs, ivecs, qvecs, uvecs, vvecs, redshifts = kerr_exact(rhovec, self.varphivec, inc, a, self.nmax, beta, chi, eta, thetabz)        
        # if self.exacttype == 'interp':
            
        # elif self.exacttype == 'exact':
        #     rvecs, ivecs, qvecs, uvecs, redshifts = kerr_exact(rhovec, self.varphivec, inc, a, self.nmax, beta, chi, eta, thetabz, interp=False)
            
        for n in range(self.nmax+1):
            profile = self.jfunc(rvecs[n], jargs) * redshifts[n]**(3+spec)
            if self.polflux:
                ivecs[n]*=profile
                qvecs[n]*=profile
                uvecs[n]*=profile
                vvecs[n]*=profile
            else:
                ivecs[n] = profile
                qvecs[n] = ivecs[n]*0
                uvecs[n] = ivecs[n]*0
                vvecs[n] = ivecs[n]*0

        tf = np.sum(ivecs)
        ivecs = [ivec*zbl/tf for ivec in ivecs]
        qvecs = [qvec*zbl/tf for qvec in qvecs]
        uvecs = [uvec*zbl/tf for uvec in uvecs]
        vvecs = [vvec*zbl/tf for vvec in vvecs]
        return ivecs, qvecs, uvecs, vvecs #, rotimxvec, rotimyvec



    def observe_same(self, obs, ampcal=True,phasecal=True,add_th_noise=True,seed=None):
        if seed is None:
            seed = random.randrange(sys.maxsize)
        if self.mode=='model':
            print("Can't observe_same in model mode!")
            return
        im = self.make_image(ra=obs.ra, dec=obs.dec, rf=obs.rf, mjd = obs.mjd, source=obs.source)
        return im.observe_same(obs, ampcal=ampcal,phasecal=phasecal, add_th_noise=add_th_noise, seed=seed)

    def modelim_ivis(self, uv, ttype='nfft'):
        return self.modelim.sample_uv(uv,ttype=ttype)[0]

    def modelim_allvis(self, uv, ttype='nfft'):
        return self.modelim.sample_uv(uv,ttype=ttype)

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
        if 'qvis' in data_types:
            qvis = obs.data['qvis']
            qsigma = obs.data['qsigma']
            qamp = np.abs(qvis)
            u = obs.data['u']
            v = obs.data['v']
            visuv = np.vstack([u,v]).T
            Nqvis = len(qvis)
        if 'uvis' in data_types:
            uvis = obs.data['uvis']
            usigma = obs.data['usigma']
            uamp = np.abs(uvis)
            u = obs.data['u']
            v = obs.data['v']
            visuv = np.vstack([u,v]).T
            Nuvis = len(uvis)
        if 'vvis' in data_types:
            vvis = obs.data['vvis']
            vsigma = obs.data['vsigma']
            vamp = np.abs(vvis)
            u = obs.data['u']
            v = obs.data['v']
            visuv = np.vstack([u,v]).T
            Nvvis = len(vvis)
        if 'mvis' in data_types:
            vis = obs.data['vis']
            qvis = obs.data['qvis']
            uvis = obs.data['uvis']
            pvis = qvis+1j*uvis
            sigma = obs.data['sigma']
            mvis = pvis/vis
            msigma = sigma * np.sqrt(2/np.abs(vis)**2 + np.abs(pvis)**2 / np.abs(vis)**4)
            u = obs.data['u']
            v = obs.data['v']
            visuv = np.vstack([u,v]).T
            Nmvis = len(mvis)
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
            #so it should have 13+N parameters where N is the number of jargs
            #M, D, inc, zbl, PA, beta, chi, thetabz, spec, f, e + jargs
            #f and e are not used in image computation, so slice around them for now
            imparams = to_eval[:11] + [to_eval[13:]]
            ivecs, qvecs, uvecs, vvecs = self.compute_image(imparams)#, rotimxvec, rotimyvec 
            out = 0.
            ivec = np.sum(ivecs,axis=0)
            qvec = np.sum(qvecs,axis=0)
            uvec = np.sum(uvecs,axis=0)
            vvec = np.sum(vvecs,axis=0)

            self.modelim.ivec = ivec
            self.modelim.qvec = qvec
            self.modelim.uvec = uvec
            self.modelim_vvec = vvec
            self.modelim.pa = to_eval[self.all_names.index('PA')]
            if 'vis' in data_types or 'qvis' in data_types or 'uvis' in data_types or 'vvis' in data_types or 'mvis' in data_types:
                model_ivis, model_qvis, model_uvis, model_vvis = self.modelim_allvis(visuv, ttype=ttype)
                if 'mvis' in data_types:
                    model_mvis = (model_qvis+1j*model_uvis)/model_ivis
                

            if 'vis' in data_types:
                sd = sqrt(sigma**2.0 + (to_eval[11]*amp)**2.0+to_eval[12]**2.0)
                # model_vis = self.modelim_ivis(visuv, ttype=ttype)
                vislike = -np.sum(np.abs(model_ivis-vis)**2 / sd**2)
                ln_norm = vislike-2*np.sum(np.log((2.0*np.pi)**0.5 * sd)) 
                out+=ln_norm
            if 'qvis' in data_types:
                sd = sqrt(qsigma**2.0 +(to_eval[11]*qamp)**2.0+to_eval[12]**2.0)
                qvislike = -np.sum(np.abs(model_qvis-qvis)**2.0/sd**2)
                ln_norm = qvislike-2*np.sum(np.log((2.0*np.pi)**0.5*sd))
                out += ln_norm
            if 'uvis' in data_types:
                sd = sqrt(usigma**2.0 +(to_eval[11]*uamp)**2.0+to_eval[12]**2.0)
                uvislike = -np.sum(np.abs(model_uvis-uvis)**2.0/sd**2)
                ln_norm = uvislike-2*np.sum(np.log((2.0*np.pi)**0.5*sd))
                out += ln_norm
            if 'vvis' in data_types:
                sd = sqrt(vsigma**2.0 +(to_eval[11]*vamp)**2.0+to_eval[12]**2.0)
                vvislike = -np.sum(np.abs(model_vvis-vvis)**2.0/sd**2)
                ln_norm = vvislike-2*np.sum(np.log((2.0*np.pi)**0.5*sd))
                out += ln_norm
            if 'mvis' in data_types:
                sd = sqrt(sigma**2.0 + (to_eval[11]*amp)**2.0+to_eval[12]**2.0)
                sd = vsigma*sd/sigma
                mvislike = -np.sum(np.abs(model_mvis-mvis)**2.0/sd**2)
                ln_norm = mvislike -2*np.sum(np.log((2.0*np.pi)**0.5*sd))
                out+=ln_norm
            if 'amp' in data_types:
                sd = sqrt(sigma**2.0 + (to_eval[11]*amp)**2.0+to_eval[12]**2.0)
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
            return out
        print("Built combined likelihood function!")
        return loglike

    def annealing_MAP(self, obs, data_types=['vis'], ttype='nfft', args=(), maxiter=1000,local_search_options={},initial_temp=5230.0):
        """
        Given an observation and a list of data product names, 
        find the MAP using scipy's dual annealing.
        """
        self.source = obs.source
        self.modelim = eh.image.make_empty(self.npix,self.fov, ra=obs.ra, dec=obs.dec, rf= obs.rf, mjd = obs.mjd, source=obs.source)#, pulse=deltaPulse2D)
        ll = self.build_likelihood(obs, data_types=data_types,ttype=ttype)
        

        res =  dual_annealing(lambda x: -ll(x), self.modeled_params, args=args, maxiter=maxiter, local_search_options=local_search_options, initial_temp=initial_temp)
        
        to_eval = []
        for name in self.all_names:
            if not(name in self.modeled_names):
                to_eval.append(self.all_params[self.all_names.index(name)])
            else:
                to_eval.append(res.x[self.modeled_names.index(name)])
        # new = KerrBam(self.fov, self.npix, self.jfunc, self.jarg_names, to_eval[13:], to_eval[0], to_eval[1], to_eval[2], to_eval[3], to_eval[4], PA=to_eval[5],  nmax=self.nmax, beta=to_eval[6], chi=to_eval[7], eta = to_eval[8], thetabz=to_eval[9], spec=to_eval[10], f=to_eval[11], e=to_eval[12],exacttype=self.exacttype, K_int = self.K_int, Fobs_int = self.Fobs_int, fobs_outer_int = self.fobs_outer_int, fobs_inner_ints = self.fobs_inner_ints, sn_outer_int = self.sn_outer_int, sn_inner_ints = self.sn_inner_ints, Mscale = self.Mscale, polflux=self.polflux,source=self.source)
        new = KerrBam(self.fov, self.npix, self.jfunc, self.jarg_names, to_eval[13:], to_eval[0], to_eval[1], to_eval[2], to_eval[3], to_eval[4], PA=to_eval[5],  nmax=self.nmax, beta=to_eval[6], chi=to_eval[7], eta = to_eval[8], thetabz=to_eval[9], spec=to_eval[10], f=to_eval[11], e=to_eval[12], Mscale = self.Mscale, polflux=self.polflux,source=self.source)
        new.modelim = new.make_image(modelim=True)
        return new
        

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
        self.modelim = eh.image.make_empty(self.npix,self.fov, ra=obs.ra, dec=obs.dec, rf= obs.rf, mjd = obs.mjd, source=obs.source)#, pulse=deltaPulse2D)
        ptform = self.build_prior_transform()
        loglike = self.build_likelihood(obs, data_types=data_types, ttype=ttype)
        sampler = self.build_sampler(loglike,ptform,dynamic=dynamic, nlive=nlive, bound=bound)#, pool=pool, queue_size=queue_size)
        print("Ready to model with this BAM's recent_sampler! Call run_nested!")
        return sampler

    def run_nested(self, maxiter=None, maxcall=None, dlogz=None, logl_max=np.inf, n_effective=None, add_live=True, print_progress=True, print_func=None, save_bounds=True, maxbatch=None):
        if self.dynamic:
            n_effective = np.inf if n_effective is None else n_effective
            dlogz = 0.01 if dlogz is None else dlogz
            self.recent_sampler.run_nested(maxiter_init=maxiter,maxcall_init=maxcall,dlogz_init=dlogz,logl_max_init=logl_max, n_effective_init=n_effective, print_progress=print_progress, print_func=None, save_bounds=True, maxbatch=maxbatch)
        else:
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


    def cornerplot(self, save='',show=True, truths=None):
        fig, axes = dyplot.cornerplot(self.recent_results, labels=self.modeled_names, truths=truths)
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

    def pickle_result(self, outname='results'):
        with open(outname+'.pkl','wb') as myfile:
            pkl.dump(self.recent_results, myfile)


    def MOP_Bam(self):
        mean, cov = self.mean_and_cov()
        to_eval = []
        for name in self.all_names:
            if not(name in self.modeled_names):
                to_eval.append(self.all_params[self.all_names.index(name)])
            else:
                to_eval.append(mean[self.modeled_names.index(name)])
        new = KerrBam(self.fov, self.npix, self.jfunc, self.jarg_names, to_eval[13:], to_eval[0], to_eval[1], to_eval[2], to_eval[3], to_eval[4], PA=to_eval[5],  nmax=self.nmax, beta=to_eval[6], chi=to_eval[7], eta = to_eval[8], thetabz=to_eval[9], spec=to_eval[10], f=to_eval[11], e=to_eval[12], Mscale = self.Mscale, polflux=self.polflux,source=self.source)
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
        new = KerrBam(self.fov, self.npix, self.jfunc, self.jarg_names, to_eval[13:], to_eval[0], to_eval[1], to_eval[2], to_eval[3], to_eval[4], PA=to_eval[5],  nmax=self.nmax, beta=to_eval[6], chi=to_eval[7], eta = to_eval[8], thetabz=to_eval[9], spec=to_eval[10], f=to_eval[11], e=to_eval[12], Mscale = self.Mscale, polflux=self.polflux,source=self.source)
        new.modelim = new.make_image(modelim=True)
        return new


    def make_image(self, ra=M87_ra, dec=M87_dec, rf= 230e9, mjd = 57854, n='all', source = '', modelim=False):
        if source == '':
            source = self.source

        if self.mode == 'model':
            print("Cannot directly make images in model mode!")
            return
        try:
            self.ivecs
        except:
            self.ivecs, self.qvecs, self.uvecs, self.vvecs = self.compute_image(self.imparams)
        
                
        # imparams = self.all_params[:9] + [self.all_params[11:]]
        # ivecs, qvecs, uvecs, rotimxvec, rotimyvec = self.compute_image(imparams)
        if n =='all':
            ivec = np.sum(self.ivecs,axis=0)
            qvec = np.sum(self.qvecs,axis=0)
            uvec = np.sum(self.uvecs,axis=0)
            vvec = np.sum(self.vvecs,axis=0)
        elif type(n) is int:
            ivec = self.ivecs[n]
            qvec = self.qvecs[n]
            uvec = self.uvecs[n]
            vvec = self.vvecs[n]

        im = eh.image.make_empty(self.npix,self.fov, ra=ra, dec=dec, rf= rf, mjd = mjd, source=source)#, pulse=deltaPulse2D)
        im.ivec = ivec
        im.qvec = qvec
        im.uvec = uvec
        im.vvec = vvec

        if modelim:
            im.pa = self.PA
        else:
            # im = im.rotate(self.PA)
            im.pa = self.PA
            mask = im.ivec<0
            im.ivec[mask]=0.
            im.qvec[mask]=0.
            im.uvec[mask]=0.
            im.vvec[mask]=0.

        # im.ivec *= self.tf / im.total_flux()
        return im

    def make_rotated_image(self, ra=M87_ra, dec=M87_dec, rf= 230e9, mjd = 57854, n='all', source = ''):
        out = self.make_image(ra=ra,dec=dec,rf=rf, mjd=mjd, n=n, source=source,modelim=False).rotate(self.PA)
        out.pa = 0
        return out

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

