import os
import numpy as np
import ehtim as eh
import matplotlib.pyplot as plt
import random
import pickle as pkl
from bam.inference.model_helpers import Gpercsq, M87_ra, M87_dec, M87_mass, M87_dist, M87_inc, isiterable
from numpy import arctan2, sin, cos, exp, log, clip, sqrt,sign
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
    return lambda x: (upper-lower)*x - lower



class Bam:
    '''The Bam class is a collection of accretion flow and black hole parameters.
    jfunc: a callable that takes (r, phi, jargs)
    if Bam is in modeling mode, jfunc should use pm functions
    '''
    #class contains knowledge of a grid in Boyer-Lindquist coordinates, priors on each pixel, and the machinery to fit them
    def __init__(self, fov, npix, jfunc, jarg_names, jargs, M, D, inc, zbl, PA=0.,  nmax=0, beta=0., chi=0., thetabz=np.pi/2, spec=1., f=0., e=0., calctype='approx'):

        self.fov = fov
        self.npix = npix
        self.MAP_estimate = None
        # self.MAP_values = None
        self.jfunc = jfunc
        self.jarg_names = jarg_names
        self.jargs = jargs
        self.trace = None
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
        self.modeled_indices = [i for i in range(len(self.all_params)) if isterable(self.all_params[i])]
        self.modeled_names = [i for i in self.all_names if isiterable(self.all_params[i])]
        self.modeled_params = [i for i in self.all_params if isiterable(i)]
        self.periodic_names = []
        self.periodic_indices=[]
        #if PA and chi are being modeled, check if they are allowing a full 2pi wrap
        #if so, let dynesty know they are periodic later
        for i in ['PA','chi']:
            if i in self.modeled_names:
                bounds = self.modeled_params[self.modeled_names.index(i)]
                if np.isclose(np.exp(1j*bounds[0]),np.exp(1j*bounds(bounds[1])),rtol=1e-12):
                    print("Found periodic prior on "+str(i))
                    self.periodic_names.append(i)
                    self.periodic_indices.append(self.modeled_names.index(i))

        # self.periodic_names = [i for i in ['PA','chi'] if i in self.modeled_names]
        # self.periodic_indices = [self.modeled_names.index(i) for i in self.periodic_names]


        print("Finished building Bam! in "+ self.mode +" mode with calctype " +self.calctype)  
        

    def compute_image(self, imparams):
        """
        Given a list of values of modeled parameters in imparams,
        compute the resulting qvec, uvec, ivec, rotimxvec, and rotimyvec.
        """
        M, D, inc, zbl, PA, beta, chi, thetabz, spec, jargs = imparams

        # self.M = M
        #this is the end of the problem setup;
        #everything after this point should bifurcate depending on calctype
        if calctype == 'approx':

             # def rho_conv(r, phi, D, theta0, r_g):
            #     rho2 = (((r/D)**2.0)*(1.0 - ((sin(theta0)**2.0)*(sin(phi)**2.0)))) + ((2.0*r*r_g/(D**2.0))*((1.0 + (sin(theta0)*sin(phi)))**2.0))
            #     rho = sqrt(rho2)
            #     return rho

            # def emission_coordinates(rho, varphi):
                


            #convert mudists to gravitational units
            rho = D / (M*Gpercsq) * self.MUDISTS
            

            phivec = arctan2(sin(self.varphivec),cos(self.varphivec)*cos(inc))
            sinprod = sin(inc)*sin(phivec)
            numerator = 1.+rho**2 - (-3.+rho**2.)*sinprod+3.*sinprod**2. + sinprod**3. 
            denomenator = (-1.+sinprod)**2 * (1+sinprod)
            sqq = sqrt(numerator/denomenator)
            rvec = (1.-sqq + sinprod*(1.+sqq))/(sinprod-1.)
             

            # rvec, phivec = emission_coordinates(self.rhovec, self.varphivec)
            rvec = clip(rvec, 2.+1.e-5,np.inf)
            # rvec = ((rvec-2)+sqrt((rvec-2)**2))/2.+2.0001
            # rvec = maximum(rvec, 2.00001)
            #begin Ramesh formalism
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
            gfac = sqrt(1. - 2./rvec)
            gfacinv = 1. / gfac
            gamma = 1. / sqrt(1. - beta**2)

            sintheta = sin(inc)
            # print(inc)
            # print(sintheta)
            costheta = cos(inc)
            sinphi = sin(phivec)
            cosphi = cos(phivec)

            cospsi = -sintheta * sinphi
            sinpsi = sqrt(1. - cospsi**2)

            cosalpha = 1. - (1. - cospsi) * (1. - 2./rvec)
            sinalpha =sqrt(1. - cosalpha**2)

            
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
            radican = np.maximum(radicand,0.)
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
            ivec = sqrt(qvec**2 + uvec**2)

            tf = np.sum(ivec)
            qvec = qvec * zbl/tf
            uvec = uvec * zbl/tf
            ivec = ivec * zbl/tf
            #now do the rotation

            rotimxvec = cos(PA)*self.imxvec - sin(PA)*self.imyvec
            rotimyvec = sin(PA)*self.imxvec + cos(PA)*self.imyvec
            return ivec, qvec, uvec, rotimxvec, rotimyvec
        else:
            pass

    def vis(self, vec, rotimxvec, rotimyvec, u, v):#, vis_types=list('i')):

        u = np.array(u)
        v = np.array(v)

        matrix = np.outer(u, rotimxvec)+np.outer(v, rotimyvec)
        A_real = np.cos(2.0*np.pi*matrix)
        A_imag = np.sin(2.0*np.pi*matrix)
        visreal_model = np.dot(A_real,vec)
        visimag_model = np.dot(A_imag,vec)
        return visreal_model+1j*visimag_model

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










    def make_image(self, ra=M87_ra, dec=M87_dec, rf= 230e9, mjd = 57854, source='M87'):
        """
        Returns an ehtim Image object corresponding to the Blimage n0 emission
        """

        if self.mode == 'model':
            print("Cannot directly make images in model mode! Call sample_blimage or MAP_blimage and display that!")
            return

        im = eh.image.make_empty(self.npix,self.fov, ra=ra, dec=dec, rf= rf, mjd = mjd, source='M87')
        im.ivec = self.ivec
        im.qvec = self.qvec
        im.uvec = self.uvec

        im = im.rotate(self.PA)
        # im.ivec *= self.tf / im.total_flux()
        return im

    def MAP_Bam(self):
        ME = self.MAP_estimate
        if isiterable(self.M):
            M_new = float(ME['M'])
        else:
            M_new = self.M
        # self.
        if isiterable(self.D):
            D_new = float(ME['D'])
        else:
            D_new = self.D

        if isiterable(self.inc):
            inc_new = float(ME['inc'])
        else:
            inc_new = self.inc

        if isiterable(self.zbl):
            zbl_new = float(ME['zbl'])
        else:
            zbl_new = self.zbl

        if isiterable(self.PA):
            PA_new = float(ME['PA'])
        else:
            PA_new = self.PA

        if isiterable(self.beta):
            beta_new = float(ME['beta'])
        else:
            beta_new = self.beta

        if isiterable(self.chi):
            chi_new = float(ME['chi'])
        else:
            chi_new = self.chi

        if isiterable(self.spec):
            spec_new = float(ME['spec'])
        else:
            spec_new = self.spec

        if isiterable(self.thetabz):
            thetabz_new = float(ME['thetabz'])
        else:
            thetabz_new = self.thetabz

        if isiterable(self.f):
            f_new = float(ME['f'])
        else:
            f_new = self.f

        if isiterable(self.e):
            e_new = float(ME['e'])
        else:
            e_new = self.e

        # jarg_names = self.jarg_names

        # for jargi in range(len(self.jargs)):
        #     jarg = jargs[jargi]
        #     if isiterable(jarg):
        #         jargs.append(pm.Uniform(jarg_names[jargi],lower=jarg[0],upper=jarg[1]))
        #     else:
        #         jargs.append(jarg)
       


        new_blimage = Blimage(self.r_lims, self.phi_lims, self.nr, self.nphi, M_new, D_new, inc_new, 0, zbl_new, PA=PA_new, beta=beta_new, chi=chi_new, spec=spec_new, nmax = self.nmax,e=e_new,f=f_new)
  
        
        if isiterable(self.j):
            new_blixels = ME['blixels']
        else:
            new_blixels = self.blixels.copy()

        new_blimage.blixels = new_blixels
        return new_blimage


    def random_sample_blimage(self):
        ME = self.trace[int(round(random.random()*len(self.trace)))]
        if isiterable(self.M):
            M_new = float(ME['M'])
        else:
            M_new = self.M
        # self.
        if isiterable(self.D):
            D_new = float(ME['D'])
        else:
            D_new = self.D

        if isiterable(self.inc):
            inc_new = float(ME['inc'])
        else:
            inc_new = self.inc

        if isiterable(self.zbl):
            zbl_new = float(ME['zbl'])
        else:
            zbl_new = self.zbl

        if isiterable(self.PA):
            PA_new = float(ME['PA'])
        else:
            PA_new = self.PA

        if isiterable(self.beta):
            beta_new = float(ME['beta'])
        else:
            beta_new = self.beta

        if isiterable(self.chi):
            chi_new = float(ME['chi'])
        else:
            chi_new = self.chi

        if isiterable(self.spec):
            spec_new = float(ME['spec'])
        else:
            spec_new = self.spec

        if isiterable(self.f):
            f_new = float(ME['f'])
        else:
            f_new = self.f

        if isiterable(self.e):
            e_new = float(ME['e'])
        else:
            e_new = self.e

        #make a new blimage with zero emissivity but same dimensions
        new_blimage = Blimage(self.r_lims, self.phi_lims, self.nr, self.nphi, M_new, D_new, inc_new, 0, zbl_new, PA=PA_new, beta=beta_new, chi=chi_new, spec=spec_new, nmax = self.nmax, f= f_new, e=e_new)
        #now go through and reset its blixels
        # r = np.sqrt(np.linspace(self.r_lims[0]**2, self.r_lims[1]**2, self.nr+1)[:-1])
        # phi = np.linspace(self.phi_lims[0],self.phi_lims[1], self.nphi+1)[:-1]
        # new_blixels = []
        # r_grid, phi_grid = np.meshgrid(r,phi)
        # rvec = r_grid.flatten()
        # phivec = phi_grid.flatten()
        
        if isiterable(self.j):
            # name = 'j_r'+str(r_index)+'_phi'+str(phi_index)
            new_blixels = ME['blixels']
            # append(Blixel(rval, phival, float(ME[name]), M_new, D_new, inc_new, mode='fixed'))
        else:
            new_blixels = self.blixels.copy()

        new_blimage.blixels = new_blixels
        return new_blimage


    def save_blimage(self,blimage_name):
        """
        Saves blimage parameters as a pickled dictionary.
        """
        out = dict()
        out['r_lims'] = self.r_lims
        out['phi_lims'] = self.phi_lims
        out['nr'] = self.nr
        out['nphi'] = self.nphi
        out['M'] = self.M
        out['D'] = self.D
        out['inc'] = self.inc
        if isiterable(self.j):
            out['j'] = self.j
        else:
            out['j'] = -1
            out['blixels'] = self.blixels
        out['PA'] = self.PA
        out['beta'] = self.beta
        out['chi'] = self.chi
        out['spec'] = self.spec
        out['f'] = self.f
        out['e'] = self.e
        out['nmax'] = self.nmax
        out['zbl'] = self.zbl
        with open(blimage_name, 'wb') as file:
            pkl.dump(out, file)

        
    def plot_raw_posterior(self, var_names = None, kind='hist'):
        """
        Thin wrapper around the pymc3 plot_posterior method.
        """
        pm.plot_posterior(self.trace, var_names = var_names, kind=kind)

    # def plot_scaled_singlevar_posterior(self, var_name, scale=1., title='', units= '',save=False):
    #     """
    #     Applies unit scaling to a posterior before plotting.
    #     """
    #     plot_custom_singlevar_posterior(self.trace, var_name, scale=scale, title=title, units= units, save=save)

    def save_trace(self, directory='./traces/',overwrite=False):
        """
        Saves self.trace to the specified directory
        """
        pm.save_trace(self.trace, directory = directory, overwrite=overwrite)

    def load_trace(self,directory):
        """
        Loads a trace, and saves it to the trace attribute (for resuming sampling).
        """
        if mode != 'model':
            print("Can't load trace into fixed blimage!")
            return
        with self.model:
            self.trace = pm.load_trace(directory=directory)



    # def plot_MAP_frames(self, FOV, npix, t=[0.],save=False, out_dir = './', scale=1, units ='', clean_edges=False):
    #     """
    #     Plots the model with the given FOV divided over npix pixels at the specified frames.
    #     """
    #     plot_frames(self.image_func, self.MAP_values, FOV, npix, t=t,save=save, out_dir=out_dir, scale=scale, units=units,clean_edges=clean_edges)

    # def ray_trace(self, FOV, npix, )



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
