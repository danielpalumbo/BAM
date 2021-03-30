import os
import pymc3 as pm 
import numpy as np
import ehtim as eh
import matplotlib.pyplot as plt
import theano
import random
import theano.tensor as tt
import pickle as pkl
from bam.inference.model_helpers import Gpercsq, M87_ra, M87_dec, M87_mass, M87_dist, M87_inc, isiterable
import eht_dmc as dmc
# from bam.inference.schwarzschildexact import getscreencoords, getwindangle, getpsin, getalphan
# from bam.inference.gradients import LogLikeGrad, LogLikeWithGrad, exact_vis_loglike
# theano.config.exception_verbosity='high'
theano.config.compute_test_value = 'ignore'

# def example_fixed_jfunc(r, phi, jargs):
#     peak_r = jargs[0]
#     thickness = jargs[1]
#     return np.exp(-4.*np.log(2)*((r-peak_r)/thickness)**2)

# def example_model_jfunc(r, phi, jargs):
#     peak_r = jargs[0]
#     thickness = jargs[1]
#     return pm.math.exp(-4.*np.log(2)*((r-peak_r)/thickness)**2)


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


        if self.mode == 'model':
            self.varphivec = theano.shared(self.varphivec)
            self.MUDISTS = theano.shared(self.MUDISTS)
            self.imxvec = theano.shared(self.imxvec)
            self.imyvec = theano.shared(self.imyvec)
            summate = pm.math.sum
            sign = pm.math.sgn
            clip = pm.math.clip
            maximum = pm.math.maximum
            cos = pm.math.cos
            sin = pm.math.sin
            arccos = tt.basic.arccos
            arcsin = tt.basic.arcsin
            arctan2 = tt.basic.arctan2
            sqrt = pm.math.sqrt
            exp = pm.math.exp
            log = pm.math.log
            model = pm.Model()
            with model:
                #build priors
                if isiterable(M):
                    M_prior = pm.Uniform('M',lower=M[0],upper=M[1])#, testval=np.mean([M[0],M[1]]))
                    # self.modeled_priors.append(M_prior)
                else:
                    M_prior = M
                # self.
                if isiterable(D):
                    D_prior = pm.Uniform('D',lower=D[0],upper=D[1])#, testval=np.mean([D[0],D[1]]))
                    # self.modeled_priors.append(D_prior)
                else:
                    D_prior = D

                if isiterable(inc):
                    inc_prior = pm.Uniform('inc',lower=inc[0],upper=inc[1])#, testval=np.mean([inc[0],inc[1]]))
                    # self.modeled_priors.append(inc_prior)
                else:
                    inc_prior = inc

                if isiterable(zbl):
                    zbl_prior = pm.Uniform('zbl',lower=zbl[0],upper=zbl[1])#, testval=np.mean([zbl[0],zbl[1]]))
                    # self.modeled_priors.append(zbl_prior)
                else:
                    zbl_prior = zbl

                if isiterable(PA):
                    mu = (PA[0]+PA[1])/2.
                    diff = np.abs(PA[1]-PA[0])
                    kappa = 1/diff**2
                    # PA_prior = pm.Uniform('PA',lower=PA[0],upper=PA[1])
                    PA_prior = pm.VonMises('PA',mu=mu, kappa=kappa)#,testval=mu)
                    # self.modeled_priors.append(PA_prior)
                else:
                    PA_prior = PA

                if isiterable(beta):
                    beta_prior = pm.Uniform('beta',lower=beta[0], upper=beta[1])#, testval=np.mean([beta[0],beta[1]]))
                    # self.modeled_priors.append(beta_prior)
                else:
                    beta_prior = beta

                if isiterable(chi):
                    mu = (chi[0]+chi[1])/2
                    diff = np.abs(chi[1]-chi[0])
                    kappa = 1/diff**2
                    # PA_prior = pm.Uniform('PA',lower=PA[0],upper=PA[1])
                    chi_prior = pm.VonMises('chi',mu=mu, kappa=kappa)#,testval = mu)
                    # self.modeled_priors.append(chi_prior)
                else:
                    chi_prior = chi

                if isiterable(thetabz):
                    thetabz_prior = pm.Uniform('thetabz',lower=theatbz[0],upper=thetabz[1])
                else:
                    thetabz_prior = thetabz

                if isiterable(spec):
                    spec_prior=pm.Uniform('spec',lower=spec[0],upper=spec[1])#, testval = np.mean([spec[0],spec[1]]))
                    # self.modeled_priors.append(spec_prior)
                else:
                    spec_prior = spec


                if isiterable(f) and calctype=='approx':
                    f_prior = pm.Uniform('f',lower=f[0],upper=f[1])#, testval = np.mean([f[0],f[1]]))
                else:
                    f_prior = f
                    if calctype == 'exact' and isiterable(f):
                        print("Can't fit f in exact mode; using f = 0")
                        f_prior = 0

                if isiterable(e) and calctype=='approx':
                    e_prior = pm.Uniform('e',lower=e[0],upper=e[1])#, testval = np.mean([e[0],e[1]]))
                    # self.modeled_priors.append(e_prior)
                else:
                    e_prior = e
                    if calctype == 'exact' and isiterable(e):
                        print("Can't fit e in exact mode; using e = 0")
                        e_prior = 0
                jarg_priors = []
                for jargi in range(len(jargs)):
                    jarg = jargs[jargi]
                    if isiterable(jarg):
                        jarg_priors.append(pm.Uniform(jarg_names[jargi],lower=jarg[0],upper=jarg[1]))
                    else:
                        jarg_priors.append(jarg)
               
            self.M_prior = M_prior
            self.D_prior = D_prior
            self.inc_prior=inc_prior
            self.zbl_prior=zbl_prior
            self.PA_prior = PA_prior
            self.beta_prior=beta_prior
            self.chi_prior = chi_prior
            self.thetabz_prior = thetabz_prior
            self.spec_prior = spec_prior
            self.f_prior = f_prior
            self.e_prior = e_prior
            self.jarg_priors = jarg_priors


            self.model=model
        else:
            summate = np.sum
            sign = np.sign
            maximum = np.maximum
            clip = np.clip
            cos = np.cos
            sin = np.sin
            arccos = np.arccos
            arcsin = np.arcsin
            arctan2 = np.arctan2
            sqrt = np.sqrt
            exp  = np.exp
            log  = np.log
            M_prior = M
            D_prior = D
            inc_prior = inc
            PA_prior = PA
            zbl_prior = zbl
            beta_prior = beta
            chi_prior = chi
            thetabz_prior = thetabz
            spec_prior = spec
            jarg_priors = jargs
            f_prior = f
            e_prior = e

        self.sys_err = f_prior
        self.abs_err = e_prior
        self.tf = zbl_prior
        self.r_g = M_prior * Gpercsq
        # self.D_prior = D_prior
        # self.M_prior = M_prior
        #this is the end of the problem setup;
        #everything after this point should bifurcate depending on calctype
        if calctype == 'approx':

            # def rho_conv(r, phi, D, theta0, r_g):
            #     rho2 = (((r/D)**2.0)*(1.0 - ((sin(theta0)**2.0)*(sin(phi)**2.0)))) + ((2.0*r*r_g/(D**2.0))*((1.0 + (sin(theta0)*sin(phi)))**2.0))
            #     rho = sqrt(rho2)
            #     return rho

            # def emission_coordinates(rho, varphi):
                


            #convert mudists to gravitational units
            rho = D_prior / (M_prior*Gpercsq) * self.MUDISTS
            self.rhovec = rho#.flatten()

            phivec = arctan2(sin(self.varphivec),cos(self.varphivec)*cos(inc_prior))
            sinprod = sin(inc_prior)*sin(phivec)
            numerator = 1.+rho**2 - (-3.+rho**2.)*sinprod+3.*sinprod**2. + sinprod**3. 
            denomenator = (-1.+sinprod)**2 * (1+sinprod)
            sqq = sqrt(numerator/denomenator)
            rvec = (1.-sqq + sinprod*(1.+sqq))/(sinprod-1.)
             

            # rvec, phivec = emission_coordinates(self.rhovec, self.varphivec)
            # rvec = clip(rvec, 2.+1.e-5,np.inf)
            rvec = ((rvec-2)+sqrt((rvec-2)**2))/2.+2.0001
            # rvec = maximum(rvec, 2.00001)
            self.rvec = rvec
            self.phivec = phivec
            #begin Ramesh formalism
            eta = chi_prior+np.pi
            beq = sin(thetabz_prior)
            bz = cos(thetabz_prior)
            br = beq*cos(eta)
            bphi = beq*sin(eta)
            coschi = cos(chi_prior)
            sinchi = sin(chi_prior)
            betax = beta_prior*coschi
            betay = beta_prior*sinchi
            bx = br
            by = bphi

            bmag = sqrt(bx**2 + by**2 + bz**2)
            gfac = sqrt(1. - 2./rvec)
            gfacinv = 1. / gfac
            gamma = 1. / sqrt(1. - beta**2)

            sintheta = sin(inc_prior)
            # print(inc_prior)
            # print(sintheta)
            costheta = cos(inc_prior)
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

            profile = self.jfunc(rvec, phivec, jarg_priors)
            # profile=1.
            # 
            # print(self.profile)
            polarizedintensity = sinzeta**(1.+spec_prior) * delta**(3. + spec_prior) * profile
            
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
            radicand = (radicand + sqrt(radicand**2))/2
            # due to machine precision, some small negative values are present. We clip these here.
            # radicand[radicand<0] = 0
            radical = sqrt(radicand)
            # plt.imshow(radicand)
            # plt.show()
            kOlth = rvec * radical * sinphi / (sqrt(sinphi**2)+1.e-10)

            xalpha = -kOlp / sintheta
            ybeta = kOlth
            nu = -xalpha
            den = sqrt((k1**2 + k2**2) * (ybeta**2 + nu**2))

            ealpha = (ybeta * k2 - nu * k1) / den
            ebeta = (ybeta * k1 + nu * k2) / den

            qvec = -mag*(ealpha**2 - ebeta**2)
            uvec = -mag*(2*ealpha*ebeta)
            ivec = sqrt(qvec**2 + uvec**2)

            tf = summate(ivec)
            self.qvec = qvec * zbl_prior/tf
            self.uvec = uvec * zbl_prior/tf
            self.ivec = ivec * zbl_prior/tf
            #now do the rotation

            self.rotimxvec = cos(PA_prior)*self.imxvec - sin(PA_prior)*self.imyvec
            self.rotimyvec = sin(PA_prior)*self.imxvec + cos(PA_prior)*self.imyvec
            
            print("Finished building Bam! in "+ self.mode +" mode with calctype " +self.calctype)  
        elif calctype == 'exact':
            pass

    def vis(self,u,v):
        if self.mode == 'model':
            print("Direct visibility computation is not possible in modeling mode!")
            return
        u = np.array(u)
        v = np.array(v)


        A_reals = []
        A_imags = []
        # for n in range(self.nmax+1):
            # if self.calctype == 'approx':
            #     matrix = np.outer(u, -(-1.)**n*(self.r_g/self.D_prior)*(self.x_c+self.delta_xvec*np.exp(-self.deltawas[n]))) + np.outer(v, (-1.)**n *(self.r_g/self.D_prior)* (self.y_c+self.delta_yvec*np.exp(-self.deltawas[n])))
            # elif self.calctype == 'exact':
            #     matrix = np.outer(u, (self.r_g/self.D_prior)*self.imxvecs[n]) + np.outer(v, (self.r_g/self.D_prior)*self.imyvecs[n])
            # A_reals.append(self.gains[n]*self.pathlengths[n]*self.boosts[n]*np.cos(2.0*np.pi*matrix))
            # A_imags.append(self.gains[n]*self.pathlengths[n]*self.boosts[n]*np.sin(2.0*np.pi*matrix))
        matrix = np.outer(u, self.rotimxvec)+np.outer(v, self.rotimyvec)
        # A_real = np.sum(A_reals, axis=0)
        # A_imag = np.sum(A_imags, axis=0)
        A_real = np.cos(2.0*np.pi*matrix)
        A_imag = np.sin(2.0*np.pi*matrix)
        visreal_model = np.dot(A_real,self.ivec)
        visimag_model = np.dot(A_imag,self.ivec)

        return visreal_model + 1j* visimag_model

    def cphase(self, u1, u2, u3, v1, v2, v3):
        if self.mode == 'model':
            print("Direct cphase computaiton is not possible in modeling mode!")
            return

        vis12 = self.vis(u1, v1)
        vis23 = self.vis(u2, v2)
        vis31 = self.vis(u3, v3)
        phase12 = np.angle(vis12)
        phase23 = np.angle(vis23)
        phase31 = np.angle(vis31)
        cphase_model = phase12+phase23+phase31
        return cphase_model

    def logcamp(self, u1, u2, u3, u4, v1, v2, v3, v4):
        if self.mode == 'model':
            print("Direct logcamp computation is not possibl in modeling mode!")
            return

        # print("Building direct image FT matrices.")
        vis12 = self.vis(u1,v1)
        vis34 = self.vis(u2,v2)
        vis23 = self.vis(u3,v3)
        vis14 = self.vis(u4,v4)
        amp12 = np.abs(vis12)
        amp34 = np.abs(vis34)
        amp23 = np.abs(vis23)
        amp14 = np.abs(vis14)
        logcamp_model = np.log(amp12)+np.log(amp34)-np.log(amp23)-np.log(amp14)
        return logcamp_model

    def logcamp_chisq(self,obs):
        if self.mode != 'fixed':
            print("Can only compute chisqs to fixed model!")
            return
        logcamp_data = obs.c_amplitudes(ctype='logcamp')
        sigmaca = logcamp_data['sigmaca']
        logcamp = logcamp_data['camp']
        model_logcamps = self.logcamp(logcamp_data['u1'],logcamp_data['u2'],logcamp_data['u3'],logcamp_data['u4'],logcamp_data['v1'],logcamp_data['v2'],logcamp_data['v3'],logcamp_data['v4'])
        logcamp_chisq = 1/len(sigmaca) * np.sum(np.abs((logcamp-model_logcamps)/sigmaca)**2)
        return logcamp_chisq

    def cphase_chisq(self,obs):
        if self.mode != 'fixed':
            print("Can only compute chisqs to fixed model!")
            return
        cphase_data = obs.c_phases(ang_unit='rad')
        cphase = cphase_data['cphase']
        sigmacp = cphase_data['sigmacp']
        model_cphases = self.cphase(cphase_data['u1'],cphase_data['u2'],cphase_data['u3'],cphase_data['v1'],cphase_data['v2'],cphase_data['v3'])
        cphase_chisq = (2.0/len(sigmacp)) * np.sum((1.0 - np.cos(cphase-model_cphases))/(sigmacp**2))
        return cphase_chisq

    def vis_chisq(self,obs):
        if self.mode !='fixed':
            print("Can only compute chisqs to fixed model!")
            return

        u = obs.data['u']
        v = obs.data['v']
        sigma = obs.data['sigma']  
        amp = obs.unpack('amp')['amp']
        vis = obs.data['vis']
        sd = np.sqrt(sigma**2.0 + (self.sys_err*amp)**2.0 + self.abs_err**2.0)

        model_vis = self.vis(u,v)
        absdelta = np.abs(model_vis-vis)
        vis_chisq = np.sum((absdelta/sd)**2)/(2*len(vis))
        return vis_chisq

    def all_chisqs(self, obs):
        if self.mode !='fixed':
            print("Can only compute chisqs to fixed model!")
            return
        logcamp_chisq = self.logcamp_chisq(obs)
        cphase_chisq = self.cphase_chisq(obs)
        vis_chisq = self.vis_chisq(obs)
        return {'logcamp':logcamp_chisq,'cphase':cphase_chisq,'vis':vis_chisq}


    def get_model_vis(self, u, v):
        # print("Building u * x matrices.")
        # A_reals = []
        # A_imags = []
        # for n in range(self.nmax+1):

        #     matrix = tt.outer(u, -(-1.)**n*(self.r_g/self.D_prior)*(self.x_c+self.delta_xvec*pm.math.exp(-self.deltawas[n]))) + tt.outer(v, (-1.)**n *(self.r_g/self.D_prior)*(self.y_c+self.delta_yvec*pm.math.exp(-self.deltawas[n])))
        #     A_reals.append(self.gains[n]*self.pathlengths[n]*self.boosts[n]*pm.math.cos(2.0*np.pi*matrix))
        #     A_imags.append(self.gains[n]*self.pathlengths[n]*self.boosts[n]*pm.math.sin(2.0*np.pi*matrix))
        # matrices = [tt.outer(u, (-1.)**n*(self.x_c+self.delta_xvec*np.exp(-np.pi*n))) + tt.outer(v, (-1.)**n * (self.y_c+self.delta_yvec*np.exp(-np.pi*n))) for n in range(nmax+1)]
        # print("Building FT coeff matrices.")
        # A_reals = [ for n in range(nmax+1)]
        # A_imags = [ for n in range(nmax+1)]
        # print("Stacking.")
        # A_real = pm.math.sum(A_reals, axis=0)
        # A_imag = pm.math.sum(A_imags, axis=0)
        # print("Dotting.")
        # visreal_model = pm.math.dot(A_real,self.blixels) + 1.
        # visimag_model = pm.math.dot(A_imag,self.blixels)
        
        matrix = tt.outer(u, self.rotimxvec)+tt.outer(v, self.rotimyvec)
        # A_real = np.sum(A_reals, axis=0)
        # A_imag = np.sum(A_imags, axis=0)
        A_real = np.cos(2.0*np.pi*matrix)
        A_imag = np.sin(2.0*np.pi*matrix)
        visreal_model = pm.math.dot(A_real,self.ivec)
        visimag_model = pm.math.dot(A_imag,self.ivec)

        return visreal_model, visimag_model



    def build_vis_likelihood(self, obs):
        """
        Expects an ehtim obsdata object and a max number of subrings to fit.
        """
        # dataset = obs.unpack(['u','v','vis','sigma'])
        vis = obs.data['vis']
        sigma = obs.data['sigma']  
        amp = obs.unpack('amp')['amp']

        sd = pm.math.sqrt(sigma**2.0 + (self.sys_err*amp)**2.0+self.abs_err**2.0)

        u = obs.data['u']
        v = obs.data['v']
        u = theano.shared(u)
        v = theano.shared(v)
        
        if self.calctype == 'approx':        
            with self.model:
                # real, imag = self.vis(u, v, nmax=nmax)
                # print("Building FT matr"+str(n)+".")
                # total_flux, _ = self.get_model_vis(theano.shared(np.array([0])),theano.shared(np.array([0])))
                visreal_model, visimag_model = self.get_model_vis(u,v)
                real_likelihood = pm.Normal('vis_real', mu= visreal_model, sd=sd,observed=np.real(vis))
                imag_likelihood = pm.Normal('vis_imag', mu= visimag_model, sd=sd,observed=np.imag(vis))
                print("Constructed approx vis likelihoods.")
        elif self.calctype == 'exact':
            pass
            # self.loglgrad = LogLikeGrad(exact_vis_loglike, self.rvec, self.phivec, self.nmax, u, v, vis, sigma, sys_err = 0.02, abs_err=0.005)
            # self.loglwithgrad = LogLikeWithGrad(exact_vis_loglike, self.rvec, self.phivec, self.nmax, u, v, vis, sigma, sys_err = 0.02, abs_err=0.005)
            # with self.model:
            #     likelihood = pm.DensityDist("likelihood", lambda v: self.logl(v), observed={"v": self.theta})
            #     print("Constructed exact vis likelihoods")

    def build_amp_likelihood(self, obs):
        """
        Expects an ehtim obsdata object and a max number of subrings to fit.
        """
        # dataset = obs.unpack(['u','v','vis','sigma'])
        # vis = obs.data['vis']
        if self.calctype == 'exact':
            print("Amplitude fitting is not yet implemented for exact mode. Sorry!")
            return
        sigma = obs.data['sigma']  
        amp = obs.unpack('amp')['amp']

        sd = pm.math.sqrt(sigma**2.0 + (self.sys_err*amp)**2.0 + self.abs_err**2.0)

        u = obs.data['u']
        v = obs.data['v']
        u = theano.shared(u)
        v = theano.shared(v)
        
        
        with self.model:
            # real, imag = self.vis(u, v, nmax=nmax)
            # print("Building FT matr"+str(n)+".")
            total_flux, _ = self.get_model_vis(theano.shared(np.array([0])),theano.shared(np.array([0])))
            visreal_model, visimag_model = self.get_model_vis(u,v)
            visamp_model = self.tf / total_flux[0] * pm.math.sqrt(visreal_model**2 + visimag_model**2)
            # real_likelihood = pm.Normal('vis_real', mu=self.tf / total_flux[0] * visreal_model, sd=sd,observed=np.real(vis))
            # imag_likelihood = pm.Normal('vis_imag', mu=self.tf / total_flux[0] * visimag_model, sd=sd,observed=np.imag(vis))
            amp_likelihood = pm.Normal('amp', mu=visamp_model,sd=sd, observed=amp)
            print("Constructed vis likelihoods.")

    def build_logcamp_likelihood(self, obs):
        """
        Expects and ehtim obsdata object and a max number of subrings to fit.
        """
        if self.calctype == 'exact':
            print("logcamp fitting is not yet implemented for exact mode. Sorry!")
            return
        data = obs.c_amplitudes(ctype='logcamp')
        u1 = theano.shared(data['u1'])
        u2 = theano.shared(data['u2'])
        u3 = theano.shared(data['u3'])
        u4 = theano.shared(data['u4'])
        v1 = theano.shared(data['v1'])
        v2 = theano.shared(data['v2'])
        v3 = theano.shared(data['v3'])
        v4 = theano.shared(data['v4'])
        logcamp = data['camp']
        logcamp_sigma = data['sigmaca']

        with self.model:
            # print("Building direct image FT matrices.")
            visreal12, visimag12 = self.get_model_vis(u1,v1)
            visreal34, visimag34 = self.get_model_vis(u2,v2)
            visreal23, visimag23 = self.get_model_vis(u3,v3)
            visreal14, visimag14 = self.get_model_vis(u4,v4)
            amp12 = pm.math.sqrt(visreal12**2 + visimag12**2)
            amp34 = pm.math.sqrt(visreal34**2 + visimag34**2)
            amp23 = pm.math.sqrt(visreal23**2 + visimag23**2)
            amp14 = pm.math.sqrt(visreal14**2 + visimag14**2)

            logcamp_model = pm.math.log(amp12)+pm.math.log(amp34)-pm.math.log(amp23)-pm.math.log(amp14)

            logcamp_likelihood = pm.Normal('logcamp',mu=logcamp_model,sd=logcamp_sigma,observed=logcamp)
            print("Finished building logcamp likelihood.")

    def build_cphase_likelihood(self, obs):
        if self.calctype == 'exact':
            print("cphase fitting is not yet implemented for exact mode. Sorry!")
            return
        data = obs.c_phases(ang_unit='rad')
        u1 = theano.shared(data['u1'])
        u2 = theano.shared(data['u2'])
        u3 = theano.shared(data['u3'])
        v1 = theano.shared(data['v1'])
        v2 = theano.shared(data['v2'])
        v3 = theano.shared(data['v3'])
        cphase = data['cphase']
        cphase_sigma = data['sigmacp']

        with self.model:
            visreal12, visimag12 = self.get_model_vis(u1, v1)
            visreal23, visimag23 = self.get_model_vis(u2, v2)
            visreal31, visimag31 = self.get_model_vis(u3, v3)
            phase12 = tt.basic.arctan2(visimag12, visreal12)
            phase23 = tt.basic.arctan2(visimag23, visreal23)
            phase31 = tt.basic.arctan2(visimag31, visreal31)
            cphase_model = phase12+phase23+phase31

            cphase_likelihood = pm.VonMises('cphase', mu = cphase_model, kappa = 1/cphase_sigma**2,observed=cphase)
            print("Finished building cphase likelihood.")



    def find_MAP(self):
        """
        Finds the MAP once the MCmodel is built.
        """
        self.MAP_estimate = pm.find_MAP(model=self.model)
        print("Finished finding MAP.")
        return self.MAP_estimate


    def sample(self, init='auto', draws=1000, start=None, tune=1000, step=None, chains=1, discard_tuned_samples=False):#, draws=5000, step=None, init='auto', start=None, trace=None, chains=None, cores=None, tune=1000):
        """
        Thin wrapper around the pymc3 sample method. 
        """
        with self.model:
            if step == None:
                step = pm.NUTS()
            self.trace = pm.sample(init=init, draws=draws, start=start, tune=tune, step=step, chains=chains, discard_tuned_samples=discard_tuned_samples)
            # self.trace = pm.sample(500, return_inferencedata=False, chains=1)
            # self.trace = pm.sample(draws=draws,step=step, init=init, start=start,trace=trace,chains=chains,cores=cores,tune=tune)
        print("Finished generating trace.")




    def save_traceplot(self, name ='traceplot.png'):
        traceplot = pm.plots.traceplot(self.trace)
        plt.savefig(name,bbox_inches='tight')
        plt.close()

    def save_energyplot(self, name='energyplot.png'):
        modelinfo = {'trace':self.trace}
        eplot = dmc.plotting.plot_energy(modelinfo)
        plt.savefig(name,bbox_inches='tight')
        plt.close()

    def advanced_sampling(self,draws=1000, tune=1000,chains=1, compute_convergence_checks=False, discard_tuned_samples=False,regularize=True,diag=False,adapt_step_size=True,max_treedepth=10,early_max_treedepth=10,num_resamples=10,fov=100*eh.RADPERUAS,npix=200,kernel=1, out_dir='./'):
        try:
            os.stat(out_dir)
        except:
            os.mkdir(out_dir)
        for i in range(1,num_resamples):

            with self.model:
                step = dmc.model_utils.get_step_for_trace(self.trace,regularize=regularize,diag=diag,adapt_step_size=adapt_step_size,max_treedepth=max_treedepth,early_max_treedepth=early_max_treedepth)
                starting_values = [t[-1] for t in self.trace._straces.values()]
                self.trace = pm.sample(draws=draws, tune=tune, start=starting_values, step=step, chains=chains, compute_convergence_checks=compute_convergence_checks, discard_tuned_samples=discard_tuned_samples)

            self.save_traceplot(name = out_dir+'traceplot'+str(i).zfill(2)+'.png')
            self.save_energyplot(name=out_dir+'energyplot'+str(i).zfill(2)+'.png')
            # blim = self.random_sample_blimage()
            # blim.save_blimage(out_dir+'random_blimage'+str(i).zfill(2)+'.pkl')
            # im = blim.make_image(fov, npix,kernel)
            # im.save_fits(out_dir+'random_image'+str(i).zfill(2)+'.fits')
            # im.display(show=False, export_pdf=out_dir+'random_image'+str(i).zfill(2)+'.png')
            self.save_trace(directory=out_dir+'trace/', overwrite=True)



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
        #         jarg_priors.append(pm.Uniform(jarg_names[jargi],lower=jarg[0],upper=jarg[1]))
        #     else:
        #         jarg_priors.append(jarg)
       


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
