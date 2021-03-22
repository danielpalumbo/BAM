import pymc3 as pm 
import numpy as np
import ehtim as eh
import matplotlib.pyplot as plt
import theano
import random
import theano.tensor as tt
import pickle as pkl
from dmc3d.inference.model_helpers import emission_coordinates, Gpercsq, M87_ra, M87_dec, M87_mass, M87_dist, M87_inc, isiterable
import eht_dmc as dmc

    
class Blimage:
    '''The Blimage class is a collection of blixels and black hole parameters.
    '''
    #class contains knowledge of a grid in Boyer-Lindquist coordinates, priors on each pixel, and the machinery to fit them
    def __init__(self, r_lims, phi_lims, nr, nphi, M, D, inc, j, zbl, PA=0,  nmax=0, beta=0, chi=0, spec=1, f=0):
        self.MAP_estimate = None
        # self.MAP_values = None
        self.trace = None
        self.r_lims = r_lims
        self.phi_lims = phi_lims
        self.nr = nr
        self.nphi = nphi
        self.M = M
        self.D = D
        self.inc = inc
        self.j = j
        self.PA = PA
        self.beta = beta
        self.chi=chi
        self.spec=spec
        self.f = f
        self.nmax = nmax
        self.zbl = zbl

        self.nblix = nr*nphi
        r = np.sqrt(np.linspace(r_lims[0]**2, r_lims[1]**2, nr+1)[:-1])
        phi = np.linspace(phi_lims[0],phi_lims[1], nphi+1)[:-1]
        self.r = r
        self.phi = phi
        r_grid, phi_grid = np.meshgrid(r,phi)
        self.rvec = r_grid.flatten()
        self.phivec = phi_grid.flatten()


        self.xvec = self.rvec*np.cos(self.phivec)
        self.yvec = self.rvec*np.sin(self.phivec)
        
        if any([isiterable(i) for i in [M, D, inc, j, zbl, PA, f, beta, chi]]):
            mode = 'model'
        else:
            mode = 'fixed' 
        self.mode = mode
        # self.x = r*np.cos(phi)
        # self.y = r*np.sin(phi)


        self.highnalpha = np.arcsin(np.sqrt(1-2/self.rvec)*np.sqrt(27)/self.rvec)

        # self.blixels = list()

        if self.mode == 'model':
            cos = pm.math.cos
            sin = pm.math.sin
            arccos = tt.basic.arccos
            arcsin = tt.basic.arcsin
            arctan2 = tt.basic.arctan2
            sqrt = pm.math.sqrt
            model = pm.Model()
            with model:
                #build priors
                if isiterable(M):
                    M_prior = pm.Uniform('M',lower=M[0],upper=M[1])
                else:
                    M_prior = M
                # self.
                if isiterable(D):
                    D_prior = pm.Uniform('D',lower=D[0],upper=D[1])
                else:
                    D_prior = D

                if isiterable(inc):
                    inc_prior = pm.Uniform('inc',lower=inc[0],upper=inc[1])
                else:
                    inc_prior = inc

                if isiterable(zbl):
                    zbl_prior = pm.Uniform('zbl',lower=zbl[0],upper=zbl[1])
                else:
                    zbl_prior = zbl

                if isiterable(PA):
                    mu = (PA[0]+PA[1])/2
                    diff = np.abs(PA[1]-PA[0])
                    kappa = 1/diff**2
                    # PA_prior = pm.Uniform('PA',lower=PA[0],upper=PA[1])
                    PA_prior = pm.VonMises('PA',mu=mu, kappa=kappa)
                else:
                    PA_prior = PA

                if isiterable(beta):
                    beta_prior = pm.Uniform('beta',lower=beta[0], upper=beta[1])
                else:
                    beta_prior = beta

                if isiterable(chi):
                    mu = (chi[0]+chi[1])/2
                    diff = np.abs(chi[1]-chi[0])
                    kappa = 1/diff**2
                    # PA_prior = pm.Uniform('PA',lower=PA[0],upper=PA[1])
                    chi_prior = pm.VonMises('chi',mu=mu, kappa=kappa)
                else:
                    chi_prior = chi

                if isiterable(spec):
                    spec_prior=pm.Uniform('spec',lower=spec[0],upper=spec[1])
                else:
                    spec_prior = spec


                if isiterable(f):
                    f_prior = pm.Uniform('f',lower=f[0],upper=f[1])
                else:
                    f_prior = f


                if isiterable(j):
                    self.blixels = pm.Uniform('blixels',lower=j[0],upper=j[1],shape=self.nblix)
                    # blixels = pm.Dirichlet('blixels', a=a, shape=self.npix)
                    # self.blixels=blixels
                else:
                    if callable(j):
                        self.blixels = j(self.rvec, self.phivec)
                    else:
                        self.blixels = j*np.ones_like(self.rvec)
            self.model=model
        else:
            cos = np.cos
            sin = np.sin
            arccos = np.arccos
            arcsin = np.arcsin
            arctan2 = np.arctan2
            sqrt = np.sqrt

            M_prior = M
            D_prior = D
            inc_prior = inc
            PA_prior = PA
            zbl_prior = zbl
            beta_prior = beta
            chi_prior = chi
            spec_prior = spec
            f_prior = f
            if callable(j):
                self.blixels=j(self.rvec, self.phivec)
            else:
                self.blixels = j*np.ones_like(self.rvec)


        def rho_conv(r, phi, D, theta0, r_g):
            rho2 = (((r/D)**2.0)*(1.0 - ((sin(theta0)**2.0)*(sin(phi)**2.0)))) + ((2.0*r*r_g/(D**2.0))*((1.0 + (sin(theta0)*sin(phi)))**2.0))
            rho = sqrt(rho2)
            return rho

        def varphi_conv(phi, theta0):
            return arctan2(np.sin(phi)*cos(theta0),cos(phi))

        self.cospsi0 = -sin(inc_prior)*np.sin(self.phivec)
        self.cosalpha0 = self.cospsi0 + 2/self.rvec *(1-self.cospsi0)

        # self.cosalpha0prime = -self.cosalpha0
        # self.cospsi0prime = - (self.cosalpha0+2/self.rvec)/(1-2/self.rvec)

        self.psi0 = arccos(self.cospsi0)
        self.psi1 = -(2*np.pi-self.psi0)

        self.alpha0 = arccos(self.cosalpha0)


        #build list of alphas
        self.alphas = []
        # self.alphaprimes = []
        for n in range(nmax+1):
            if n == 0:
                self.alphas.append(self.alpha0)
                # self.alphaprimes.append(np.pi-self.alpha0)
            else:
                alpha = np.arcsin((-1)**n * np.sqrt(1-2/self.rvec) * np.sqrt(27)/self.rvec)
                alpha[self.rvec > 3] =  (-1)**n * (np.pi - (-1)**n * alpha[self.rvec>3])
                self.alphas.append(alpha)
                # self.alphaprimes.append(np.pi-alpha)

        #build list of all relevant psis
        self.psis = []
        # self.psiprimes = []
        for n in range(nmax+1):
            if n == 0:
                self.psis.append(self.psi0)
            elif n == 1:
                self.psis.append(self.psi1)
            else:
                self.psis.append(self.psis[n-2]+2*np.pi*(-1)**n)
            # self.psiprimes.append((-1)**n * arccos(-(cos(self.alphas[n]) + 2./self.rvec)/(1.-2./self.rvec)))
            
        #finally, build winding angles
        #if using partial winding angles
        self.winding_angles = [self.psis[n]-self.alphas[n] for n in range(nmax+1)]
        # self.winding_angles = [self.psis[n]+self.psiprimes[n]-np.pi*(-1)**n for n in range(nmax+1)]
        self.deltawas = [sqrt(self.winding_angles[n]**2) - sqrt(self.winding_angles[0]**2) for n in range(nmax+1)]



        self.r_g = M_prior * Gpercsq
        self.rho_c = np.sqrt(27) * self.r_g / D_prior
        self.rhovec = rho_conv(self.rvec*self.r_g, self.phivec, D_prior, inc_prior, self.r_g) 
        self.delta_rhovec = self.rhovec - self.rho_c
        self.rhos = [self.rho_c+self.delta_rhovec*np.exp(-self.deltawas[n]) for n in range(nmax+1)]
        self.gains = [sqrt(self.rhos[n]**2/self.rhos[0]**2)*np.exp(-self.deltawas[n]) for n in range(nmax+1)]





        self.varphivec = varphi_conv(self.phivec, inc_prior)
        # self.varphivec = varphi_conv(self.phivec, inc_prior, mode)
        # self.imxvec = self.rhovec*pm.math.cos(self.varphivec)
        # self.imyvec = self.rhovec*pm.math.sin(self.varphivec)
        pre_imxvec = self.rhovec*cos(self.varphivec)
        pre_imyvec = self.rhovec*sin(self.varphivec)
        self.imxvec = cos(PA_prior)*pre_imxvec - sin(PA_prior)*pre_imyvec
        self.imyvec = sin(PA_prior)*pre_imxvec + cos(PA_prior)*pre_imyvec
        # self.effective_rhovec = sqrt(self.imxvec**2 + self.imyvec**2)
        # self.delta_rhovec = self.effective_rhovec-self.rho_c
        # self.effective_varphivec = self.varphivec - PA_prior
        pre_x_c = self.rho_c * cos(self.varphivec)
        pre_y_c = self.rho_c * sin(self.varphivec)
        self.x_c = cos(PA_prior)*pre_x_c - sin(PA_prior)*pre_y_c
        self.y_c = sin(PA_prior)*pre_x_c + cos(PA_prior)*pre_y_c
        # self.x_c = self.rho_c*pm.math.cos(self.effective_varphivec)
        # self.y_c = self.rho_c*pm.math.sin(self.effective_varphivec)
        self.delta_xvec = self.imxvec - self.x_c
        self.delta_yvec = self.imyvec - self.y_c

        self.sys_err = f_prior
        self.tf = zbl_prior

        #velocity and redshift effects from ring model paper
        self.cosxis = [cos(inc_prior)/sin(self.psis[n]) for n in range(nmax+1)]
        self.sinxis = [sin(inc_prior)*np.cos(self.phivec)/sin(self.psis[n]) for n in range(nmax+1)]
        self.ktp = 1/(1-2/self.rvec)**(1/2)
        self.kxps = [cos(self.alphas[n])/(1-2/self.rvec)**(1/2) for n in range(nmax+1)]
        self.kyps = [-self.sinxis[n]*sin(self.alphas[n]/(1-2/self.rvec)**(1/2)) for n in range(nmax+1)]
        self.kzps = [self.cosxis[n]*sin(self.alphas[n]/(1-2/self.rvec)**(1/2)) for n in range(nmax+1)]

        self.betax = beta_prior * cos(chi_prior)
        self.betay = beta_prior * sin(chi_prior)
        self.gamma = 1/sqrt(1-beta_prior**2)

        self.ktfs = [self.gamma*self.ktp - self.gamma*self.betax*self.kxps[n]-self.gamma*self.betay*self.kyps[n] for n  in range(nmax+1)] 
        self.kxfs = [-self.gamma*self.betax*self.ktp +(1+(self.gamma-1)*cos(chi_prior)**2)*self.kxps[n] +(self.gamma-1)*cos(chi_prior)*sin(chi_prior)*self.kyps[n] for n in range(nmax+1)]
        self.kyfs = [-self.gamma*self.betay*self.ktp +(1+(self.gamma-1)*sin(chi_prior)**2)*self.kyps[n] +(self.gamma-1)*cos(chi_prior)*sin(chi_prior)*self.kxps[n] for n in range(nmax+1)]
        self.kzfs = [self.kzps[n] for n in range(nmax+1)]
        self.dopplers = [1/self.ktfs[n] for n in range(nmax+1)]
        self.boosts = [self.dopplers[n]**(3+spec_prior) for n in range(nmax+1)]
        self.pathlengths = [sqrt((self.ktfs[n]/self.kzfs[n])**2) for n in range(nmax+1)]

        print("Finished building blimage in "+ self.mode +" mode.")            

    def unscaled_vis(self,u,v):
        if self.mode == 'model':
            print("Direct visibility computation is not possible in modeling mode!")
            return
        u = np.array(u)
        v = np.array(v)


        A_reals = []
        A_imags = []
        for n in range(self.nmax+1):

            matrix = np.outer(u, (-1.)**n*(self.x_c+self.delta_xvec*np.exp(-self.deltawas[n]))) + np.outer(v, (-1.)**n * (self.y_c+self.delta_yvec*np.exp(-self.deltawas[n])))
            A_reals.append(self.gains[n]*self.pathlengths[n]*self.boosts[n]*np.cos(2.0*np.pi*matrix))
            A_imags.append(self.gains[n]*self.pathlengths[n]*self.boosts[n]*np.sin(2.0*np.pi*matrix))
        
        A_real = np.sum(A_reals, axis=0)
        A_imag = np.sum(A_imags, axis=0)
        visreal_model = np.dot(A_real,self.blixels)
        visimag_model = np.dot(A_imag,self.blixels)

        return visreal_model + 1j* visimag_model + 1.

    def vis(self, u, v):
        if self.mode == 'model':
            print("Direct visibility computation is not possible in modeling mode!")
            return
        unscaled = self.unscaled_vis(u,v)
        total_flux = self.unscaled_vis(0,0)
        return self.tf / total_flux * unscaled

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


    def vis_loglike(self, u, v, obs_vis, obs_sigma):
        model_vis = self.vis(u,v)
        obs_amp = np.abs(obs_vis)
        sd = np.sqrt(obs_sigma**2.0 + (self.sys_err*obs_amp)**2.0)
        term = np.abs(model_vis-obs_vis)**2. / sd**2
        return -0.5 / len(obs_amp) *np.sum(term)

    def logcamp_loglike(self, u1, u2, u3, u4, v1, v2, v3, v4, obs_logcamp, obs_sigma):
        model_logcamp = self.logcamp(u1, u2, u3, u4, v1, v2, v3, v4)
        return 1/len(obs_sigma) * np.sum(np.abs((obs_logcamp-model_logcamp)/obs_sigma)**2)

    def cphase_loglike(self, u1, u2, u3, v1, v2, v3, obs_cphase, obs_sigma):
        model_cphase = self.cphase(u1,u2,u3,v1,v2,v3)
        return (2.0/len(obs_sigma)) * np.sum((1.0 - np.cos(obs_cphase-model_cphase))/(obs_sigma**2))


    

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
        sd = np.sqrt(sigma**2.0 + (self.sys_err*amp)**2.0)

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
        print("Building u * x matrices.")
        A_reals = []
        A_imags = []
        for n in range(self.nmax+1):

            matrix = tt.outer(u, (-1.)**n*(self.x_c+self.delta_xvec*np.exp(-self.deltawas[n]))) + tt.outer(v, (-1.)**n * (self.y_c+self.delta_yvec*np.exp(-self.deltawas[n])))
            A_reals.append(self.gains[n]*self.pathlengths[n]*self.boosts[n]*pm.math.cos(2.0*np.pi*matrix))
            A_imags.append(self.gains[n]*self.pathlengths[n]*self.boosts[n]*pm.math.sin(2.0*np.pi*matrix))
        # matrices = [tt.outer(u, (-1.)**n*(self.x_c+self.delta_xvec*np.exp(-np.pi*n))) + tt.outer(v, (-1.)**n * (self.y_c+self.delta_yvec*np.exp(-np.pi*n))) for n in range(nmax+1)]
        # print("Building FT coeff matrices.")
        # A_reals = [ for n in range(nmax+1)]
        # A_imags = [ for n in range(nmax+1)]
        print("Stacking.")
        A_real = pm.math.sum(A_reals, axis=0)
        A_imag = pm.math.sum(A_imags, axis=0)
        print("Dotting.")
        visreal_model = pm.math.dot(A_real,self.blixels) + 1.
        visimag_model = pm.math.dot(A_imag,self.blixels)
        return visreal_model, visimag_model



    def build_vis_likelihood(self, obs):
        """
        Expects an ehtim obsdata object and a max number of subrings to fit.
        """
        # dataset = obs.unpack(['u','v','vis','sigma'])
        vis = obs.data['vis']
        sigma = obs.data['sigma']  
        amp = obs.unpack('amp')['amp']

        sd = pm.math.sqrt(sigma**2.0 + (self.sys_err*amp)**2.0)

        u = obs.data['u']
        v = obs.data['v']
        u = theano.shared(u)
        v = theano.shared(v)
        
        
        with self.model:
            # real, imag = self.vis(u, v, nmax=nmax)
            # print("Building FT matr"+str(n)+".")
            total_flux, _ = self.get_model_vis(theano.shared(np.array([0])),theano.shared(np.array([0])))
            visreal_model, visimag_model = self.get_model_vis(u,v)
            real_likelihood = pm.Normal('vis_real', mu=self.tf / total_flux[0] * visreal_model, sd=sd,observed=np.real(vis))
            imag_likelihood = pm.Normal('vis_imag', mu=self.tf / total_flux[0] * visimag_model, sd=sd,observed=np.imag(vis))
            print("Constructed vis likelihoods.")

    def build_logcamp_likelihood(self, obs):
        """
        Expects and ehtim obsdata object and a max number of subrings to fit.
        """
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


    def sample(self, init='jitter+adapt_full', draws=1000, start=None, tune=1000, step=None, chains=1, discard_tuned_samples=False):#, draws=5000, step=None, init='auto', start=None, trace=None, chains=None, cores=None, tune=1000):
        """
        Thin wrapper around the pymc3 sample method. 
        """
        with self.model:
            if step == None:
                step = pm.NUTS()
            self.trace = pm.sample(init=init, draws=draws, start=start, tune=tune, step=step, chains=chains, discard_tuned_samples=discard_tuned_samples)

            # self.trace = pm.sample(draws=draws,step=step, init=init, start=start,trace=trace,chains=chains,cores=cores,tune=tune)
        print("Finished generating trace.")

    def plot_screen_n0(self, outname='n0im.png',save=True):
        """
        Plots the screen positions of the direct image of each blixel.
        """
        plt.plot(self.imxvec, self.imyvec,'.')
        plt.title('Direct Image Lensed Positions')
        if save:
            plt.savefig(outname,bbox_inches='tight')
        plt.show()


    def plot_screen_n(self,n=2,outname = 'singlenim.png',save=True):
        xs = (-1.)**n*(self.x_c+self.delta_xvec*np.exp(-self.deltawas[n]))
        ys = (-1.)**n * (self.y_c+self.delta_yvec*np.exp(-self.deltawas[n]))
        # xs = (-1.)**n*(self.x_c+self.delta_xvec*np.exp(-np.pi*n))
        # ys = (-1.)**n*(self.y_c+self.delta_yvec*np.exp(-np.pi*n))
        # print(ys)
        plt.plot(xs,ys,'.')#,label=str(n))
        # plt.legend()
        plt.title('n = '+str(n))
        plt.gca().set_aspect('equal')
        plt.gca().set_xticks(plt.yticks()[0])
        if save:
            plt.savefig(outname,bbox_inches='tight',dpi=300)
        plt.show()

    def plot_screen_n_upto(self,nmax=2,outname = 'multinim.png',save=True):
        xs = [(-1.)**n*(self.x_c+self.delta_xvec*np.exp(-self.deltawas[n]))/eh.RADPERUAS for n in range(nmax+1)]
        ys = [(-1.)**n * (self.y_c+self.delta_yvec*np.exp(-self.deltawas[n]))/eh.RADPERUAS for n in range(nmax+1)]
        # print(ys)
        for n in range(nmax+1):
            plt.plot(xs[n],ys[n],'.',label='n = '+str(n),markersize=10/(n+1))
        plt.legend()
        plt.gca().set_aspect('equal')
        plt.gca().set_xticks(plt.yticks()[0])
        plt.xlabel(r'$\alpha$ ($\mu$as)')
        plt.ylabel(r'$\beta$ ($\mu$as)')
        plt.title("Lensed subring positions")
        if save:
            plt.savefig(outname,bbox_inches='tight',dpi=300)
        plt.show()

    def plot_blixels(self,outname= 'blixels.png',save=True):
        """
        Plots the screen positions of the direct image of each blixel.
        """
        plt.plot(self.xvec, self.yvec,'.')
        plt.title('Blixel Positions')
        plt.xlabel('x (M)')
        plt.ylabel('y (M)')
        plt.gca().set_aspect('equal')
        yticks = plt.yticks()[0]

        plt.gca().set_xticks(yticks)
        if save:
            plt.savefig(outname,bbox_inches='tight',dpi=300)
        plt.show()


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
        for i in range(1,num_resamples):

            with self.model:
                step = dmc.model_utils.get_step_for_trace(self.trace,regularize=regularize,diag=diag,adapt_step_size=adapt_step_size,max_treedepth=max_treedepth,early_max_treedepth=early_max_treedepth)
                starting_values = [t[-1] for t in self.trace._straces.values()]
                self.trace = pm.sample(draws=draws, tune=tune, start=starting_values, step=step, chains=chains, compute_convergence_checks=compute_convergence_checks, discard_tuned_samples=discard_tuned_samples)

            self.save_traceplot(name = out_dir+'traceplot'+str(i).zfill(2)+'.png')
            self.save_energyplot(name=out_dir+'energyplot'+str(i).zfill(2)+'.png')
            blim = self.random_sample_blimage()
            blim.save_blimage(out_dir+'random_blimage'+str(i).zfill(2)+'.pkl')
            im = blim.make_image(fov, npix,kernel)
            im.save_fits(out_dir+'random_image'+str(i).zfill(2)+'.fits')
            im.display(show=False, export_pdf=out_dir+'random_image'+str(i).zfill(2)+'.png')
            self.save_trace(directory=out_dir+'trace/', overwrite=True)



    def make_image(self, fov, npix, kernel, ra=M87_ra, dec=M87_dec, rf= 230e9, mjd = 57854, source='M87'):
        """
        Returns an ehtim Image object corresponding to the Blimage n0 emission
        """

        if self.mode == 'model':
            print("Cannot directly make images in model mode! Call sample_blimage or MAP_blimage and display that!")
            return

        im = eh.image.make_empty(npix,fov, ra=ra, dec=dec, rf= rf, mjd = mjd, source='M87')

        pxi = (np.arange(npix)-0.01)/npix-0.5
        pxj = np.arange(npix)/npix-0.5
        # get angles measured East of North
        PXI,PXJ = np.meshgrid(pxi,pxj)
        varphi = np.arctan2(-PXJ,PXI)# - np.pi/2
        # varphi[varphi<0.] += 2.*np.pi

        #get grid of angular radii
        mui = pxi*fov
        muj = pxj*fov
        MUI,MUJ = np.meshgrid(mui,muj)
        MUDISTS = np.sqrt(np.power(MUI,2.)+np.power(MUJ,2.))
        rho = self.D / (Gpercsq * self.M) * MUDISTS
        # plt.imshow(rho)
        # plt.colorbar()
        # plt.show()
        # self.cospsi0 = -sin(inc_prior)*np.sin(self.phivec)
        # self.cosalpha0 = self.cospsi0 + 2/self.rvec *(1-self.cospsi0)

        # # self.cosalpha0prime = -self.cosalpha0
        # # self.cospsi0prime = - (self.cosalpha0+2/self.rvec)/(1-2/self.rvec)

        # self.psi0 = arccos(self.cospsi0)
        # self.psi1 = -(2*np.pi-self.psi0)

        # self.alpha0 = np.pi - arccos(self.cosalpha0)


        # #build list of alphas
        # self.alphas = []
        # self.alphaprimes = []
        # for n in range(nmax+1):
        #     if n == 0:
        #         self.alphas.append(self.alpha0)
        #         self.alphaprimes.append(np.pi-self.alpha0)
        #     else:
        #         alpha = np.arcsin((-1)**n * np.sqrt(1-2/self.rvec) * np.sqrt(27)/self.rvec)
        #         alpha[self.rvec > 3] =  (-1)**n * (np.pi - (-1)**n * alpha[self.rvec>3])
        #         self.alphas.append(alpha)
        #         self.alphaprimes.append(np.pi-alpha)

        # #build list of all relevant psis
        # self.psis = []
        # self.psiprimes = []
        # for n in range(nmax+1):
        #     if n == 0:
        #         self.psis.append(self.psi0)
        #     elif n == 1:
        #         self.psis.append(self.psi1)
        #     else:
        #         self.psis.append(self.psis[n-2]+2*np.pi*(-1)**n)
        #     self.psiprimes.append((-1)**n * arccos(-(cos(self.alphas[n]) + 2./self.rvec)/(1.-2./self.rvec)))
            
        # #finally, build winding angles
        # self.winding_angles = [self.psis[n]+self.psiprimes[n]-np.pi*(-1)**n for n in range(nmax+1)]
        # self.deltawas = [sqrt(self.winding_angles[n]**2) - sqrt(self.winding_angles[0]**2) for n in range(nmax+1)]
        
        for n in range(self.nmax+1):
            #convert mudists to gravitational units
            rho_sub = np.sqrt(27) + (rho-np.sqrt(27))*np.exp(np.pi*n)
            varphi_sub = varphi+np.pi*n
            r, phi = emission_coordinates(rho_sub, varphi_sub, self.inc)
            
            x = r*np.cos(phi)
            y = r*np.sin(phi)
            # plt.imshow(x)
            # plt.colorbar()
            # plt.show()
            # im.ivec += self.blixels /(kernel*2*np.pi)* np.exp(-(self.xvec-x)**2 / (2*kernel**2) - (self.yvec-y)**2 / (2*kernel**2))            

            addvec = np.zeros_like(x)
            for veci in range(self.nblix):
                blixvec = (self.gains[n][veci]*self.boosts[n][veci]*self.pathlengths[n][veci]* self.blixels[veci] /(kernel*2*np.pi)* np.exp(-(self.xvec[veci]-x)**2 / (2*kernel**2) - (self.yvec[veci]-y)**2 / (2*kernel**2)))
                blixvec[rho_sub<0] = 0
                addvec+=blixvec
            # plt.imshow(addvec)
            # plt.colorbar()
            # plt.show()
            # print(np.sum(addvec))
            if np.sum(addvec)>0:
                if n > 0:
                    addvec *= prior_flux / np.sum(addvec) * np.exp(-np.pi*n)
            prior_flux = np.sum(addvec)
            if n ==0:
                addvec += 1/(kernel*2*np.pi)* np.exp(-(x)**2 / (2*kernel**2) - (y)**2 / (2*kernel**2))
            im.ivec += addvec.flatten()
            # print(prior_flux)
        im.ivec *= self.tf / im.total_flux()
        return im.rotate(self.PA)

    def MAP_blimage(self):
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

        new_blimage = Blimage(self.r_lims, self.phi_lims, self.nr, self.nphi, M_new, D_new, inc_new, 0, zbl_new, PA=PA_new, beta=beta_new, chi=chi_new, spec=spec_new, nmax = self.nmax)
  
        
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

        #make a new blimage with zero emissivity but same dimensions
        new_blimage = Blimage(self.r_lims, self.phi_lims, self.nr, self.nphi, M_new, D_new, inc_new, 0, zbl_new, PA=PA_new, beta=beta_new, chi=chi_new, spec=spec_new, nmax = self.nmax)
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
