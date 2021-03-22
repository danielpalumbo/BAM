import theano.tensor as tt

import cython

import numpy as np

import warnings

from dmc3d.inference.schwarzschildexact import getscreencoords , getpsin, getalphan, getwindangle
from dmc3d.inference.model_helpers import emission_coordinates, Gpercsq, M87_ra, M87_dec, M87_mass, M87_dist, M87_inc, isiterable

#r_lims, phi_lims, nr, nphi, M, D, inc, j, zbl, PA=0,  nmax=0, beta=0, chi=0, spec=1, f=0
def unscaled_model_vis(theta, rvec, phivec, nmax, u, v, vec=True,nmin=0):
    """
    Returns the model visibilities at u, v given parameters theta.
    """
    nblix = len(rvec)
    j = theta[-nblix:]
    M, D, inc, zbl, PA, beta, chi, spec = theta[:-nblix]

    rg = Gpercsq * M / D

    rhovecs = []
    varphivecs = []
    psivecs = []
    alphavecs = []
    wavecs = []
    for n in range(nmin, nmax+1):
        # subpsivec = []
        # subalphavec = []
        # subwavec = []
        if vec:
            b, varphi = getscreencoords(rvec, phivec, inc, n)
        
            subrhovec = b
            subvarphivec = varphi
        else:
            subrhovec=[]
            subvarphivec=[]
            for i in range(len(rvec)):
                b, varphi = getscreencoords(rvec[i], phivec[i], inc, n)
                # psin = getpsin(b, inc, phivec[i], n)
                # alphan = getalphan(b, rvec[i], inc, psin)
                # wa = getwindangle(b,rvec[i],phivec[i], inc, n)
                subrhovec.append(b)
                subvarphivec.append(varphi)
                # subpsivec.append(psin)
                # subalphavec.append(alphan)
                # subwavec.append(wa)
            subrhovec = np.array(subrhovec)
            subvarphivec = np.array(subvarphivec)
            

        
        # subpsivec = np.array(subpsivec)
        # subalphavec = np.array(subalphavec)
        # subwavec = np.array(subwavec)
        subpsivec = getpsin(inc, phivec, n)
        subalphavec = getalphan(subrhovec, rvec, inc, subpsivec)
        subwavec = subpsivec - subalphavec

        rhovecs.append(subrhovec)
        varphivecs.append(subvarphivec)
        psivecs.append(subpsivec)
        alphavecs.append(subalphavec)
        wavecs.append(subwavec)

    cosxis = [np.cos(inc)/np.sin(psivecs[n]) for n in range(nmax+1)]
    sinxis = [np.sin(inc)*np.cos(phivec)/np.sin(psivecs[n]) for n in range(nmax+1)]
    ktp = 1/(1-2/rvec)**(1/2)
    kxps = [np.cos(alphavecs[n])/(1-2/rvec)**(1/2) for n in range(nmax+1)]
    kyps = [-sinxis[n]*np.sin(alphavecs[n]/(1-2/rvec)**(1/2)) for n in range(nmax+1)]
    kzps = [cosxis[n]*np.sin(alphavecs[n]/(1-2/rvec)**(1/2)) for n in range(nmax+1)]

    betax = beta * np.cos(chi)
    betay = beta * np.sin(chi)
    gamma = 1/np.sqrt(1-beta**2)

    ktfs = [gamma*ktp - gamma*betax*kxps[n]-gamma*betay*kyps[n] for n  in range(nmax+1)] 
    kxfs = [-gamma*betax*ktp +(1+(gamma-1)*np.cos(chi)**2)*kxps[n] +(gamma-1)*np.cos(chi)*np.sin(chi)*kyps[n] for n in range(nmax+1)]
    kyfs = [-gamma*betay*ktp +(1+(gamma-1)*np.sin(chi)**2)*kyps[n] +(gamma-1)*np.cos(chi)*np.sin(chi)*kxps[n] for n in range(nmax+1)]
    kzfs = [kzps[n] for n in range(nmax+1)]
    dopplers = [1/ktfs[n] for n in range(nmax+1)]
    boosts = [dopplers[n]**(3+spec) for n in range(nmax+1)]
    pathlengths = [np.sqrt((ktfs[n]/kzfs[n])**2) for n in range(nmax+1)]


    pre_imxvecs = [rg*rhovecs[n]*np.cos(varphivecs[n]) for n in range(nmax+1)]
    pre_imyvecs = [rg*rhovecs[n]*np.sin(varphivecs[n]) for n in range(nmax+1)]
    imxvecs = [np.cos(PA)*pre_imxvecs[n] - np.sin(PA)*pre_imyvecs[n] for n in range(nmax+1)]
    imyvecs = [np.sin(PA)*pre_imxvecs[n] + np.cos(PA)*pre_imyvecs[n] for n in range(nmax+1)]

    deltawas = [np.sqrt(wavecs[n]**2) - np.sqrt(wavecs[0]**2) for n in range(nmax+1)]
    gains = [np.sqrt(rhovecs[n]**2/rhovecs[0]**2)*np.exp(-deltawas[n]) for n in range(nmax+1)]

    u = np.array(u)
    v = np.array(v)

    A_reals = []
    A_imags = []
    for n in range(nmin,nmax+1):
        matrix = np.outer(u, imxvecs[n]) + np.outer(v, imyvecs[n])
        A_reals.append(gains[n]*pathlengths[n]*boosts[n]*np.cos(2.0*np.pi*matrix))
        A_imags.append(gains[n]*pathlengths[n]*boosts[n]*np.sin(2.0*np.pi*matrix))
    
    A_real = np.sum(A_reals, axis=0)
    A_imag = np.sum(A_imags, axis=0)
    visreal_model = np.dot(A_real,j)
    visimag_model = np.dot(A_imag,j)

    return visreal_model + 1j*visimag_model + 1

def scaled_model_vis(theta,rvec, phivec, nmax, u, v, vec=True,nmin=0):
    nblix = len(rvec)
    j = theta[-nblix:]
    M, D, inc, zbl, PA, beta, chi, spec = theta[:-nblix]

    # M, D, inc, zbl, PA, beta, chi, spec = theta
    unscaled = unscaled_model_vis(theta,rvec, phivec, nmax,u,v,vec=vec,nmin=nmin)
    total_flux = unscaled_model_vis(theta,rvec, phivec, nmax,0,0,vec=vec,nmin=nmin)
    return zbl / total_flux * unscaled


def exact_vis_loglike(theta, rvec, phivec, nmax, u, v, obs_vis, obs_sigma, sys_err=0.02, abs_err=0.005):
    model_vis = scaled_model_vis(theta, rvec, phivec, nmax, u, v)
    obs_amp = np.abs(obs_vis)
    sd = np.sqrt(obs_sigma**2.0 + (sys_err*obs_amp)**2.0+abs_err**2.0)
    term = np.abs(model_vis-obs_vis)**2. / sd**2
    return -0.5 / len(obs_amp) *np.sum(term)



# class LogLike(tt.Op):

#     itypes = [tt.dvector]
#     otypes = [tt.dscalar]

#     def __init__(self, loglike, u, v, obs_vis, obs_sigma, sys_err = 0.02, abs_err=0.005):
#         self.likelihood = loglike
#         self.u = u
#         self.v = v
#         self.obs_vis = obs_vis
#         self.obs_sigma = obs_sigma
#         self.sys_err = sys_err
#         self.abs_err = abs_err

#     def perform(self, node, inputs, outputs):
#         (theta,) = inputs
#         logl = self.likelihood(theta, self.u, self.v, self.obs_vis, self.obs_sigma, sys_err = self.sys_err, abs_err = self.abs_err)

#         outputs[0][0] = np.array(logl)








def gradients(vals, func, releps=1e-3, abseps=None, mineps=1e-9, reltol=1e-3,
              epsscale=0.5):
    """
    Calculate the partial derivatives of a function at a set of values. The
    derivatives are calculated using the central difference, using an iterative
    method to check that the values converge as step size decreases.

    Parameters
    ----------
    vals: array_like
        A set of values, that are passed to a function, at which to calculate
        the gradient of that function
    func:
        A function that takes in an array of values.
    releps: float, array_like, 1e-3
        The initial relative step size for calculating the derivative.
    abseps: float, array_like, None
        The initial absolute step size for calculating the derivative.
        This overrides `releps` if set.
        `releps` is set then that is used.
    mineps: float, 1e-9
        The minimum relative step size at which to stop iterations if no
        convergence is achieved.
    epsscale: float, 0.5
        The factor by which releps if scaled in each iteration.

    Returns
    -------
    grads: array_like
        An array of gradients for each non-fixed value.
    """
    vals = np.array([float(val) for val in vals])
    # print("vals")
    # print(vals)
    grads = np.zeros(len(vals))
    # print('func')
    # print(func)
    # maximum number of times the gradient can change sign
    flipflopmax = 10.

    # set steps
    if abseps is None:
        if isinstance(releps, float):
            eps = np.abs(vals)*releps
            eps[eps == 0.] = releps  # if any values are zero set eps to releps
            teps = releps*np.ones(len(vals))
        elif isinstance(releps, (list, np.ndarray)):
            if len(releps) != len(vals):
                raise ValueError("Problem with input relative step sizes")
            eps = np.multiply(np.abs(vals), releps)
            eps[eps == 0.] = np.array(releps)[eps == 0.]
            teps = releps
        else:
            raise RuntimeError("Relative step sizes are not a recognised type!")
    else:
        if isinstance(abseps, float):
            eps = abseps*np.ones(len(vals))
        elif isinstance(abseps, (list, np.ndarray)):
            if len(abseps) != len(vals):
                raise ValueError("Problem with input absolute step sizes")
            eps = np.array(abseps)
        else:
            raise RuntimeError("Absolute step sizes are not a recognised type!")
        teps = eps

    # for each value in vals calculate the gradient
    count = 0
    for i in range(len(vals)):
        # initial parameter diffs
        leps = eps[i]
        cureps = teps[i]
        flipflop = 0

        # get central finite difference
        fvals = np.copy(vals)
        bvals = np.copy(vals)

        # central difference
        fvals[i] += 0.5*leps  # change forwards distance to half eps
        bvals[i] -= 0.5*leps  # change backwards distance to half eps
        cdiff = (func(fvals)-func(bvals))/leps
        # print("cdiff")
        # print(cdiff)
        while 1:
            fvals[i] -= 0.5*leps  # remove old step
            bvals[i] += 0.5*leps
            # change the difference by a factor of two
            cureps *= epsscale
            if cureps < mineps or flipflop > flipflopmax:
                # if no convergence set flat derivative (TODO: check if there is a better thing to do instead)
                warnings.warn("Derivative calculation did not converge: setting flat derivative.")
                grads[count] = 0.
                break
            leps *= epsscale

            # central difference
            fvals[i] += 0.5*leps  # change forwards distance to half eps
            bvals[i] -= 0.5*leps  # change backwards distance to half eps
            cdiffnew = (func(fvals)-func(bvals))/leps

            if cdiffnew == cdiff:
                grads[count] = cdiff
                break

            # check whether previous diff and current diff are the same within reltol
            rat = (cdiff/cdiffnew)
            # print(rat)
            if np.isfinite(rat) and rat > 0.:
                # gradient has not changed sign
                if np.abs(1.-rat) < reltol:
                    grads[count] = cdiffnew
                    break
                else:
                    cdiff = cdiffnew
                    continue
            else:
                cdiff = cdiffnew
                flipflop += 1
                continue

        count += 1

    return grads

class LogLikeWithGrad(tt.Op):

    itypes = [tt.dvector]
    otypes = [tt.dscalar]

    def __init__(self, loglike, rvec, phivec, nmax, u, v, obs_vis, obs_sigma,  sys_err = 0.02, abs_err=0.005):
        self.likelihood = loglike
        self.rvec = rvec
        self.phivec = phivec
        self.nmax = nmax
        self.u = u
        self.v = v
        self.obs_vis = obs_vis
        self.obs_sigma = obs_sigma
        self.sys_err = sys_err
        self.abs_err = abs_err

        self.logpgrad = LogLikeGrad(self.likelihood, self.rvec, self.phivec, self.nmax, self.u, self.v, self.obs_vis, self.obs_sigma, sys_err = self.sys_err, abs_err = self.abs_err)

    def perform(self, node, inputs, outputs):
        (theta,) = inputs
        logl = self.likelihood(theta, self.rvec, self.phivec, self.nmax, self.u, self.v, self.obs_vis, self.obs_sigma, sys_err = self.sys_err, abs_err = self.abs_err)

        outputs[0][0] = np.array(logl)

    def grad(self, inputs, g):
        (theta,) = inputs
        return [g[0] * self.logpgrad(theta)]

class LogLikeGrad(tt.Op):
    itypes = [tt.dvector]
    otypes = [tt.dvector]

    def __init__(self, loglike, rvec, phivec, nmax, u, v, obs_vis, obs_sigma, sys_err = 0.02, abs_err=0.005):
        self.likelihood = loglike
        self.rvec = rvec
        self.phivec = phivec
        self.nmax = nmax
        self.u = u
        self.v = v
        self.obs_vis = obs_vis
        self.obs_sigma = obs_sigma
        self.sys_err = sys_err
        self.abs_err = abs_err

    def perform(self, node, inputs, outputs):
        (theta,) = inputs

        def lnlike(values):
            return self.likelihood(values, self.rvec, self.phivec, self.nmax, self.u, self.v, self.obs_vis, self.obs_sigma, sys_err = self.sys_err, abs_err = self.abs_err)

        grads = gradients(theta, lnlike)

        outputs[0][0] = grads

