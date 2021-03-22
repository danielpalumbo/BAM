import numpy as np
from dmc3d.inference.model_helpers import rho_conv, varphi_conv, Gpercsq
# from dmc3d.inference.model_helpers import isiterable, generate_prediction
# from dmc3d.inference.model_definitions import MODEL_FUNC_DICT, MODEL_IMAGE_DICT, MODEL_PARAMS_DICT
# from dmc3d.visualization.model_plots import plot_frames, plot_custom_singlevar_posterior




class Blixel:
    '''
    The Blixel is a model component that knows its r and phi coordinates, as well as either a fixed j or prior.
    It also knows its M, D, or inc
    ''' 
    def __init__(self, r, phi, j, M, D, inc, mode):

        self.r = r
        self.phi = phi
        self.x = r*np.cos(phi)
        self.y = r*np.sin(phi)
        self.j = j
        self.M = M
        self.D = D
        self.rho_c = np.sqrt(27) * M/D*Gpercsq
        self.inc = inc
        self.rho = rho_conv(self.r, self.phi, self.inc) * M/D*Gpercsq
        self.mode = mode
        self.varphi = varphi_conv(self.phi, self.inc, mode)
        self.delta_rho = self.rho-self.rho_c


    def vis(self, u, v, nmax=1):
        """
        Given u and v coordinates, return the complex visibilities at those points for all nmax desired.
        """
        normu = np.sqrt(u**2+v**2)
        varphiu = np.arctan2(v,u)
        real_subs = [self.j*np.exp(-np.pi*n)*np.cos(-2.*np.pi*normu*(self.rho_c+self.delta_rho*np.exp(-np.pi*n))*np.cos(varphiu - self.varphi - n*np.pi)) for n in range(nmax+1)]
        imag_subs = [self.j*np.exp(-np.pi*n)*np.sin(-2.*np.pi*normu*(self.rho_c+self.delta_rho*np.exp(-np.pi*n))*np.cos(varphiu - self.varphi - n*np.pi)) for n in range(nmax+1)]
        _r = np.sum(real_subs, axis=0)
        _i = np.sum(imag_subs, axis=0)
        # subs = [self.j*np.exp(-np.pi*n)*np.exp(-2*np.pi*1j*normu*(self.rho_c+self.delta_rho*np.exp(-np.pi*n))*np.cos(varphiu - self.varphi - n*np.pi)) for n in range(nmax+1)]
        # _v = np.sum(subs, axis=0)
        return _r, _i


    def evalj(self, r, phi, kernel):
        """
        The blixel is infinitesimal. To evaluate it at arbitrary r and phi, it must
        be convolved with a small kernel in the midplane, assumed
        to be a circular Gaussian with sigma specified in M by kernel.
        """
        x = r*np.cos(phi)
        y = r*np.sin(phi)
        return self.j /(kernel*2*np.pi)* np.exp(-(self.x-x)**2 / (2*kernel**2) - (self.y-y)**2 / (2*kernel**2))

