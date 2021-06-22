"""
Implementation of the Kerr toy model for use in interpolative BAM.
"""


from functools import wraps
import numpy as np
import scipy.optimize as op
import mpmath as mp
import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline, interp1d, interp2d
from scipy.integrate import quad_vec
import scipy.interpolate as si

#define elliptic functions (need the mpmath version to take complex args)
sn = np.frompyfunc(mp.ellipfun, 3, 1)
ef = np.frompyfunc(mp.ellipf, 2, 1)



#given rho, varphi, n, inc, spin, get lambda and eta



def get_lam_eta(rho, varphi, inc, a):
    """
    Analytic: get lambda and eta from screen coords, inc, and spin
    """
    alpha = rho*np.cos(varphi)
    beta = rho*np.sin(varphi)
    lam = -alpha*np.sin(inc)
    eta = (alpha**2-a**2)*np.cos(inc)**2 + beta**2
    return lam, eta

def get_up_um(lam, eta, a):
    """
    Analytic: get u_plus and u_minus from lambda, eta, and spin
    """
    del_theta = 1/2*(1-(eta+lam**2)/a**2)
    up = del_theta + np.sqrt(del_theta**2 + eta/a**2)
    um = del_theta - np.sqrt(del_theta**2 + eta/a**2)
    return up, um









