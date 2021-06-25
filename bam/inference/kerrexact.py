"""
Implementation of the Kerr toy model for use in interpolative BAM.
"""


import numpy as np
import mpmath as mp
import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline, interp1d, interp2d
# import scipy.interpolate as si

#define elliptic functions (need the mpmath version to take complex args)
sn = np.frompyfunc(mp.ellipfun, 3, 1)
ef = np.frompyfunc(mp.ellipf, 2, 1)



#given rho, varphi, n, inc, spin, get lambda and eta



def get_lam_eta(alpha, beta, inc, a):
    """
    Analytic: get lambda and eta from screen coords, inc, and spin
    """
    # alpha = rho*np.cos(varphi)
    # beta = rho*np.sin(varphi)
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

def get_radroots(lam, eta, a):
    """
    Analytic: get r1, r2, r3, and r4 for a given lambda and eta
    """
    A = a**2 - eta - lam**2
    B = 2*(eta+(lam-a)**2)
    C = -a**2 * eta
    P = - A**2 / 12 - C
    Q = -A/3 * ((A/6)**2 - C)-B**2/8
    H = -Q + np.sqrt(12*P**3 + 81*Q**2)
    z = np.sqrt((-2*(3**(1/3) * P) + 2**(1/3)*H**(2/3))/(2*6**(2/3)*H**(1/3))-A/6)
    # z = np.abs(z)
    r1 = -z - np.sqrt(-A/2 - z**2 + B/(4*z))
    r2 = -z + np.sqrt(-A/2 - z**2 + B/(4*z))
    r3 = z - np.sqrt(-A/2 - z**2 - B/(4*z))
    r4 = z + np.sqrt(-A/2 - z**2 - B/(4*z))
    return r1,r2,r3,r4

def get_radius_exact(rho, varphi, inc, a, n):
    """
    Numerical: get rs from rho, varphi, inc, a, and subimage index n.
    """
    alpha = rho*np.cos(varphi)
    beta = rho*np.sin(varphi)
    lam, eta = get_lam_eta(alpha,beta, inc, a)
    lam = np.complex128(lam)
    eta = np.complex128(eta)
    up, um = get_up_um(lam, eta, a)
    # up[up/um > 1] = np.nan
    # um[up/um > 1] = np.nan
    # plt.imshow(up/um)
    # plt.imshow(np.real(up/um))
    # plt.colorbar()
    # plt.show()
    r1, r2, r3, r4 = get_radroots(lam, eta, a)
    r31 = r3-r1
    r32 = r3-r2
    r42 = r4-r2
    r41 = r4-r1
    k = r32*r41 / (r31*r42)
    #build m array from beta sign and subimage index
    m = np.sign(beta)
    m[m<0] = 0
    m += n
    fobs = np.complex128(ef(np.arcsin(np.sqrt(r31/r41)), k))
    Fobs = np.complex128(ef(np.arcsin(np.cos(inc)/np.sqrt(up)), up/um))
    Ir = np.real(1/np.sqrt(-um*a**2)*(2*m*np.complex128(ef(np.pi/2, up/um)) - np.sign(beta)*Fobs))
    # plt.imshow(Ir)
    # plt.title('Ir')
    # plt.show()
    ffac = 1 / 2 * (r31 * r42)**(1/2)
    snsqr = np.complex128(sn('sn', ffac * Ir - fobs, k))**2
    rs = np.real((r4*r31 - r3*r41*snsqr) / (r31-r41*snsqr))
    return rs

def build_all_interpolators(rho_interp):
    nrho = len(rho_interp)
    varphi_interp = np.linspace(1e-5,2*np.pi,50)
    theta_interp = np.linspace(1e-3,np.pi/2-1e-3,50)
    a_interp = np.linspace(1e-3,1-1e-3,50)
    rr, vv, tt, aa = np.meshgrid(rho_interp, varphi_interp, theta_interp, a_interp)


    # rr, vv = np.meshgrid(rho_interp, varphi_interp)
    aa = rr*np.cos(vv)
    bb = rr*np.sin(vv)
    all_lam, all_eta = get_lam_eta(aa,bb,tt,aa)
    all_up, all_um = get_up_um(all_lam, all_eta, aa)
    all_r1, all_r2, all_r3, all_r4 = get_radroots(all_lam, all_eta, aa)

    all_r32 = all_r3-all_r2
    all_r41 = all_r4-all_r1
    all_r31 = all_r3-all_r1
    all_r42 = all_r4-all_r2
    all_k = all_r32*all_r41/all_r31/all_r42
    all_fobsarcsin = np.arcsin(np.sqrt(all_r31/all_r41))


    lam_interp = np.complex128(np.linspace(np.min(all_lam), np.max(all_lam), nrho))
    eta_interp = np.complex128(np.linspace(np.min(all_eta), np.max(all_eta), nrho))
    up_interp = np.linspace(np.min(all_up), np.max(all_up), nrho)
    um_interp = np.linspace(np.min(all_um), np.max(all_um), nrho)
    k_interp = np.linspace(np.min(all_k), np.max(all_k), nrho)
    fobsarcsin_interp = np.linspace(np.min(all_fobsarcsin), np.max(all_fobsarcsin))
    um_interp-=1e-15
    # return um_interp
    ll, ee = np.meshgrid(lam_interp, eta_interp)

    pp, mm = np.meshgrid(up_interp,um_interp)
    urat = pp / mm
    # plt.imshow(urat)
    # plt.colorbar()
    # plt.show()
    min_urat = np.min(urat)
    max_urat = np.max(urat)
    urat_interp = np.linspace(min_urat,max_urat, nrho)
    return urat_interp
    
    K_exact = ef(np.pi/2, urat_interp)
    K_interp = interp1d(urat_interp, K_exact)


    # ff, kk =     


    fobs = np.complex128(ef(np.arcsin(np.sqrt(r31/r41)), k))
    Fobs = np.complex128(ef(np.arcsin(np.cos(inc)/np.sqrt(up)), up/um))




    # r1, r2, r3, r4 = get_radroots(lam, eta, a)
    # r31 = r3-r1
    # r32 = r3-r2
    # r42 = r4-r2
    # r41 = r4-r1
    


    return urat_interp







rho_interp = np.linspace(1e-3, 30,100)

out = build_all_interpolators(rho_interp)


# npix = 40
# pxi = (np.arange(npix)-0.01)/npix-0.5
# pxj = np.arange(npix)/npix-0.5
# # get angles measured north of west
# PXI,PXJ = np.meshgrid(pxi,pxj)
# varphi = np.arctan2(-PXJ,PXI)+1e-15# - np.pi/2
# # self.varphivec = varphi.flatten()

# #get grid of angular radii
# fov = 11
# mui = pxi*fov
# muj = pxj*fov
# MUI,MUJ = np.meshgrid(mui,muj)
# rho = np.sqrt(np.power(MUI,2.)+np.power(MUJ,2.))


# inc = 17/180*np.pi
# a = 0.01
# n = 1
# r = get_radius_exact(rho, varphi, inc, a, n)
# r[r<0] = 0
# plt.imshow(r)
# plt.colorbar()
# plt.show()


# a = 0.99
# inc = 60/180*np.pi
# n = 1
# r = get_radius_exact(rho, varphi, inc, a, n)
# r[r<0] = 0
# r[r>2*np.max(rho)] = 2*np.max(rho)
# plt.imshow(r)
# plt.colorbar()
# plt.show()
