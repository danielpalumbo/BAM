"""
Implementation of the Kerr toy model for use in interpolative BAM.
"""


import numpy as np
import mpmath as mp
import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline, interp1d, interp2d
import scipy.interpolate as si

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
    print(np.arcsin(np.sqrt(r31/r41)))
    fobs = np.complex128(ef(np.arcsin(np.sqrt(r31/r41)), k))
    plt.imshow(np.imag(fobs).reshape((40,40)))
    plt.colorbar()
    plt.show()
    Fobs = np.complex128(ef(np.arcsin(np.cos(inc)/np.sqrt(up)), up/um))
    Ir = np.real(1/np.sqrt(-um*a**2)*(2*m*np.complex128(ef(np.pi/2, up/um)) - np.sign(beta)*Fobs))
    
    ffac = 1 / 2 * (r31 * r42)**(1/2)
    # plt.imshow(np.real(ffac*Ir-fobs))
    # plt.colorbar()
    # plt.show()
    snnum = np.complex128(sn('sn',ffac*Ir-fobs,k))

    # plt.imshow(np.angle(ffac*Ir-fobs).reshape((40,40)))
    # plt.title("Angle(ffac*Ir-fobs)")
    # plt.colorbar()
    # plt.show()
    # plt.imshow(np.angle(k).reshape((40,40)))
    # plt.title('Angle(k)')
    # plt.colorbar()
    # plt.show()
    # plt.imshow(np.imag(snnum).reshape((40,40)))
    # plt.colorbar()
    # plt.show()
    # plt.imshow(np.imag(ffac*Ir-fobs).reshape((40,40)))
    # plt.colorbar()
    # plt.show()
    # plt.imshow(np.imag(k).reshape((40,40)))
    # plt.colorbar()
    # plt.show()
    snsqr = np.complex128(sn('sn', ffac * Ir - fobs, k))**2
    # plt.imshow(np.real(snsqr).reshape((40,40)))
    # plt.colorbar()
    # plt.show()
    rs = np.real((r4*r31 - r3*r41*snsqr) / (r31-r41*snsqr))
    return rs

def get_radius_interpolative(rho, varphi, inc, a, n, K_int, fobs_int, Fobs_int, sn_int_real, sn_int_imag):
    """
    Numerical: get rs from rho, varphi, inc, a, and subimage index n, interpolatively.
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
    fobs = np.complex128(fobs_int(np.arcsin(np.sqrt(r31/r41)), k))
    Fobs = np.complex128(Fobs_int(np.arcsin(np.cos(inc)/np.sqrt(up)), up/um))
    plt.imshow(np.imag(fobs).reshape((40,40)))
    plt.colorbar()
    plt.show()
    Ir = np.real(1/np.sqrt(-um*a**2)*(2*m*np.complex128(K_int(up/um)) - np.sign(beta)*Fobs))
    # plt.imshow(np.real(Fobs).reshape((40,40)))
    # plt.colorbar()
    # plt.show()
    ffac = 1 / 2 * (r31 * r42)**(1/2)
    # plt.imshow(np.angle(ffac*Ir-fobs).reshape((40,40)))
    # plt.title("Angle(ffac*Ir-fobs)")
    # plt.colorbar()
    # # plt.savefig('angle(ffrac*Ir-fobs).png')
    # plt.show()
    # plt.imshow(np.angle(k).reshape((40,40)))
    # plt.title('Angle(k)')
    # plt.colorbar()
    # plt.show()
    sn_real = sn_int_real(ffac*Ir - fobs, k)
    sn_imag = sn_int_imag(ffac*Ir - fobs, k)
    # plt.imshow(sn_imag.reshape((40,40)))
    # plt.colorbar()
    # plt.show()
    # plt.imshow(np.imag(ffac*Ir-fobs).reshape((40,40)))
    # plt.colorbar()
    # plt.show()
    # plt.imshow(np.imag(k).reshape((40,40)))
    # plt.colorbar()
    # plt.show()
    sn_comp = sn_real+sn_imag*1j
    snsqr = sn_comp**2
    # plt.imshow(np.real(snsqr).reshape((40,40)))
    # plt.colorbar()
    # plt.show()
    rs = np.real((r4*r31 - r3*r41*snsqr) / (r31-r41*snsqr))
    return rs

def build_sn_interpolator(ffacIr_fobs_diff, k):
    ff, kk = np.meshgrid(ffacIr_fobs_diff, k)
    sn_exact = np.complex128(sn('sn',ff, kk))
    sn_real = np.real(sn_exact)
    sn_imag = np.imag(sn_exact)
    sn_int_base_real = interp2d(ffacIr_fobs_diff, k, sn_real)#, bounds_error=False, fill_value=0)
    sn_int_real = lambda x, y: si.dfitpack.bispeu(sn_int_base_real.tck[0], sn_int_base_real.tck[1], sn_int_base_real.tck[2], sn_int_base_real.tck[3], sn_int_base_real.tck[4], x, y)[0]
    sn_int_base_imag = interp2d(ffacIr_fobs_diff, k, sn_imag)
    sn_int_imag = lambda x, y: si.dfitpack.bispeu(sn_int_base_imag.tck[0], sn_int_base_imag.tck[1], sn_int_base_imag.tck[2], sn_int_base_imag.tck[3], sn_int_base_imag.tck[4], x, y)[0]
    return sn_int_real, sn_int_imag

def build_Fobs_interpolator(Fobsangle, urat):
    FF, uu = np.meshgrid(Fobsangle, urat)
    Fobs = np.real(np.complex128(ef(FF, uu)))
    Fobs_int_base = interp2d(Fobsangle, urat, Fobs)#, bounds_error=False, fill_value=0)
    Fobs_int = lambda x, y: si.dfitpack.bispeu(Fobs_int_base.tck[0], Fobs_int_base.tck[1], Fobs_int_base.tck[2], Fobs_int_base.tck[3], Fobs_int_base.tck[4], x, y)[0]
    return Fobs_int

def build_fobs_interpolator(fobsangle, k):
    ff, kk = np.meshgrid(fobsangle, k)
    fobs = np.real(np.complex128(ef(ff, kk)))    
    fobs_int_base = interp2d(fobsangle, k, fobs)#, bounds_error=False, fill_value=0)
    fobs_int = lambda x, y: si.dfitpack.bispeu(fobs_int_base.tck[0], fobs_int_base.tck[1], fobs_int_base.tck[2], fobs_int_base.tck[3], fobs_int_base.tck[4], x, y)[0]
    return fobs_int

def build_K_interpolator(urat):
    K = np.complex128(ef(np.pi/2, urat))
    return interp1d(urat, K)


k = np.linspace(-1,1)
fobsangle = np.linspace(0, np.pi/2)
fobs_int = build_fobs_interpolator(fobsangle, k)



urat = np.linspace(-10,10,100)
Fobsangle = np.linspace(0, np.pi/2)
Fobs_int = build_Fobs_interpolator(Fobsangle, urat)

K_int = build_K_interpolator(urat)

ffacIr_fobs_diff = np.linspace(-5,10)
sn_int_real, sn_int_imag = build_sn_interpolator(ffacIr_fobs_diff, k)



# rho_interp = np.linspace(1e-3, 30,100)

# out = build_all_interpolators(rho_interp)


npix = 40
pxi = (np.arange(npix)-0.01)/npix-0.5
pxj = np.arange(npix)/npix-0.5
# get angles measured north of west
PXI,PXJ = np.meshgrid(pxi,pxj)
varphi = np.arctan2(-PXJ,PXI)+1e-15# - np.pi/2
# self.varphivec = varphi.flatten()

#get grid of angular radii
fov = 16
mui = pxi*fov
muj = pxj*fov
MUI,MUJ = np.meshgrid(mui,muj)
rho = np.sqrt(np.power(MUI,2.)+np.power(MUJ,2.))


inc = 17/180*np.pi
a = 0.99
n = 0
rhovec = rho.flatten()
varphivec = varphi.flatten()
r_interped = get_radius_interpolative(rhovec, varphivec, inc, a, n, K_int, fobs_int, Fobs_int, sn_int_real, sn_int_imag)
# r[r<0] = 0
# plt.imshow(r_interped.reshape((npix,npix)),extent=[-fov//2, fov//2, -fov//2, fov//2])
# plt.colorbar()
# plt.show()


a = 0.99
inc = 17/180*np.pi
n = 0
r = get_radius_exact(rhovec, varphivec, inc, a, n)
# r[r==np.max(r)] = 0
# plt.imshow(r.reshape((npix,npix)),extent=[-fov//2, fov//2, -fov//2, fov//2])
# plt.colorbar()
# plt.show()



# plt.imshow(r.reshape((npix,npix))-r_interped.reshape((npix,npix)),extent=[-fov//2, fov//2, -fov//2, fov//2])
# plt.colorbar()
# plt.show()




# r[r<0] = 0
# r[r>2*np.max(rho)] = 2*np.max(rho)
# plt.imshow(r)
# plt.colorbar()
# plt.show()