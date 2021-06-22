#computes screen varphi and impact parameter as a function of phi and rs for a schwarzschild black hole-
#using formulas from 3dmc notes and arxiv:2010.07330, 2005.03856, 1910.12873

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

#get varphi from BL phi
def getvarphi(blphi, theta, n):
    return (np.arctan2(np.sin(blphi)*np.cos(theta), np.cos(blphi)) + n * np.pi)

#get BL phi from varphi
def getphi(varphi, theta, n):
    return (np.arctan2(np.sin(varphi),np.cos(varphi)*np.cos(theta))+n*np.pi)

#compute mino time gtheta
def gettau(b, varphi, n, theta):
    if theta==0:
        return ((2 * n + 1) / b * np.pi / 2)
    m = n*np.ones_like(varphi)
    m[np.sin(varphi)>0] = n+1
    # m = n if np.sin(varphi)<0 else n+1
    signphi = np.ones_like(varphi)
    signphi[np.sin(varphi)<0] = -1
    # sinphi = 1 if np.sin(varphi) >= 0 else -1 #just need to be consistent about defining sinvarphi\geq 0 together
    return 1/b * (np.pi * m - signphi*np.arcsin(np.cos(theta) / np.sqrt(np.cos(theta)**2 * np.cos(varphi)**2 + np.sin(varphi)**2)))


#closed form expression for rs as a function of screen coords (to be inverted)
def getradroots(b): #note!!! this needs to take the G+L roots for kerr and *then* set a=0 (else everything complex conjugated)
    fac1 = (-b**6 + 6*b**4 * (9 + np.sqrt(81-3*b**2 + 0j)))**(1/3)
    xi = (b**2 + fac1)**2 / 6 / fac1
    z0 = (xi / 2)**(1/2)
    a0 = -b**2
    b0 =  2*(b**2)
    rootfac1 = (-a0 / 2 - z0**2 + b0 / 4 / z0)**(1/2)
    rootfac2 = (-a0 / 2 - z0**2 - b0 / 4 / z0)**(1/2)
    r1 = -z0 - rootfac1
    r3 = z0 - rootfac2
    r4 = z0 + rootfac2
    return r1, r3, r4 #r2 = 0 by default



def rinvert(b, varphi, n, theta):
    tau = gettau(b, varphi, n, theta)
    bratio = np.complex128(np.sqrt(27) / b)
    rootfac = -bratio + np.sqrt(-1 + bratio**2)
    r1, r3, r4 = getradroots(b)
    r31 = r3 - r1
    r41 = r4 - r1
    k = r3 * r41 / r31 / r4
    ffac = 1 / 2 * (r31 * r4)**(1/2)
    fobs = np.complex128(ef(np.arcsin(np.sqrt(r31/r41)), k))
    snnum = r41*(np.complex128(sn('sn', ffac * tau - fobs, k)))**2
    rs = np.real((r4 * r31 - r3 * snnum) / (r31 - snnum))
    return rs

def r_from_rho_and_tau(b, tau, r1, r3, r4):
    # bratio = np.complex128(np.sqrt(27) / b)
    # rootfac = -bratio + np.sqrt(-1 + bratio**2)
    # r1, r3, r4 = getradroots(b)
    r31 = r3 - r1
    r41 = r4 - r1
    k = r3 * r41 / r31 / r4
    ffac = 1 / 2 * (r31 * r4)**(1/2)
    fobs = np.complex128(ef(np.arcsin(np.sqrt(r31/r41)), k))
    snnum = r41*(np.complex128(sn('sn', ffac * tau - fobs, k)))**2
    rs = np.real((r4 * r31 - r3 * snnum) / (r31 - snnum))
    return rs    

def gettautot(b, r1, r3, r4):
    # r1, r3, r4 = getradroots(b) #get radial roots - r4 will be complex if inside crit curve (which is fine)

    ifacout = lambda r: np.abs(1/np.sqrt(r*(r-r1)*(r-r3)))
    iintout = lambda u: ifacout(u**2+r4)
    irtotout = 4*quad_vec(iintout, 0, np.inf)[0] #outside crit curve - needs change of variables to make sure integrand doesn't explode

    iintin = lambda r: 1/np.sqrt(r**4-r*(r-2)*b**2)
    irtotin = 2*quad_vec(iintin, 2, np.inf)[0] #inside crit curve - no change of variables necessary since denom has no roots

    return np.nan_to_num((b<np.sqrt(27))*irtotin,nan=0) + np.nan_to_num((b>np.sqrt(27))*irtotout,nan=0)


# #screen impact param is the root of this equation
# def geodesiceq(b):
#     global varphi1
#     global theta1
#     global r1
#     global n1

#     return np.abs(r1-rinvert(b, varphi1, n1, theta1))

#find the root using scipy.optimize
def findb_base(r, varphi, theta, n):

    varphi1 = varphi
    theta1 = theta
    r1 = r
    n1 = n
    
    #screen impact param is the root of this equation
    def geodesiceq(b):

        return np.abs(r1-rinvert(b, varphi1, n1, theta1))


    #findroot params
    tol0 = 1e-10
    maxiter = 1000
    options0 = {'maxiter' : maxiter}

    #set seed values - add one for n=0 and subring approx for n>0
    if n==0:
        guess = r + 1
    else:
        bc = np.sqrt(27)
        cp = 1944 / (12 + 7 * np.sqrt(3))
        m = n if np.sin(varphi) < 0 else n+1
        gamma = np.pi
        guess = bc + 1 / cp * np.exp(-gamma * (m + 0.5)) #face-on, bound orbit approximation, assumes outside crit curve (which I think is true for rs>4)
        
    solb = op.root(geodesiceq, guess, method='lm', tol=tol0, options=options0)

    #check to make sure numerics didn't mess anything up
    if solb.fun / r >= 1e-6:
        print('err is {} at r={} and varphi = {} and theta = {}'.format(solb.fun / r, r, varphi, theta))

    # listerr.append(solb.fun / r)
    return (solb.x[0])

findb = np.vectorize(findb_base)

#numerically compute screen coordinates
def getscreencoords_base(r, blphi, theta, n):
    varphi = getvarphi(blphi, theta, n)
    b = findb(r, varphi, theta, n)
    return b, varphi

getscreencoords = np.vectorize(getscreencoords_base)


#now we need to find the winding number. there are a couple steps to this.
#first, compute psin using 3mdc formalism

def getpsin(theta, blphi, n):
    psib = np.arccos(-np.sin(theta) * np.sin(blphi)) % np.pi if n % 2 == 0 else -2 * np.pi + np.arccos(-np.sin(theta) * np.sin(blphi))
    psin = psib + int(n / 2) * (-1)**n * 2 * np.pi
    return psin


#gets psit as a function of b
def getpsit(b, r1, r3, r4):
    # r1, r3, r4 = getradroots(b)
    r31 = r3 - r1
    r41 = r4 - r1
    k = r3 * r41 / r31 / r4
    argx2 = r31  / r41
    x2 = np.arcsin(np.sqrt(argx2))
    prefac = 2 * b / np.sqrt(r31 * r4)
    psit = prefac * np.complex128(ef(x2, k))
    return psit




#next, need sign(p^r) at the emission radius to determine if ray hits turning point
def getsignpr(b, psin):
    # if b <= np.sqrt(27):
    #     return 1
    psit = getpsit(b)
    out = (np.abs(psin) < psit).astype(int)
    out[np.abs(psin)<psit] = -1
    out[b <= np.sqrt(27)]=1
    return out
    # if np.abs(psin) < psit:
    #     return 1
    # elif np.abs(psin) > psit:
    #     return -1
    # return 0

#now compute alpha_n
def getalphan(b, r, theta, psin):
    signpr = getsignpr(b, psin)
    arctannum = np.arctan(1 / np.sqrt(r**2/b**2/(1-2/r)-1))
    signpsin = np.sign(psin)
    out = signpsin * (np.pi-arctannum)
    # mask = (signpr == 1)*(0<psin)*(psin<np.pi)
    # out[mask] = (signpsin*arctannum)[mask]
    return out

    # if signpr == 1 or (0<psin<np.pi):
    #     return signpsin * arctannum
    # elif signpr == -1:
    #     return signpsin * (np.pi - arctannum)
    # return (signpsin * np.pi / 2)

#subtract from psi_n  to get total winding angle = xi_n^R
def getwindangle(b, r, blphi, theta, n):
    psin = getpsin(theta, blphi, n)
    alphan = getalphan(b, r, theta, psin)
    return (psin - alphan)


# def build_psit_interpolator(rho_interp):
#     #compute exact psit these rhos
#     exact_psit = getpsit(rho_interp)

#     #build interpolator
#     eint = interp1d(rho_interp,exact_psit)
#     return eint    


# def build_r_interpolator(rho_interp,tau_interp):
#     rv, tv = np.meshgrid(rho_interp, tau_interp)
#     #compute exact r
#     exact_r = r_from_rho_and_tau(rv, tv)
#     eint = RectBivariateSpline(rho_interp, tau_interp, exact_r)
#     # eint = interp2d(rv, tv, exact_r)
#     return eint

def build_all_interpolators(rho_interp, nmax=0):
    """
    Given a set of rho values, 
    buid reasonable interpolators for radius, total Mino time, and psit.

    The root finding is done only once.
    """
    nrho = len(rho_interp)
    r1, r3, r4 = getradroots(rho_interp)
    print('Number of points in rho: ',nrho)

    psit = getpsit(rho_interp, r1, r3, r4) #note: contains imaginary components for b < sqrt(27)
    psit_interpolator = interp1d(rho_interp, np.real(psit), bounds_error=False, fill_value='extrapolate')

    tautot = gettautot(rho_interp, r1, r3, r4)
    max_tau = np.max(tautot)
    print("Interpolating over tau from 0.0001 to "+str(max_tau))
    # tau_interp_lowend = gettau(rho_interp, 3*np.pi/2*np.ones_like(rho_interp),0, np.pi/2)
    # tau_interp_highend = gettau(rho_interp, np.pi/2*np.ones_like(rho_interp),nmax, np.pi/2)
    # tau_interp = np.linspace(np.min(tau_interp_lowend),np.max(tau_interp_highend),nrho)
    # tau_interp = np.minimum(max_tau, tau_interp)
    tau_interp = np.linspace(0.0001,max_tau,nrho) 
    tautot_interpolator = interp1d(rho_interp, tautot, bounds_error=False, fill_value='extrapolate')

    rr, tt = np.meshgrid(rho_interp, tau_interp)
    r11, tt = np.meshgrid(r1, tau_interp)
    r33, tt = np.meshgrid(r3, tau_interp)
    r44, tt = np.meshgrid(r4, tau_interp)
    r = r_from_rho_and_tau(rr, tt, r11, r33, r44)
    print("Computed r values when building interpolator:")
    print("Mean", np.mean(r))
    print("Median",np.median(r))
    print("Max",np.max(r))
    print("Min",np.min(r))
    print("Nans", np.sum(np.isnan(r)))
    # plt.imshow(rr)
    # plt.colorbar()
    # plt.show()
    # plt.imshow(tt)
    # plt.colorbar()
    # plt.show()
    # plt.imshow(r)
    # plt.colorbar()
    # plt.show()
    r[r<0] = -r[r<0]
    r[r>5*rho_interp[-1]]=5*rho_interp[-1]
    r = np.nan_to_num(r)
    print("Computed r values when building interpolator after cleanup:")
    print("Mean", np.mean(r))
    print("Median",np.median(r))
    print("Max",np.max(r))
    print("Min",np.min(r))
    print("Nans", np.sum(np.isnan(r)))
    
    # r_interpolator = RectBivariateSpline(tau_interp, rho_interp, r)
    r_int = interp2d(rho_interp, tau_interp, r)#, bounds_error=False, fill_value=0)


    r_interpolator = lambda x, y: si.dfitpack.bispeu(r_int.tck[0], r_int.tck[1], r_int.tck[2], r_int.tck[3], r_int.tck[4], x, y)[0]

    return r_interpolator, tautot_interpolator, psit_interpolator

def exact(rhovec, varphivec, inc, nmax):
    b = rhovec
    varphivec[varphivec==0.]+=1e-5
    varphi = varphivec
    theta = inc
    rvecs = []
    for n in range(nmax+1):
        tau = gettau(b, varphi, n, theta)
        bratio = np.complex128(np.sqrt(27) / b)
        rootfac = -bratio + np.sqrt(-1 + bratio**2)
        r1, r3, r4 = getradroots(b)
        r31 = r3 - r1
        r41 = r4 - r1
        k = r3 * r41 / r31 / r4
        ffac = 1 / 2 * (r31 * r4)**(1/2)
        fobs = np.complex128(ef(np.arcsin(np.sqrt(r31/r41)), k))
        snnum = r41*(np.complex128(sn('sn', ffac * tau - fobs, k)))**2
        rvecs.append(np.real((r4 * r31 - r3 * snnum) / (r31 - snnum)))

    phivecs = [getphi(varphivec, inc, n) for n in range(nmax+1)]
    psivec = getpsin(inc, phivecs[0], 0)
    psivecs = [psivec + n*np.pi for n in range(nmax+1)]
    # psivecs = [getpsin(inc, phivecs[0], 0) for n in range(nmax+1)]

    alphavecs = []
    psit = getpsit(b)
    # plt.imshow(2*psit.reshape(int(np.sqrt(len(psit))),int(np.sqrt(len(psit)))))
    # plt.show()
    for n in range(nmax+1):
        # signpr = getsignpr(b, psin)
        r = rvecs[n]
        # psin = getpsin(theta, phivecs[n],n)
        psin = psivecs[n]
        if n > 0:
            nosubim_mask = 2*psit < psivecs[0]+np.pi
            rvecs[n][nosubim_mask] = np.nan
        out = (np.abs(psin) < psit).astype(int)
        out[np.abs(psin)<psit] = -1
        out[b <= np.sqrt(27)]=1
        signpr = out

        arctannum = np.arctan(1 / np.sqrt(r**2/b**2/(1-2/r)-1))
        signpsin = np.sign(psin)
        out = signpsin * arctannum#(np.pi-arctannum)
        # mask = (signpr == 1)*(0<psin)*(psin<np.pi)
        # out[mask] = (signpsin*arctannum)[mask]
        alphavecs.append(out)
    # return out
    # alphavecs = [getalphan(rhovec, rvecs[n], inc, psivecs[n]) for n in range(self.nmax+1)]
    return rvecs, phivecs, psivecs, alphavecs


def interpolative_exact(rhovec, varphivec, inc, nmax, r_interpolator, tautot_interpolator, psit_interpolator):
    """
    Given 1d (flattened image) arrays of rho and varphi
    as well as inclination and the maximum n,
    compute vectors of r, phi, psi and alpha
    using interpolated values of r, total Mino time, psit.
    """
    rvecs = []

    for n in range(nmax+1):
        tau = gettau(rhovec, varphivec, n, inc)
        tautot = tautot_interpolator(rhovec)
        # tau[tau>tautot]=np.nan
        npix = int(np.sqrt(len(tau)))
        rvecs.append(r_interpolator(rhovec, tau))


    phivecs = [getphi(varphivec, inc, n) for n in range(nmax+1)]
    psivec = getpsin(inc, phivecs[0], 0)
    psivecs = [psivec + n*np.pi for n in range(nmax+1)]
    alphavecs = []
    psit=psit_interpolator(rhovec)
    for n in range(nmax+1):
        r = rvecs[n]
        psin = psivecs[n]
        if n > 0:
            nosubim_mask = 2*psit < psivecs[0]+np.pi
            rvecs[n][nosubim_mask] = np.nan
        out = (np.abs(psin) < psit).astype(int)
        out[np.abs(psin)<psit] = -1
        out[rhovec <= np.sqrt(27)]=1
        signpr = out

        arctannum = np.arctan(1 / np.sqrt(r**2/rhovec**2/(1-2/r)-1))
        signpsin = np.sign(psin)
        out = signpsin * arctannum#(np.pi-arctannum)
        # mask = (signpr == 1)*(0<psin)*(psin<np.pi)
        # out[mask] = (signpsin*arctannum)[mask]
        alphavecs.append(out)

    return rvecs, phivecs, psivecs, alphavecs







