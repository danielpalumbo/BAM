#computes screen varphi and impact parameter as a function of phi and rs for a schwarzschild black hole-
#using formulas from 3dmc notes and arxiv:2010.07330, 2005.03856, 1910.12873

import numpy as np
import scipy.optimize as op
import mpmath as mp
import matplotlib.pyplot as plt


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
    sinphi = np.ones_like(varphi)
    sinphi[np.sin(varphi)<0] = -1
    # sinphi = 1 if np.sin(varphi) >= 0 else -1 #just need to be consistent about defining sinvarphi\geq 0 together
    return 1/b * (np.pi * m - sinphi*np.arcsin(np.cos(theta) / np.sqrt(np.cos(theta)**2 * np.cos(varphi)**2 + np.sin(varphi)**2)))


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
def getpsit(b):
    r1, r3, r4 = getradroots(b)
    r31 = r3 - r1
    r41 = r4 - r1
    k = r3 * r41 / r31 / r4
    argx2 = r31  / r41
    x2 = np.arcsin(np.sqrt(argx2))
    prefac = 2 * b / np.sqrt(r31 * r4)
    psit = np.abs(prefac * np.complex128(ef(x2, k)))
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
    mask = (signpr == 1)*(0<psin)*(psin<np.pi)
    out[mask] = (signpsin*arctannum)[mask]
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
# def main():
#     global listerr;
#     listerr=[]

#     theta = 89*np.pi/180
#     rvals = np.arange(2.8, 3.5, .01)
#     varphi = -np.pi/2
#     n = 1
# #    blphivals = np.linspace(0, 2*np.pi, 50)
#     varphivals = getvarphi(blphivals, theta, n)
#     bvals = findb
# #    bvals = np.array([getscreencoords(r, blphi, theta, n)[0] for blphi in blphivals])
#     psinvals = [getpsin(theta, blphi, n) for blphi in blphivals]
#     signvals = [getsignpr(bvals[i], r, theta, psinvals[i]) for i in range(len(blphivals))]
#     for i in range(len(signvals)):
#         print('varphi: {}, sign: {}'.format(np.sin(varphivals[i]), signvals[i]))

# if __name__=="__main__":
#     main()
