"""
Implementation of the Kerr toy model for use in interpolative BAM.
"""


import numpy as np
import mpmath as mp
import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline, interp1d, interp2d
import scipy.interpolate as si

#define elliptic functions (need the mpmath version to take complex args)
jacobi_ellip = np.frompyfunc(mp.ellipfun, 3, 1)
ef_base = np.frompyfunc(mp.ellipf, 2, 1)
def ef(u, k):
    return np.complex128(ef_base(u, k))
def sn(u, k):
    return np.complex128(jacobi_ellip('sn',u,k))
def cn(u, k):
    return np.complex128(jacobi_ellip('cn',u,k))
def dn(u, k):
    return np.complex128(jacobi_ellip('dn',u,k))


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
    H = -9*Q + np.sqrt(12*P**3 + 81*Q**2)
    z = np.sqrt((-2*(3**(1/3) * P) + 2**(1/3)*H**(2/3))/(2*6**(2/3)*H**(1/3))-A/6)
    r1 = -z - np.sqrt(-A/2 - z**2 + B/(4*z))
    r2 = -z + np.sqrt(-A/2 - z**2 + B/(4*z))
    r3 = z - np.sqrt(-A/2 - z**2 - B/(4*z))
    r4 = z + np.sqrt(-A/2 - z**2 - B/(4*z))
    return r1,r2,r3,r4



#get radial roots
def getlametatest(alpha, beta, spin, theta0, lam1=0, eta1=0, uselameta=False):
    lam = -alpha * np.sin(theta0)# * (not uselameta) + lam1 * uselameta
    eta0 = (beta**2 + (alpha**2 - spin**2) * (np.cos(theta0))**2)# * (not uselameta) + eta1 * uselameta
    return lam, eta0

#get radial roots
def getrootsrad(lam, eta, spin):#, spin, theta0, lam1=0, eta1=0, uselameta=False):
    # lam = -alpha * np.sin(theta0) * (not uselameta) + lam1 * uselameta
    # eta0 = (beta**2 + (alpha**2 - spin**2) * (np.cos(theta0))**2) * (not uselameta) + eta1 * uselameta
    Aconst  = spin**2-eta-lam**2 + 0j
    # print(Aconst)
    Bconst = 2*(eta + (lam - spin)**2)
    # print(Bconst)
    Cconst = -spin**2 * eta
    # print(Cconst)
    Pconst = -1 * Aconst**2 / 12 - Cconst
    print(Pconst)
    Qconst = -Aconst / 3 * (Aconst**2 / 36 - Cconst) - Bconst**2 / 8
    print(Qconst)
    discr = -9 * Qconst + np.sqrt(12 * Pconst**3 + 81 * Qconst**2 + 0j)

    #root to resolvent cubic using version of Cardano's method consistent with Mathematica (and python)
    xi = (-2 * 3**(1/3) * Pconst + 2**(1/3) * discr**(2/3)) / (6**(2/3) * discr**(1/3)) - Aconst / 3

    zconst = np.sqrt(xi / 2+0j)
    print(zconst)
    rootfac1 = -1 * Aconst / 2 - zconst**2 + Bconst / 4 / zconst
    rootfac2 = -1 * Aconst / 2 - zconst**2 - Bconst / 4 / zconst
    finalconst1 = np.sqrt(np.real(rootfac1)+(1j)*np.imag(rootfac1))
    finalconst2 = np.sqrt(np.real(rootfac2)+(1j)*np.imag(rootfac2))
    
    r1 = -1 * zconst - finalconst1
    r2 = -1 * zconst + finalconst1
    r3 = zconst - finalconst2
    r4 = zconst + finalconst2

    return np.array([r1, r2, r3, r4])


def kerr_exact(rho, varphi, inc, a, n):
    """
    Numerical: get rs from rho, varphi, inc, a, and subimage index n.
    """
    alpha = rho*np.cos(varphi)
    beta = rho*np.sin(varphi)
    lam, eta = get_lam_eta(alpha,beta, inc, a)
    lam = np.complex128(lam)
    eta = np.complex128(eta)
    up, um = get_up_um(lam, eta, a)
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
    ffac = 1 / 2 * (r31 * r42)**(1/2)
    snnum = np.complex128(sn(ffac*Ir-fobs,k))
    snsqr = snnum**2
    rs = np.real((r4*r31 - r3*r41*snsqr) / (r31-r41*snsqr))
    rs[eta<0] = np.nan
    return rs

def Delta(r, a):
    return r**2 - 2*r + a**2

def Sigma(r, a, inc):
    return r**2 + a**2 * np.cos(inc)**2

def R(r, a, lam, eta):
    return (r**2 + a**2 - a*lam)**2 - Delta(r,a) * (lam + (a-lam)**2)

def kerr_interpolative(rho, varphi, inc, a, nmax, K_int, Fobs_int, fobs_outer_int, fobs_inner_ints, sn_outer_int, sn_inner_ints):
    """
    Numerical: get rs from rho, varphi, inc, a, and subimage index n, interpolatively.
    """
    rp = 1+np.sqrt(1-a**2)
    fobs_inner_int_real, fobs_inner_int_imag = fobs_inner_ints
    sn_xk_int_real, sn_xk_int_imag, cndn_xk_int_real, cndn_xk_int_imag, sn_yk_int_real, sn_yk_int_imag, cndn_yk_int_real, cndn_yk_int_imag = sn_inner_ints

    alpha = rho*np.cos(varphi)
    beta = rho*np.sin(varphi)
    lam, eta = get_lam_eta(alpha,beta, inc, a)
    lam = np.complex128(lam)
    eta = np.complex128(eta)
    up, um = get_up_um(lam, eta, a)
    urat = np.minimum(np.real(up/um),1)
    urat[eta<0] = 1
    r1, r2, r3, r4 = get_radroots(lam, eta, a)

    r31 = r3-r1
    r32 = r3-r2
    r42 = r4-r2
    r41 = r4-r1
    r1 = np.real(r1)
    r2 = np.real(r2)
    crit_mask = np.abs(np.imag(r3))>1e-14

    k = r32*r41 / (r31*r42)
    print(np.any(np.real(k)<0))
    r31_phase = np.angle(r31)
    delta321_phase = np.angle(r32) - np.angle(r31)
    Fobs = np.complex128(Fobs_int(np.arcsin(np.cos(inc)/np.sqrt(up)), urat))
    fobs = np.complex128(np.ones_like(rho))
    fobs[~crit_mask] = np.complex128(fobs_outer_int(np.real(np.arcsin(np.sqrt(r31/r41)[~crit_mask])), np.real(k[~crit_mask])))
    fobs[crit_mask] = fobs_inner_int_real(r31_phase[crit_mask], delta321_phase[crit_mask]) + 1j*fobs_inner_int_imag(r31_phase[crit_mask], delta321_phase[crit_mask])
    
    #build mino time precursor values:
    #see appendix A of G+L lensing
    Ir_turn = np.real(2/np.sqrt(r31*r42)*fobs)
    
    # real_rp_rat = (rp-np.real(r4[~crit_mask])) / (rp-np.real(r3[~crit_mask])) * np.real(r31/r41)[~crit_mask]
    # print(np.max(real_rp_rat))
    # print(np.min(real_rp_rat))
    # I2r_angle = np.arcsin(np.sqrt(real_rp_rat))
    # I2r = fobs_outer_int(I2r_angle, np.real(k)[~crit_mask])
    Ir_total = 2*Ir_turn
    # Ir_total[~crit_mask] -= I2r

    Agl = np.real(np.sqrt(r32*r42)[crit_mask])
    Bgl = np.real(np.sqrt(r31*r41)[crit_mask])
    k3 = ((Agl+Bgl)**2 - np.real(r2-r1)[crit_mask]**2)/(4*Agl*Bgl)
    I3r_angle = np.arccos((Agl*(rp-r1[crit_mask])-Bgl*(rp-r2[crit_mask]))/(Agl*(rp-r1[crit_mask])+Bgl*(rp-r2[crit_mask])))
    I3r = fobs_outer_int(I3r_angle, k3)
    #even though we are inside the critical curve, we will use the outer fobs interpolator since the args are real
    Ir_total[crit_mask] = 1/np.sqrt(Agl*Bgl)*fobs_outer_int(np.arccos((Agl-Bgl)/(Agl+Bgl)), k3)
    Ir_total[eta<0] = np.nan
    # plt.imshow(Ir_total.reshape((80,80)))
    # plt.colorbar()
    # plt.title('Ir total')
    # plt.show()

    signpr = np.ones_like(Ir_turn)


    #everything before here is n-independent
    #build m array from beta sign and subimage index
    m = np.sign(beta)
    m[m>0] = 0
    rs = []
    ps = []
    for n in range(nmax+1):
        m += 1
        Ir = np.real(1/np.sqrt(-um*a**2)*(2*m*np.complex128(K_int(urat)) - np.sign(beta)*Fobs))
        signpr[~crit_mask] = np.sign(Ir_turn-Ir)[~crit_mask]
        signptheta = (-1)**m * np.sign(beta)

        ffac = 1 / 2 * (r31 * r42)**(1/2)
        snnum = np.complex128(np.ones_like(rho))
        snnum[~crit_mask] = np.complex128(sn_outer_int((ffac*Ir-fobs)[~crit_mask], k[~crit_mask]))
        
        A = 1/2*np.sqrt(np.abs(r31*r42))*Ir
        # print(np.min(A))

        sn_xk = sn_xk_int_real(A[crit_mask],delta321_phase[crit_mask])+1j*sn_xk_int_imag(A[crit_mask],delta321_phase[crit_mask])
        cndn_xk = cndn_xk_int_real(A[crit_mask],delta321_phase[crit_mask])+1j*cndn_xk_int_imag(A[crit_mask],delta321_phase[crit_mask])    

        sn_yk = sn_yk_int_real(r31_phase[crit_mask],delta321_phase[crit_mask])+1j*sn_yk_int_imag(r31_phase[crit_mask],delta321_phase[crit_mask])
        cndn_yk = cndn_yk_int_real(r31_phase[crit_mask],delta321_phase[crit_mask])+1j*cndn_yk_int_imag(r31_phase[crit_mask],delta321_phase[crit_mask])
        
        snnum[crit_mask] = (sn_xk*cndn_yk+sn_yk*cndn_xk)/(1-k[crit_mask]*(sn_xk*sn_yk)**2)
        snsqr = snnum**2
        
        r = np.real((r4*r31 - r3*r41*snsqr) / (r31-r41*snsqr))
        r[eta<0] =np.nan
        r[Ir>Ir_total] = np.nan
        # r[r<2]=np.nan
        rs.append(r)

        pt = 1/r**2 * (-a*(a-lam) + (r**2+a**2) * (r**2 + a**2 -a * lam) / Delta(r,a))
        pr = signpr * 1/r**2 * np.sqrt(R(r, a, lam, eta))
        pphi = 1/r**2 * (-(a-lam)+a/Delta(r,a)*(r**2+a**2 - a*lam))
        ptheta = signptheta*np.sqrt(eta) / r**2
        ps.append([pt, pr, pphi, ptheta])


    return rs, ps

def build_Fobs_interpolator(Fobs_angle, urat):
    FF, uu = np.meshgrid(Fobs_angle, urat)
    Fobs = np.real(ef(FF, uu))
    Fobs_int_base = interp2d(Fobsangle, urat, Fobs)#, bounds_error=False, fill_value=0)
    Fobs_int = lambda x, y: si.dfitpack.bispeu(Fobs_int_base.tck[0], Fobs_int_base.tck[1], Fobs_int_base.tck[2], Fobs_int_base.tck[3], Fobs_int_base.tck[4], x, y)[0]
    return Fobs_int

def build_K_interpolator(urat):
    K = ef(np.pi/2, urat)
    return interp1d(urat, K)

def build_fobs_outer_interpolator(fobs_angle, k):
    ff, kk = np.meshgrid(fobs_angle, k)
    fobs = np.real(ef(ff, kk))
    fobs_int_base = interp2d(fobsangle, k, fobs)#, bounds_error=False, fill_value=0)
    fobs_int = lambda x, y: si.dfitpack.bispeu(fobs_int_base.tck[0], fobs_int_base.tck[1], fobs_int_base.tck[2], fobs_int_base.tck[3], fobs_int_base.tck[4], x, y)[0]
    return fobs_int

def build_fobs_inner_interpolators(r31_phase, delta321_phase):
    rr, dd = np.meshgrid(r31_phase, delta321_phase)
    ff = np.arcsin(np.sqrt(np.exp(2j*rr)))
    kk = np.exp(2j*dd)
    fobs = ef(ff,kk)
    fobs_real = np.real(fobs)
    fobs_imag = np.imag(fobs)
    fobs_int_base_real = interp2d(r31_phase, delta321_phase, fobs_real)
    fobs_int_base_imag = interp2d(r31_phase, delta321_phase, fobs_imag)  
    fobs_int_real = lambda x, y: si.dfitpack.bispeu(fobs_int_base_real.tck[0],fobs_int_base_real.tck[1],fobs_int_base_real.tck[2],fobs_int_base_real.tck[3],fobs_int_base_real.tck[4],x,y)[0] 
    fobs_int_imag = lambda x, y: si.dfitpack.bispeu(fobs_int_base_imag.tck[0],fobs_int_base_imag.tck[1],fobs_int_base_imag.tck[2],fobs_int_base_imag.tck[3],fobs_int_base_imag.tck[4],x,y)[0] 
    return [fobs_int_real, fobs_int_imag]

def build_sn_outer_interpolator(ffacIr_fobs_diff, k):
    ff, kk = np.meshgrid(ffacIr_fobs_diff, k)
    sn_exact = np.complex128(sn(ff, kk))
    sn_real = np.real(sn_exact)
    # sn_imag = np.imag(sn_exact)
    sn_int_base_real = interp2d(ffacIr_fobs_diff, k, sn_real)#, bounds_error=False, fill_value=0)
    sn_int_real = lambda x, y: si.dfitpack.bispeu(sn_int_base_real.tck[0], sn_int_base_real.tck[1], sn_int_base_real.tck[2], sn_int_base_real.tck[3], sn_int_base_real.tck[4], x, y)[0]
    # sn_int_base_imag = interp2d(ffacIr_fobs_diff, k, sn_imag)
    # sn_int_imag = lambda x, y: si.dfitpack.bispeu(sn_int_base_imag.tck[0], sn_int_base_imag.tck[1], sn_int_base_imag.tck[2], sn_int_base_imag.tck[3], sn_int_base_imag.tck[4], x, y)[0]
    return sn_int_real#, sn_int_imag

def build_sn_inner_interpolators(A, r31_phase, delta321_phase):

    #First, deal with functions of A and delta321_phase
    #these are (x|k)
    AA, dd = np.meshgrid(A, delta321_phase)
    xx = AA * np.exp(-1j*dd / 2)
    kk = np.exp(2j*dd)
    sn_xk = sn(xx,kk)
    cn_xk = cn(xx,kk)
    dn_xk = dn(xx,kk)
    cndn_xk = cn_xk*dn_xk
    sn_xk_int_base_real = interp2d(A, delta321_phase, np.real(sn_xk))
    cndn_xk_int_base_real = interp2d(A, delta321_phase, np.real(cndn_xk))
    sn_xk_int_base_imag = interp2d(A, delta321_phase, np.imag(sn_xk))
    cndn_xk_int_base_imag = interp2d(A, delta321_phase, np.imag(cndn_xk))
    sn_xk_int_real = lambda x, y: si.dfitpack.bispeu(sn_xk_int_base_real.tck[0],sn_xk_int_base_real.tck[1],sn_xk_int_base_real.tck[2],sn_xk_int_base_real.tck[3],sn_xk_int_base_real.tck[4],x,y)[0]
    cndn_xk_int_real = lambda x, y: si.dfitpack.bispeu(cndn_xk_int_base_real.tck[0],cndn_xk_int_base_real.tck[1],cndn_xk_int_base_real.tck[2],cndn_xk_int_base_real.tck[3],cndn_xk_int_base_real.tck[4],x,y)[0]
    sn_xk_int_imag = lambda x, y: si.dfitpack.bispeu(sn_xk_int_base_imag.tck[0],sn_xk_int_base_imag.tck[1],sn_xk_int_base_imag.tck[2],sn_xk_int_base_imag.tck[3],sn_xk_int_base_imag.tck[4],x,y)[0]
    cndn_xk_int_imag = lambda x, y: si.dfitpack.bispeu(cndn_xk_int_base_imag.tck[0],cndn_xk_int_base_imag.tck[1],cndn_xk_int_base_imag.tck[2],cndn_xk_int_base_imag.tck[3],cndn_xk_int_base_imag.tck[4],x,y)[0]

    #Next, deal with functions of r31_phase and delta321_phase
    #these are (y|k) in the notes
    rr, dd = np.meshgrid(r31_phase, delta321_phase)
    ff = np.arcsin(np.sqrt(np.exp(2j*rr)))
    kk = np.exp(2j*dd)
    fobs = ef(ff,kk)
    yy = -fobs
    sn_yk = sn(yy, kk)
    cn_yk = cn(yy, kk)
    dn_yk = dn(yy, kk)
    cndn_yk = cn_yk*dn_yk
    sn_yk_int_base_real = interp2d(r31_phase, delta321_phase, np.real(sn_yk))
    cndn_yk_int_base_real = interp2d(r31_phase, delta321_phase, np.real(cndn_yk))
    sn_yk_int_base_imag = interp2d(r31_phase, delta321_phase, np.imag(sn_yk))
    cndn_yk_int_base_imag = interp2d(r31_phase, delta321_phase, np.imag(cndn_yk))
    sn_yk_int_real = lambda x, y: si.dfitpack.bispeu(sn_yk_int_base_real.tck[0],sn_yk_int_base_real.tck[1],sn_yk_int_base_real.tck[2],sn_yk_int_base_real.tck[3],sn_yk_int_base_real.tck[4],x,y)[0]
    cndn_yk_int_real = lambda x, y: si.dfitpack.bispeu(cndn_yk_int_base_real.tck[0],cndn_yk_int_base_real.tck[1],cndn_yk_int_base_real.tck[2],cndn_yk_int_base_real.tck[3],cndn_yk_int_base_real.tck[4],x,y)[0]
    sn_yk_int_imag = lambda x, y: si.dfitpack.bispeu(sn_yk_int_base_imag.tck[0],sn_yk_int_base_imag.tck[1],sn_yk_int_base_imag.tck[2],sn_yk_int_base_imag.tck[3],sn_yk_int_base_imag.tck[4],x,y)[0]
    cndn_yk_int_imag = lambda x, y: si.dfitpack.bispeu(cndn_yk_int_base_imag.tck[0],cndn_yk_int_base_imag.tck[1],cndn_yk_int_base_imag.tck[2],cndn_yk_int_base_imag.tck[3],cndn_yk_int_base_imag.tck[4],x,y)[0]

    return [sn_xk_int_real, sn_xk_int_imag, cndn_xk_int_real, cndn_xk_int_imag, sn_yk_int_real, sn_yk_int_imag, cndn_yk_int_real, cndn_yk_int_imag]


# def compare_sn(sn_inner_ints, A, r31_phase, delta321_phase):
#     sn_xk_int_real, sn_xk_int_imag, cndn_xk_int_real, cndn_xk_int_imag, sn_yk_int_real, sn_yk_int_imag, cndn_yk_int_real, cndn_yk_int_imag = sn_inner_ints

#     x = A*np.exp(-1j*delta321_phase/2)
#     fobs_angle = np.arcsin(np.sqrt(np.exp(2j*r31_phase)))
#     k = np.exp(2j*delta321_phase)
#     fobs = ef(fobs_angle, k)
#     y = -fobs
#     sn_xk = sn_xk_int_real(A,delta321_phase)+1j*sn_xk_int_imag(A,delta321_phase)
#     print("Exact sn_xk", sn(x,k))
#     print("Interped sn_xk", sn_xk)
#     cndn_xk = cndn_xk_int_real(A,delta321_phase)+1j*cndn_xk_int_imag(A,delta321_phase)   
#     print("Exact cndn_xk", cn(x,k)*dn(x,k)) 
#     print("Interped cndn_xk", cndn_xk)

#     sn_yk = sn_yk_int_real(r31_phase,delta321_phase)+1j*sn_yk_int_imag(r31_phase,delta321_phase)
#     print("Exact sn_yk", sn(y,k))
#     print("Interped sn_yk", sn_yk)
#     cndn_yk = cndn_yk_int_real(r31_phase,delta321_phase)+1j*cndn_yk_int_imag(r31_phase,delta321_phase)
#     print("Exact cndn_yk", cn(y,k)*dn(y,k)) 
#     print("Interped cndn_yk", cndn_yk)
    
#     sn_interped = (sn_xk*cndn_yk+sn_yk*cndn_xk)/(1-k*(sn_xk*sn_yk)**2)
#     print("Interpolated", sn_interped)
#     sn_exact = sn(x+y, k)
#     print("Exact", sn_exact)
#     # snsqr = snnum**2


# def test_summation(x, y, k):
#     print(sn(x+y,k))
#     print((sn(x,k)*cn(y,k)*dn(y,k) + sn(y,k)*cn(x,k)*dn(x,k))/(1-k*(sn(x,k)*sn(y,k))**2))

# ngrid = 100
# k = np.linspace(0,1,ngrid)
# fobsangle = np.linspace(0, np.pi/2,ngrid)
# fobs_outer_int = build_fobs_outer_interpolator(fobsangle, k)

# r31_phase = np.linspace(-np.pi,np.pi,ngrid)
# delta321_phase = np.linspace(-np.pi,np.pi,ngrid)
# fobs_inner_ints = build_fobs_inner_interpolators(r31_phase, delta321_phase)


# urat = np.linspace(-60,1,ngrid)
# Fobsangle = np.linspace(0, np.pi/2,ngrid)
# Fobs_int = build_Fobs_interpolator(Fobsangle, urat)

# K_int = build_K_interpolator(urat)

# ffacIr_fobs_diff = np.linspace(-5,10,ngrid)
# sn_outer_int = build_sn_outer_interpolator(ffacIr_fobs_diff, k)
# A = np.linspace(0,3,ngrid)
# sn_inner_ints = build_sn_inner_interpolators(A, r31_phase, delta321_phase)



# rho_interp = np.linspace(1e-3, 30,100)

# out = build_all_interpolators(rho_interp)


npix = 80
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


inc = 1e-5/180*np.pi
a = 0.5
nmax = 2
rhovec = rho.flatten()
varphivec = varphi.flatten()
rs_interped = kerr_interpolative(rhovec, varphivec, inc, a, nmax, K_int, Fobs_int,fobs_outer_int, fobs_inner_ints, sn_outer_int, sn_inner_ints)[0]
# r[r<0] = 0
# r0 = rs_interped[0]
# r1 = rs_interped[1]
# r1[r1>50]=50
# r1[r1<0]=0
plt.imshow(rs_interped[2].reshape((npix,npix)),extent=[-fov//2, fov//2, -fov//2, fov//2])
plt.colorbar()
plt.show()


# # a = 0.99
# # inc = 17/180*np.pi
# # n = 0
# r = kerr_exact(rhovec, varphivec, inc, a, nmax)
# # r[r<0]=0
# # r[r>2*np.max(rhovec)] = 0#2*np.max(rhovec)
# r[r==np.max(r)] = 0
# plt.imshow(r.reshape((npix,npix)),extent=[-fov//2, fov//2, -fov//2, fov//2])
# plt.colorbar()
# plt.show()



# plt.imshow(r.reshape((npix,npix))-r_interped.reshape((npix,npix)),extent=[-fov//2, fov//2, -fov//2, fov//2])
# plt.colorbar()
# plt.show()

# compare_sn(sn_inner_ints, 0.1, 0.3, 0.4)




# r[r<0] = 0
# r[r>2*np.max(rho)] = 2*np.max(rho)
# plt.imshow(r)
# plt.colorbar()
# plt.show()
