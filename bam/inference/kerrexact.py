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


minkmetric = np.diag([-1, 1, 1, 1])


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


def Delta(r, a):
    return r**2 - 2*r + a**2

def Xi(r, a, inc):
    return (r**2+a**2)**2 - Delta(r, a)*np.sin(inc)**2

def omega(r, a, inc):
    return 2*a*r/Xi(r, a, inc)

def Sigma(r, a, inc):
    return r**2 + a**2 * np.cos(inc)**2

def R(r, a, lam, eta):
    return (r**2 + a**2 - a*lam)**2 - Delta(r,a) * (lam + (a-lam)**2)


def kerr_exact(rho, varphi, inc, a, nmax, beta, chi, br, bphi, bz, interp = True, interps = None):
    """
    Numerical: get rs from rho, varphi, inc, a, and subimage index n.
    """
    if interp:
        K_int, Fobs_int, fobs_outer_int, fobs_inner_ints, sn_outer_int, sn_inner_ints = interps
    
    alpha = rho*np.cos(varphi)
    beta = rho*np.sin(varphi)
    lam, eta = get_lam_eta(alpha,beta, inc, a)
    lam = np.complex128(lam)
    eta = np.complex128(eta)
    up, um = get_up_um(lam, eta, a)
    r1, r2, r3, r4 = get_radroots(lam, eta, a)
    crit_mask = np.abs(np.imag(r3))>1e-14
    r31 = r3-r1
    r32 = r3-r2
    r42 = r4-r2
    r41 = r4-r1
    k = r32*r41 / (r31*r42)
    if interp:
        Fobs = np.complex128(Fobs_int(np.arcsin(np.cos(inc)/np.sqrt(up)), urat))
        fobs = np.complex128(np.ones_like(rho))
        fobs[~crit_mask] = np.complex128(fobs_outer_int(np.real(np.arcsin(np.sqrt(r31/r41)[~crit_mask])), np.real(k[~crit_mask])))
        fobs[crit_mask] = fobs_inner_int_real(r31_phase[crit_mask], delta321_phase[crit_mask]) + 1j*fobs_inner_int_imag(r31_phase[crit_mask], delta321_phase[crit_mask])
    else:
        fobs = np.complex128(ef(np.arcsin(np.sqrt(r31/r41)), k))
        Fobs = np.complex128(ef(np.arcsin(np.cos(inc)/np.sqrt(up)), up/um))


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
    I3r_angle = np.arccos((Agl*(rp-np.real(r1)[crit_mask])-Bgl*(rp-np.real(r2)[crit_mask]))/(Agl*(rp-np.real(r1)[crit_mask])+Bgl*(rp-np.real(r2)[crit_mask])))
    
    if interp:
        I3r = fobs_outer_int(I3r_angle, k3)
    else:
        I3r = np.real(ef(I3r_angle, k3))
    #even though we are inside the critical curve, we will use the outer fobs interpolator since the args are real
    Ir_total[crit_mask] = 1/np.sqrt(Agl*Bgl)*ef(np.arccos((Agl-Bgl)/(Agl+Bgl)), k3)
    Ir_total[eta<0] = np.nan

    signpr = np.ones_like(Ir_turn)


    #everything before here is n-independent
    #build m array from beta sign and subimage index
    m = np.sign(beta)
    m[m>0] = 0
    rvecs = []
    qvecs = []
    uvecs = []
    ivecs = []
    # plowered = []
    # praised = []
    for n in range(nmax+1):
        m += 1
        if interp:
            Ir = np.real(1/np.sqrt(-um*a**2)*(2*m*np.complex128(K_int(urat)) - np.sign(beta)*Fobs))
        else:
            Ir = np.real(1/np.sqrt(-um*a**2)*(2*m*np.complex128(ef(np.pi/2, up/um)) - np.sign(beta)*Fobs))
        signpr[~crit_mask] = np.sign(Ir_turn-Ir)[~crit_mask]
        signptheta = (-1)**m * np.sign(beta)

        ffac = 1 / 2 * (r31 * r42)**(1/2)
        if interp:
            snnum = np.complex128(np.ones_like(rho))
            snnum[~crit_mask] = np.complex128(sn_outer_int((ffac*Ir-fobs)[~crit_mask], k[~crit_mask]))
            A = 1/2*np.sqrt(np.abs(r31*r42))*Ir
            
            sn_xk = sn_xk_int_real(A[crit_mask],delta321_phase[crit_mask])+1j*sn_xk_int_imag(A[crit_mask],delta321_phase[crit_mask])
            cndn_xk = cndn_xk_int_real(A[crit_mask],delta321_phase[crit_mask])+1j*cndn_xk_int_imag(A[crit_mask],delta321_phase[crit_mask])    

            sn_yk = sn_yk_int_real(r31_phase[crit_mask],delta321_phase[crit_mask])+1j*sn_yk_int_imag(r31_phase[crit_mask],delta321_phase[crit_mask])
            cndn_yk = cndn_yk_int_real(r31_phase[crit_mask],delta321_phase[crit_mask])+1j*cndn_yk_int_imag(r31_phase[crit_mask],delta321_phase[crit_mask])
            
            snnum[crit_mask] = (sn_xk*cndn_yk+sn_yk*cndn_xk)/(1-k[crit_mask]*(sn_xk*sn_yk)**2)
        else:
            snnum = np.complex128(sn(ffac*Ir-fobs,k))
        # snnum[~crit_mask] = np.complex128(sn_outer_int((ffac*Ir-fobs)[~crit_mask], k[~crit_mask]))
        snsqr = snnum**2
        
        rr = np.real((r4*r31 - r3*r41*snsqr) / (r31-r41*snsqr))
        r[eta<0] =np.nan
        r[Ir>Ir_total] = np.nan
        # rs.append(r)
        rvecs.append(r)
        bigR = R(r, a, lam, eta)
        bigDelta = Delta(r, a)

        #lowered
        pt_low = -1
        pr_low = signpr * np.sqrt(bigR)/bigDelta
        pphi_low = lam
        ptheta_low = signptheta*np.sqrt(eta)
        plowers = np.array([pt_low, pr_low, ptheta_low, pphi_low])
        # plowered.append([pt_low, pr_low, pphi_low, ptheta_low])

        #raised
        pt = 1/r**2 * (-a*(a-lam) + (r**2+a**2) * (r**2 + a**2 -a * lam) / bigDelta)
        pr = signpr * 1/r**2 * np.sqrt(bigR)
        pphi = 1/r**2 * (-(a-lam)+a/Delta(r,a)*(r**2+a**2 - a*lam))
        ptheta = signptheta*np.sqrt(eta) / r**2
        # praised.append([pt_up, pr_up, pphi_up, ptheta_up])


        #now everything to generate polarization
        emutetrad = np.array([[1/r*np.sqrt(Xi/d), 0, 0, omega/r*np.sqrt(Xi/d)], [0, np.sqrt(d)/r, 0, 0], [0, 0, 0, r/np.sqrt(Xi)], [0, 0, -1/r, 0]])

        coschi = np.cos(chi)
        sinchi = np.sin(chi)

        # #fluid frame tetrad
        coordtransform = np.matmul(np.matmul(minkmetric, getlorentzboost(-beta, chi)), emutetrad)
        coordtransforminv = np.transpose(np.matmul(getlorentzboost(-beta, chi), emutetrad))

        # #lowered momenta at source
        # signpr = 1 if setman==True else getsignpr(b, spin, theta0, varphi, mbar)
        # plowers = np.array([-1, signpr * np.sqrt(RR)/d, np.sign(np.cos(theta0))*((-1)**(mbar+1))*np.sqrt(eta0), lam])

        rs = r

        #raised
        # pt = 1 / (rs**2) * (-spin * (spin - lam) + (rs**2 + spin**2) * (rs**2 + spin**2 - spin * lam) / d)
        # pr = signpr * np.sqrt(RR) / rs**2
        # ptheta = np.sign(np.cos(theta0))*((-1)**(mbar+1))*np.sqrt(eta0) / rs**2
        # pphi = 1/(rs**2) * (-(spin -lam) + (spin * (rs**2 + spin**2 - spin * lam)) / d)

        #fluid frame momenta
        pupperfluid = np.matmul(coordtransform, plowers)
        redshift = 1 / (pupperfluid[0])

        lp = pupperfluid[0]/pupperfluid[3]

        #fluid frame polarization
        fupperfluid = np.cross(pupperfluid[1:], bvec)
        fupperfluid = (np.insert(fupperfluid, 0, 0)) / (np.linalg.norm(pupperfluid[1:]))

        #apply the tetrad to get kerr f
        kfuppers = np.matmul(coordtransforminv, fupperfluid)
        kft = kfuppers[0]
        kfr = kfuppers[1]
        kftheta = kfuppers[2]
        kfphi = kfuppers[3]

        #kappa1 and kappa2
        AA = (pt * kfr - pr * kft) + spin * (pr * kfphi - pphi * kfr)
        BB = (rs**2 + spin**2) * (pphi * kftheta - ptheta * kfphi) - spin * (pt * kftheta - ptheta * kft)
        kappa1 = rs * AA
        kappa2 = -rs * BB

        #screen appearance
        nu = -(alpha + spin * np.sin(theta0))
        ealpha = (beta * kappa2 - nu * kappa1) / (nu**2 + beta**2)
        ebeta = (beta * kappa1 + nu * kappa2) / (nu**2 + beta**2)
        # intensity = redshift**4 * (ealpha**2 + ebeta**2) * lp
        qvec = -(ealpha**2 - ebeta**2) * lp
        uvec = -2*ealpha*ebeta * lp
        ivec = np.sqrt(q**2+u**2)
        qvecs.append(qvec)
        uvecs.append(uvec)
        ivecs.append(ivec)
        redshifts.append(redshift)
        # if ebeta == 0 and not use2:
        #     angle = np.pi/2

        # else:
        #     angle = np.arctan2(ebeta, ealpha) if use2==True else -np.arctan(ealpha / ebeta)

        #normbad = ealpha**2+ebeta**2
        # ealpha *= redshift**2*np.sqrt(np.abs(lp))
        # ebeta *= redshift**2*np.sqrt(np.abs(lp))

        # ealpha *= np.sqrt(intensity / normbad)
        # ebeta *= np.sqrt(intensity / normbad)
        # return ealpha, ebeta
        # return [intensity, angle] if normalret == True else [ealpha, ebeta]



    return rvecs, ivecs, qvecs, uvecs, redshifts



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
    up[eta<0] = np.nan
    urat = np.minimum(np.real(up/um),1)
    r1, r2, r3, r4 = get_radroots(lam, eta, a)

    r31 = r3-r1
    r32 = r3-r2
    r42 = r4-r2
    r41 = r4-r1
    # r1 = np.real(r1)
    # r2 = np.real(r2)
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
    I3r_angle = np.arccos((Agl*(rp-np.real(r1)[crit_mask])-Bgl*(rp-np.real(r2)[crit_mask]))/(Agl*(rp-np.real(r1)[crit_mask])+Bgl*(rp-np.real(r2)[crit_mask])))
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
        rs.append(r)

        pt = 1/r**2 * (-a*(a-lam) + (r**2+a**2) * (r**2 + a**2 -a * lam) / Delta(r,a))
        pr = signpr * 1/r**2 * np.sqrt(R(r, a, lam, eta))
        pphi = 1/r**2 * (-(a-lam)+a/Delta(r,a)*(r**2+a**2 - a*lam))
        ptheta = signptheta*np.sqrt(eta) / r**2
        ps.append([pt, pr, pphi, ptheta])


    return rs, ps




#gets kappa and EVPA - a lot from Daniel's kappa script
def getevec(r, spin, plowers, puppers, theta0, beta, chi, bvec, normalret=True, use2=True, setman=False, retnew=False, retf=False):
    # alpha = b * np.cos(varphi)
    # beta = b * np.sin(varphi)
    # eta0 = beta**2 + (alpha**2 - spin**2) * (np.cos(theta0))**2


    # if np.min(eta0) <= 0:
    #     print('vortical')
    
    # d = r**2 - 2 * r + spin**2
    # Xi = (r**2 + spin**2)**2 - d * spin**2
    # omega = 2 * spin * r / Xi
    # sigma = r**2
    # lam = -alpha * np.sin(theta0)
    # eta0 = beta**2 + (alpha**2 - spin**2) * (np.cos(theta0))**2
    # RR = (r**2 + spin**2 - spin * lam)**2 - d * (eta0 + (spin - lam)**2)
    # bvec = np.asarray(bvec)
    # bvec /= np.linalg.norm(bvec)

    # #zamo frame tetrad
    emutetrad = np.array([[1/r*np.sqrt(Xi/d), 0, 0, omega/r*np.sqrt(Xi/d)], [0, np.sqrt(d)/r, 0, 0], [0, 0, 0, r/np.sqrt(Xi)], [0, 0, -1/r, 0]])

    # #minkowski metric
    # coschi = np.cos(chi)
    # sinchi = np.sin(chi)
    # minkmetric = np.diag([-1, 1, 1, 1])

    # #fluid frame tetrad
    coordtransform = np.matmul(np.matmul(minkmetric, getlorentzboost(-eta, chi)), emutetrad)
    coordtransforminv = np.transpose(np.matmul(getlorentzboost(-boost, chi), emutetrad))

    # #lowered momenta at source
    # signpr = 1 if setman==True else getsignpr(b, spin, theta0, varphi, mbar)
    # plowers = np.array([-1, signpr * np.sqrt(RR)/d, np.sign(np.cos(theta0))*((-1)**(mbar+1))*np.sqrt(eta0), lam])

    # rs = r

    #raised
    # pt = 1 / (rs**2) * (-spin * (spin - lam) + (rs**2 + spin**2) * (rs**2 + spin**2 - spin * lam) / d)
    # pr = signpr * np.sqrt(RR) / rs**2
    # ptheta = np.sign(np.cos(theta0))*((-1)**(mbar+1))*np.sqrt(eta0) / rs**2
    # pphi = 1/(rs**2) * (-(spin -lam) + (spin * (rs**2 + spin**2 - spin * lam)) / d)

    #fluid frame momenta
    pupperfluid = np.matmul(coordtransform, plowers)
    redshift = 1 / (pupperfluid[0])

    lp = 1#pupperfluid[0]/pupperfluid[3]

    #fluid frame polarization
    fupperfluid = np.cross(pupperfluid[1:], bvec)
    fupperfluid = (np.insert(fupperfluid, 0, 0)) / (np.linalg.norm(pupperfluid[1:]))

    #apply the tetrad to get kerr f
    kfuppers = np.matmul(coordtransforminv, fupperfluid)
    kft = kfuppers[0]
    kfr = kfuppers[1]
    kftheta = kfuppers[2]
    kfphi = kfuppers[3]

    if retf:
        return ptheta

    #kappa1 and kappa2
    AA = (pt * kfr - pr * kft) + spin * (pr * kfphi - pphi * kfr)
    BB = (rs**2 + spin**2) * (pphi * kftheta - ptheta * kfphi) - spin * (pt * kftheta - ptheta * kft)
    kappa1 = rs * AA
    kappa2 = -rs * BB
    if retnew:
        return [kappa1, kappa2]

    #screen appearance
    nu = -(alpha + spin * np.sin(theta0))
    ealpha = (beta * kappa2 - nu * kappa1) / (nu**2 + beta**2)
    ebeta = (beta * kappa1 + nu * kappa2) / (nu**2 + beta**2)
    intensity = redshift**4 * (ealpha**2 + ebeta**2) * lp

    if ebeta == 0 and not use2:
        angle = np.pi/2

    else:
        angle = np.arctan2(ebeta, ealpha) if use2==True else -np.arctan(ealpha / ebeta)

    #normbad = ealpha**2+ebeta**2
    ealpha *= redshift**2*np.sqrt(np.abs(lp))
    ebeta *= redshift**2*np.sqrt(np.abs(lp))

    # ealpha *= np.sqrt(intensity / normbad)
    # ebeta *= np.sqrt(intensity / normbad)

    return [intensity, angle] if normalret == True else [ealpha, ebeta]




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

def build_all_interpolators():
    """
    For now, a bunch of magic numbers that give decent interpolators for reasonable FOVs and resolutions.
    """

    ngrid = 100
    k = np.linspace(0,1,ngrid)
    fobsangle = np.linspace(0, np.pi/2,ngrid)
    fobs_outer_int = build_fobs_outer_interpolator(fobsangle, k)
    print("Built fobs outer interpolator.")
    r31_phase = np.linspace(-np.pi,np.pi,ngrid)
    delta321_phase = np.linspace(-np.pi,np.pi,ngrid)
    fobs_inner_ints = build_fobs_inner_interpolators(r31_phase, delta321_phase)
    print("Built fobs inner interpolators.")

    urat = np.linspace(-60,1,ngrid)
    Fobsangle = np.linspace(0, np.pi/2,ngrid)
    Fobs_int = build_Fobs_interpolator(Fobsangle, urat)
    print("Built Fobs interpolator.")
    K_int = build_K_interpolator(urat)
    print("Built K interpolator.")
    ffacIr_fobs_diff = np.linspace(-5,10,ngrid)
    sn_outer_int = build_sn_outer_interpolator(ffacIr_fobs_diff, k)
    print("Built sn outer interpolator.")
    A = np.linspace(0,3,ngrid)
    sn_inner_ints = build_sn_inner_interpolators(A, r31_phase, delta321_phase)
    print("Built sn inner interpolators.")
    return K_int, Fobs_int, fobs_outer_int, fobs_inner_ints, sn_outer_int, sn_inner_ints


#first, make lorentz transformation matrix
def getlorentzboost(boost, chi):
    gamma = 1 / np.sqrt(1 - boost**2)
    coschi = np.cos(chi)
    sinchi = np.sin(chi)
    lorentzboost = np.array([[gamma, -gamma*boost*coschi, -gamma*boost*sinchi, 0],[-gamma*boost*coschi, (gamma-1)*coschi**2+1, (gamma-1)*sinchi*coschi, 0],[-gamma*boost*sinchi, (gamma-1)*sinchi*coschi, (gamma-1)*sinchi**2+1, 0],[0,0,0,1]])
    return lorentzboost



# npix = 80
# pxi = (np.arange(npix)-0.01)/npix-0.5
# pxj = np.arange(npix)/npix-0.5
# # get angles measured north of west
# PXI,PXJ = np.meshgrid(pxi,pxj)
# varphi = np.arctan2(-PXJ,PXI)+1e-15# - np.pi/2
# # self.varphivec = varphi.flatten()

# #get grid of angular radii
# fov = 16
# mui = pxi*fov
# muj = pxj*fov
# MUI,MUJ = np.meshgrid(mui,muj)
# rho = np.sqrt(np.power(MUI,2.)+np.power(MUJ,2.))


# inc = 80/180*np.pi
# a = 0.5
# nmax = 2
# rhovec = rho.flatten()
# varphivec = varphi.flatten()
# rs_interped = kerr_interpolative(rhovec, varphivec, inc, a, nmax, K_int, Fobs_int,fobs_outer_int, fobs_inner_ints, sn_outer_int, sn_inner_ints)[0]
# # r[r<0] = 0
# # r0 = rs_interped[0]
# # r1 = rs_interped[1]
# # r1[r1>50]=50
# # r1[r1<0]=0
# plt.imshow(rs_interped[0].reshape((npix,npix)),extent=[-fov//2, fov//2, -fov//2, fov//2])
# plt.colorbar()
# plt.show()


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
