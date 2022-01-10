"""
Implementation of the Kerr toy model for exact computation.
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.special import ellipj, ellipk, ellipkinc

minkmetric = np.diag([-1, 1, 1, 1])

np.seterr(invalid='ignore')
print("KerrBAM is silencing numpy warnings about invalid inputs (default: warn, now ignore). To undo, call np.seterr(invalid='warn').")
realkey = lambda x: np.real(x)


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
    return r1, r2, r3, r4


def Delta(r, a):
    return r**2 - 2*r + a**2

def Xi(r, a, theta):
    return (r**2+a**2)**2 - Delta(r, a)* a**2 * np.sin(theta)**2

def omega(r, a, theta):
    return 2*a*r/Xi(r, a, theta)

def Sigma(r, a, theta):
    return r**2 + a**2 * np.cos(theta)**2

def R(r, a, lam, eta):
    return (r**2 + a**2 - a*lam)**2 - Delta(r,a) * (eta + (a-lam)**2)

def getlorentzboost(boost, chi):
    gamma = 1 / np.sqrt(1 - boost**2) 
    coschi = np.cos(chi)
    sinchi = np.sin(chi)
    lorentzboost = np.array([[gamma, -gamma*boost*coschi, -gamma*boost*sinchi, 0],[-gamma*boost*coschi, (gamma-1)*coschi**2+1, (gamma-1)*sinchi*coschi, 0],[-gamma*boost*sinchi, (gamma-1)*sinchi*coschi, (gamma-1)*sinchi**2+1, 0],[0,0,0,1]])
    return lorentzboost

def kerr_exact(rho, varphi, inc, a, nmax, boost, chi, fluid_eta, thetabz):
    """
    Numerical: get rs from rho, varphi, inc, a, and subimage index n.
    """

    zeros = np.zeros_like(rho)
    npix = len(zeros)
    xdim = int(np.sqrt(npix))
    rp = 1+np.sqrt(1-a**2)
    alpha = rho*np.cos(varphi)
    beta = rho*np.sin(varphi)
    lam, eta = get_lam_eta(alpha,beta, inc, a)
    etamask = eta<0
    # print(etamask)
    up, um = get_up_um(lam, eta, a)
    urat = up/um
    r1, r2, r3, r4 = get_radroots(np.complex128(lam), np.complex128(eta), a)
    crit_mask = np.abs(np.imag(r3))>1e-10
    cr1 = np.real(r1)[crit_mask]
    cr2 = np.real(r2)[crit_mask]
    r31 = r3-r1
    r32 = r3-r2
    r42 = r4-r2
    r41 = r4-r1
    k = np.real(r32*r41 / (r31*r42))[~crit_mask]

    if fluid_eta is None:
        fluid_eta = chi+np.pi
    bz = np.cos(thetabz)
    beq = np.sqrt(1-bz**2)
    br = beq*np.cos(fluid_eta)
    bphi = beq*np.sin(fluid_eta)
    
    bvec = np.array([br, bphi, bz])
    Fobs_sin = np.clip(np.cos(inc)/np.sqrt(up), -1, 1)
    Fobs = ellipkinc(np.arcsin(Fobs_sin), urat)
    fobs = ellipkinc(np.real(np.arcsin(np.sqrt(r31/r41)[~crit_mask])), k)
    #note: only outside the critical curve, since nothing inside has a turning point
    Ir_turn = np.real(2/np.sqrt(r31*r42)[~crit_mask]*fobs)

    Ir_total = np.ones_like(rho)    
    Ir_total[~crit_mask] = 2*Ir_turn

    Agl = np.real(np.sqrt(r32*r42)[crit_mask])
    Bgl = np.real(np.sqrt(r31*r41)[crit_mask])
    k3 = ((Agl+Bgl)**2 - (cr2-cr1)**2)/(4*Agl*Bgl)
    I3r_angle = np.arccos((Agl-Bgl)/(Agl+Bgl))
    I3r = ellipkinc(I3r_angle, k3) / np.sqrt(Agl*Bgl)

    I3rp_angle = np.arccos((Agl*(rp-np.real(r1)[crit_mask])-Bgl*(rp-np.real(r2)[crit_mask]))/(Agl*(rp-np.real(r1)[crit_mask])+Bgl*(rp-np.real(r2)[crit_mask])))
    I3rp = ellipkinc(I3rp_angle, k3) / np.sqrt(Agl*Bgl)    
    
    Ir_total[crit_mask] = I3r - I3rp

    # return Ir_total
    Ir_total[etamask] = np.nan


    signpr = np.ones_like(rho)


    #everything before here is n-independent
    #build m array from beta sign and subimage index
    m = np.sign(beta)
    m[m>0] = 0
    rvecs = []
    qvecs = []
    uvecs = []
    ivecs = []
    vvecs = []
    redshifts = []
    lam=np.real(lam)
    eta=np.real(eta)
    for n in range(nmax+1):
        m += 1
        Ir = 1/np.sqrt(-um*a**2)*(2*m*ellipk(urat) - np.sign(beta)*Fobs)
        signpr[~crit_mask] = np.sign(Ir_turn-Ir[~crit_mask])
        X3 = np.sqrt(Agl*Bgl)*(Ir[crit_mask] - signpr[crit_mask] * I3r)
        cnnum = ellipj(X3, k3)[1]
        signptheta = (-1)**m * np.sign(beta)
        ffac = 1 / 2 * np.real(r31 * r42)**(1/2)
        snnum = np.ones_like(rho)
        snnum[~crit_mask] = ellipj(((ffac*Ir)[~crit_mask]-fobs), k)[0]
        snsqr = snnum**2
        snsqr[etamask] = np.nan
        snsqr[np.abs(snsqr)>1.1]=np.nan
        
        r = np.real((r4*r31 - r3*r41*snsqr) / (r31-r41*snsqr))
        
        r[crit_mask] = ((Bgl*cr2 - Agl*cr1) + (Bgl*cr2+Agl*cr1)*cnnum) / ((Bgl-Agl)+(Bgl+Agl)*cnnum)
        r[etamask] =np.nan
        r[Ir>Ir_total] = np.nan
        rvecs.append(np.nan_to_num(r))
        bigR = R(r, a, lam, eta)
        bigDelta = Delta(r, a)
        bigXi = Xi(r, a, np.pi/2)
        littleomega = omega(r, a, np.pi/2)

        #lowered
        pt_low = -1*np.ones_like(r)
        pr_low = signpr * np.sqrt(bigR)/bigDelta
        pr_low[pr_low>10] = 10
        pr_low[pr_low<-10] = -10
        # print(np.sum(np.isnan(pr_low)))
        pphi_low = lam
        ptheta_low = signptheta*np.sqrt(eta)
        plowers = np.array(np.hsplit(np.array([pt_low, pr_low, ptheta_low, pphi_low]),npix))

        #raised
        pt = 1/r**2 * (-a*(a-lam) + (r**2+a**2) * (r**2 + a**2 -a * lam) / bigDelta)
        pr = signpr * 1/r**2 * np.sqrt(bigR)
        pphi = 1/r**2 * (-(a-lam)+a/Delta(r,a)*(r**2+a**2 - a*lam))
        ptheta = signptheta*np.sqrt(eta) / r**2
        # praised.append([pt_up, pr_up, pphi_up, ptheta_up])
        #now everything to generate polarization
        
        emutetrad = np.array([[1/r*np.sqrt(bigXi/bigDelta), zeros, zeros, littleomega/r*np.sqrt(bigXi/bigDelta)], [zeros, np.sqrt(bigDelta)/r, zeros, zeros], [zeros, zeros, zeros, r/np.sqrt(bigXi)], [zeros, zeros, -1/r, zeros]])
        emutetrad = np.transpose(emutetrad,(2,0,1))
        boostmatrix = getlorentzboost(-boost, chi)
        #fluid frame tetrad
        coordtransform = np.matmul(np.matmul(minkmetric, boostmatrix), emutetrad)
        coordtransforminv = np.transpose(np.matmul(boostmatrix, emutetrad), (0,2, 1))
        rs = r
        pupperfluid = np.matmul(coordtransform, plowers)
        redshift = 1 / (pupperfluid[:,0,0])
        lp = np.abs(pupperfluid[:,0,0]/pupperfluid[:,3,0])
        
        #fluid frame polarization
        pspatialfluid = pupperfluid[:,1:]
        fupperfluid = np.cross(pspatialfluid, bvec, axisa = 1)
        fupperfluid = np.insert(fupperfluid, 0, 0, axis=2)# / (np.linalg.norm(pupperfluid[1:]))
        fupperfluid = np.swapaxes(fupperfluid, 1,2)
        vvec = np.dot(np.swapaxes(pspatialfluid,1,2), bvec).T[0]

        #apply the tetrad to get kerr f
        kfuppers = np.matmul(coordtransforminv, fupperfluid)
        kft = kfuppers[:,0,0]
        kfr = kfuppers[:,1,0]
        kftheta = kfuppers[:,2,0]
        kfphi = kfuppers[:, 3,0]
        spin = a
        #kappa1 and kappa2
        AA = (pt * kfr - pr * kft) + spin * (pr * kfphi - pphi * kfr)
        BB = (rs**2 + spin**2) * (pphi * kftheta - ptheta * kfphi) - spin * (pt * kftheta - ptheta * kft)
        kappa1 = rs * AA
        kappa2 = -rs * BB
        # kappa1 = np.clip(np.real(kappa1), -20, 20)
        
        #screen appearance
        nu = -(alpha + spin * np.sin(inc))

        norm = (nu**2 + beta**2) * np.sqrt(kappa1**2+kappa2**2)
        ealpha = (beta * kappa2 - nu * kappa1) / norm
        ebeta = (beta * kappa1 + nu * kappa2) / norm

        qvec = -(ealpha**2 - ebeta**2)
        uvec = -2*ealpha*ebeta
        
        qvec *= lp
        uvec *= lp
        ivec = np.sqrt(qvec**2+uvec**2)
        # rpmask = np.abs(r-rp) < 0.01*rp
        # ivec[rpmask]=0.
        # qvec[rpmask]=0.
        # uvec[rpmask]=0.
        # vvec[rpmask]=0.
        qvecs.append(np.real(np.nan_to_num(qvec)))
        uvecs.append(np.real(np.nan_to_num(uvec)))
        ivecs.append(np.real(np.nan_to_num(ivec)))
        vvecs.append(np.real(np.nan_to_num(vvec)))
        redshifts.append(np.real(np.nan_to_num(redshift)))



    return rvecs, ivecs, qvecs, uvecs, vvecs, redshifts


