"""
Implementation of the Kerr toy model for exact computation.
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.special import ellipj, ellipk, ellipkinc
from scipy.interpolate import griddata
from skimage.transform import rescale, resize
from scipy.signal import convolve2d
from bam.inference.model_helpers import get_rho_varphi_from_FOV_npix


minkmetric = np.diag([-1, 1, 1, 1])

kernel = np.ones((3,3))

np.seterr(invalid='ignore')
print("KerrBAM is silencing numpy warnings about invalid inputs (default: warn, now ignore). To undo, call np.seterr(invalid='warn').")


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




def emissivity_model(r, phi, a, lam, eta, signpr, boost, chi, fluid_eta, iota, compute_V=False):
    npix = int(np.sqrt(len(r)))
    zeros = np.zeros_like(r)
    bigR = R(r, a, lam, eta)
    bigDelta = Delta(r, a)
    bigXi = Xi(r, a, np.pi/2)
    littleomega = omega(r, a, np.pi/2)


    if fluid_eta is None:
        fluid_eta = chi+np.pi
    bz = np.cos(iota)
    beq = np.sqrt(1-bz**2)
    br = beq*np.cos(fluid_eta)
    bphi = beq*np.sin(fluid_eta)
    
    bvec = np.array([br, bphi, bz])

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
    if compute_V:
        vvec = np.dot(np.swapaxes(pspatialfluid,1,2), bvec).T[0]
    else:
        vvec = zeros
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
    # ivec = np.sqrt(qvec**2+uvec**2)
    # rpmask = np.abs(r-rp) < 0.01*rp
    # ivec[rpmask]=0.
    # qvec[rpmask]=0.
    # uvec[rpmask]=0.
    # vvec[rpmask]=0.
    # ivec = np.real(np.nan_to_num(ivec))
    qvec = np.real(np.nan_to_num(qvec))
    uvec = np.real(np.nan_to_num(uvec))
    if compute_V:
        vvec = np.real(np.nan_to_num(vvec))
    redshift = np.real(np.nan_to_num(redshift))
    # if adap_fac > 1 and nmax>0:
    #     rvec = rescale(rvec.reshape((xdim,xdim)),adap_fac,order=1).flatten()
    #     # ivec = rescale(ivec.reshape((xdim,xdim)),adap_fac,order=1).flatten()
    #     qvec = rescale(qvec.reshape((xdim,xdim)),adap_fac,order=1).flatten()
    #     uvec = rescale(uvec.reshape((xdim,xdim)),adap_fac,order=1).flatten()
        
    #     if compute_V:
    #         vvec = rescale(vvec.reshape((xdim,xdim)),adap_fac,order=1).flatten()
    #     else:
    #         vvec = np.zeros(adap_fac**2*xdim**2)
    #     redshift = rescale(redshift.reshape((xdim,xdim)),adap_fac,order=1).flatten()
    ivec = np.sqrt(qvec**2+uvec**2)

#these should return r, phi, tau, tau_tot

def ray_trace_by_case(a, alpha, beta, r1, r2, r3, r4, up, um, inc, nmax, case, adap_fac= 1,axisymmetric = True, nmin=0):
    """
    Case 1: r1, r2, r3, r4 are real.
    Returns 
    """
    if len(alpha) == 0:
        return [[] for n in range(nmin, nmax+1)], [[] for n in range(nmin, nmax+1)], [[] for n in range(nmin, nmax+1)], [[] for n in range(nmin, nmax+1)]
    r31 = r3-r1
    r32 = r3-r2
    r42 = r4-r2
    r41 = r4-r1
    k = (r32*r41 / (r31*r42))
    urat = up/um
    m = np.sign(beta) + nmin 
    m[m>0] = 0
    rvecs = []
    phivecs = []
    Irmasks = []
    signprs = []
    if case == 1:
        Fobs_sin = np.clip(np.cos(inc)/np.sqrt(up), -1, 1)
        Fobs = ellipkinc(np.arcsin(Fobs_sin), urat)
        fobs = ellipkinc(np.arcsin(np.sqrt(r31/r41)), k)
        #note: only outside the critical curve, since nothing inside has a turning point
        Ir_turn = np.real(2/np.sqrt(r31*r42)*fobs)
        Ir_total = 2*Ir_turn

        # signpr = np.ones_like(rho)



        # redshifts = []

        for n in range(nmin, nmax+1):
            m += 1
            Ir = 1/np.sqrt(-um*a**2)*(2*m*ellipk(urat) - np.sign(beta)*Fobs)
            Irmask = Ir<Ir_total
            signpr = np.sign(Ir_turn-Ir)
            signptheta = (-1)**m * np.sign(beta)
            ffac = 1 / 2 * np.real(r31 * r42)**(1/2)
            # snnum = np.ones_like(rho)
            snnum = ellipj(((ffac*Ir)-fobs), k)[0]
            snsqr = snnum**2
            # snsqr[eta_mask] = np.nan
            snsqr[np.abs(snsqr)>1.1]=np.nan
            
            r = np.real((r4*r31 - r3*r41*snsqr) / (r31-r41*snsqr))
            # rvecs.append(r)
            # r[eta_mask] =np.nan
            r[~Irmask] = np.nan
            rvec = np.nan_to_num(r)
            Irmasks.append(Irmask)
            if not axisymmetric:
                pass

    if case == 3:
        Agl = np.real(np.sqrt(r32*r42))
        Bgl = np.real(np.sqrt(r31*r41))
        k3 = ((Agl+Bgl)**2 - (r2-r1)**2)/(4*Agl*Bgl)
        I3r_angle = np.arccos((Agl-Bgl)/(Agl+Bgl))
        I3r = ellipkinc(I3r_angle, k3) / np.sqrt(Agl*Bgl)

        Fobs_sin = np.clip(np.cos(inc)/np.sqrt(up), -1, 1)
        Fobs = ellipkinc(np.arcsin(Fobs_sin), urat)

        I3rp_angle = np.arccos((Agl*(rp-r1)-Bgl*(rp-r2))/(Agl*(rp-r1)+Bgl*(rp-r2)))
        I3rp = ellipkinc(I3rp_angle, k3) / np.sqrt(Agl*Bgl)    
        Ir_total = I3r - I3rp
        signpr = np.ones_like(r31)
        for n in range(nmin, nmax+1):
            m += 1
            Ir = 1/np.sqrt(-um*a**2)*(2*m*ellipk(urat) - np.sign(beta)*Fobs)
            Irmask = Ir<Ir_total

            X3 = np.sqrt(Agl*Bgl)*(Ir - signpr * I3r)
            cnnum = ellipj(X3, k3)[1]
            signptheta = (-1)**m * np.sign(beta)
            ffac = 1 / 2 * np.real(r31 * r42)**(1/2)

            r = ((Bgl*r2 - Agl*r1) + (Bgl*r2+Agl*r1)*cnnum) / ((Bgl-Agl)+(Bgl+Agl)*cnnum)
            r[~Irmask] = np.nan
            rvec = np.nan_to_num(r)
            # rvecs.append(np.nan_to_num(r))
            rvecs.append(rvec)
            signprs.append(signpr)
            Irmasks.append(Irmask)
            if not axisymmetric:
                pass
    if case ==2:
        pass
    if case ==4:
        pass
    return rvecs, phivecs, Irmask, signprs





def ray_trace_all(mudists, fov_uas, MoDuas, varphi, inc, a, nmax, adap_fac = 1, axisymmetric=True, nmin=0):
    if np.isclose(a,0):
        a = 1e-6
    ns = range(nmin, nmax+1)
    rho = mudists/MoDuas
    zeros = np.zeros_like(rho)
    npix = len(zeros)
    xdim = int(np.sqrt(npix))
    rp = 1+np.sqrt(1-a**2)
    alpha = rho*np.cos(varphi)
    beta = rho*np.sin(varphi)
    lam, eta = get_lam_eta(alpha,beta, inc, a)
    up, um = get_up_um(lam, eta, a)
    # urat = up/um
    r1, r2, r3, r4 = get_radroots(np.complex128(lam), np.complex128(eta), a)
    rr1 = np.real(r1)
    rr2 = np.real(r2)
    rr3 = np.real(r3)
    rr4 = np.real(r4)
    ir1 = np.imag(r1)
    ir2 = np.imag(r2)
    ir3 = np.imag(r3)
    ir4 = np.imag(r4)
    ir1_0 = np.isclose(ir1,0)
    ir2_0 = np.isclose(ir2,0)
    ir3_0 = np.isclose(ir3,0)
    ir4_0 = np.isclose(ir4,0)
    c12 = (np.isclose(np.imag(r1),-np.imag(r2)))
    c34 = (np.isclose(np.imag(r3),-np.imag(r4)))

    allreal = ir1_0 * ir2_0 * ir3_0 * ir4_0

    case1 = allreal * (rr2<rp)*(rr3>rp)
    case2 = allreal * (rr4<rp)
    case3 = ir1_0 * ir2_0 * ~ir3_0 * ~ir3_0 * c34 * (rr2<rp)
    case4 = ~ir1_0 * ~ir2_0 * ~ir3_0 * ~ir4_0 * c12 * c34

    rvecs1, phivecs1, Irmasks1, signprs1 = ray_trace_by_case(a,alpha[case1],beta[case1],rr1[case1],rr2[case1],rr3[case1],rr4[case1],up[case1],um[case1],inc,nmax,1,adap_fac=adap_fac,axisymmetric=axisymmetric,nmin=nmin)
    rvecs2, phivecs2, Irmasks2, signprs2 = ray_trace_by_case(a,alpha[case2],beta[case2],rr1[case2],rr2[case2],rr3[case2],rr4[case2],up[case2],um[case2],inc,nmax,2,adap_fac=adap_fac,axisymmetric=axisymmetric,nmin=nmin)
    rvecs3, phivecs3, Irmasks3, signprs3 = ray_trace_by_case(a,alpha[case3],beta[case3],rr1[case3],rr2[case3],r3[case3],r4[case3],up[case3],um[case3],inc,nmax,3,adap_fac=adap_fac,axisymmetric=axisymmetric,nmin=nmin)
    rvecs4, phivecs4, Irmasks4, signprs4 = ray_trace_by_case(a,alpha[case4],beta[case4],r1[case4],r2[case4],r3[case4],r4[case4],up[case4],um[case4],inc,nmax,4,adap_fac=adap_fac,axisymmetric=axisymmetric,nmin=nmin)

    #stitch together cases
    all_rvecs = []
    all_phivecs = []
    all_Irmasks = []
    all_signprs = []
    for ni in range(len(ns)):#nmin, nmax+1):
        # n = ns[ni]
        r_all = np.ones_like(rho)
        phi_all = np.ones_like(rho)
        Irmask_all = np.ones_like(rho)
        signpr_all = np.ones_like(rho)
        r_all[case1]=rvecs1[ni]
        r_all[case2]=rvecs2[ni]
        r_all[case3]=rvecs3[ni]
        r_all[case4]=rvecs4[ni]
        all_rvecs.append(r_all)
        phi_all[case1]=phivecs1[ni]
        phi_all[case2]=phivecs2[ni]
        phi_all[case3]=phivecs3[ni]
        phi_all[case4]=phivecs4[ni]
        all_phivecs.append(phi_all)
        Irmask_all[case1]=Irmasks1[ni]
        Irmask_all[case2]=Irmasks2[ni]
        Irmask_all[case3]=Irmasks3[ni]
        Irmask_all[case4]=Irmasks4[ni]
        all_Irmasks.append(Irmask_all)
        signpr_all[case1]=signprs1[ni]
        signpr_all[case2]=signprs2[ni]
        signpr_all[case3]=signprs3[ni]
        signpr_all[case4]=signprs4[ni]
        all_signprs.append(signpr_all)

    #do adaptive raytracing; for each except the lowest n subimage,
    #ray trace higher ns at higher resolution around the Irmask.
    if adap_fac > 1 and nmax>nmin:
        for ni in range(1,len(ns)):
            n = ns[ni]
            Irmask = all_Irmasks[ni]
            Irmask = Irmask.reshape((xdim,xdim))
            Irmask = convolve2d(Irmask, kernel,mode='same')
            Irmask = rescale(Irmask, adap_fac, order=0)
            Irmask = np.array(Irmask,dtype=bool).flatten()

            submudists, subvarphi = get_rho_varphi_from_FOV_npix(fov_uas, adap_fac*xdim)
            submudists = submudists[Irmask]
            subvarphi = subvarphi[Irmask]
            # subrho = rescale(rho.reshape((xdim,xdim)),adap_fac,order=1).flatten()[Irmask]
            # subvarphi = varphi_grid_from_npix(adap_fac*xdim)[Irmask]
            # subvarphi = rescale(varphi.reshape((xdim,xdim)),adap_fac,order=1).flatten()[Irmask]
            sub_rvecs, sub_phivecs, sub_signprs = ray_trace_all(submudists, fov_uas, MoDuas, subvarphi, inc, a, n, axisymmetric=axisymmetric, nmin=n, adap_fac=1)
            # subrvec, subivec, subqvec, subuvec, subvvec, subredshift = raytrace_single_n(submudists, fov_uas, MoDuas, subvarphi, inc, a, n, nmin=n, adap_fac=1)
            
            rvec = np.zeros(adap_fac**2*xdim**2)
            phivec = np.zeros(adap_fac**2*xdim**2)
            signpr = np.zeros(adap_fac**2*xdim**2)
            rvec[Irmask]=sub_rvecs[0].flatten()
            signpr[Irmask]=sub_signprs[0].flatten()
            if not axisymmetric:
                phivec[Irmask]=sub_phivecs[0].flatten()
            all_rvecs[ni]=rvec
            all_phivecs[ni]=phivec
            all_signprs[ni]=signpr


    return all_rvecs, all_phivecs, all_signprs


#want to disentangle ray-tracing quantities (mudists, fov_uas, MoDuas, varphi, inc, a, nmax, adap_fac, axisymmetric)
#from fluid properties (boost, chi, fluid_eta, iota)

def kerr_exact(mudists, fov_uas, MoDuas, varphi, inc, a, nmax, boost, chi, fluid_eta, iota, adap_fac = 1, compute_V=False, axisymmetric=True):
    """
    Numerical: get rs from rho, varphi, inc, a, and subimage index n.
    """

    rho = mudists/MoDuas
    zeros = np.zeros_like(rho)
    npix = len(zeros)
    xdim = int(np.sqrt(npix))
    rp = 1+np.sqrt(1-a**2)
    alpha = rho*np.cos(varphi)
    beta = rho*np.sin(varphi)
    lam, eta = get_lam_eta(alpha,beta, inc, a)
    eta_mask = eta<0
    # print(eta_mask)
    up, um = get_up_um(lam, eta, a)
    urat = up/um
    r1, r2, r3, r4 = get_radroots(np.complex128(lam), np.complex128(eta), a)
    rr1 = np.real(r1)
    rr2 = np.real(r2)
    rr3 = np.real(r3)
    rr4 = np.real(r4)
    ir1 = np.imag(r1)
    ir2 = np.imag(r2)
    ir3 = np.imag(r3)
    ir4 = np.imag(r4)
    ir1_0 = np.isclose(ir1,0)
    ir2_0 = np.isclose(ir2,0)
    ir3_0 = np.isclose(ir3,0)
    ir4_0 = np.isclose(ir4,0)
    c12 = (np.isclose(np.imag(r1),-np.imag(r2)))
    c34 = (np.isclose(np.imag(r3),-np.imag(r4)))

    allreal = ir1_0 * ir2_0 * ir3_0 * ir4_0

    case1 = allreal * (rr2<rp)*(rr3>rp)
    case2 = allreal * (rr4<rp)
    case3 = ir1_0 * ir2_0 * ~ir3_0 * ~ir3_0 * c34 * (rr2<rp)
    case4 = ~ir1_0 * ~ir2_0 * ~ir3_0 * ~ir4_0 * c12 * c34


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
    bz = np.cos(iota)
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
    Ir_total[eta_mask] = np.nan


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
        Irmask = Ir<Ir_total

        if (adap_fac > 1) and n>0:
            Irmask = Irmask.reshape((xdim,xdim))
            Irmask = convolve2d(Irmask, kernel,mode='same')
            Irmask = rescale(Irmask, adap_fac, order=0)
            Irmask = np.array(Irmask,dtype=bool).flatten()

            subrho, subvarphi = get_rho_varphi_from_FOV_npix(fov_uas, adap_fac*xdim)
            subrho = subrho[Irmask] / MoDuas
            subvarphi = subvarphi[Irmask]
            # subrho = rescale(rho.reshape((xdim,xdim)),adap_fac,order=1).flatten()[Irmask]
            # subvarphi = varphi_grid_from_npix(adap_fac*xdim)[Irmask]
            # subvarphi = rescale(varphi.reshape((xdim,xdim)),adap_fac,order=1).flatten()[Irmask]
            subrvec, subivec, subqvec, subuvec, subvvec, subredshift = raytrace_single_n(subrho, subvarphi, inc, a, n, boost, chi, fluid_eta, iota, compute_V=compute_V)
            rvec = np.zeros(adap_fac**2*xdim**2)
            ivec = np.zeros(adap_fac**2*xdim**2)
            qvec = np.zeros(adap_fac**2*xdim**2)
            uvec = np.zeros(adap_fac**2*xdim**2)
            vvec = np.zeros(adap_fac**2*xdim**2)
            redshift = np.zeros(adap_fac**2*xdim**2)
            rvec[Irmask] = subrvec.flatten()
            ivec[Irmask] = subivec.flatten()
            qvec[Irmask] = subqvec.flatten()
            uvec[Irmask] = subuvec.flatten()
            if compute_V:
                vvec[Irmask] = subvvec.flatten()
            redshift[Irmask] = subredshift.flatten()
        else:
            signpr[~crit_mask] = np.sign(Ir_turn-Ir[~crit_mask])
            X3 = np.sqrt(Agl*Bgl)*(Ir[crit_mask] - signpr[crit_mask] * I3r)
            cnnum = ellipj(X3, k3)[1]
            signptheta = (-1)**m * np.sign(beta)
            ffac = 1 / 2 * np.real(r31 * r42)**(1/2)
            snnum = np.ones_like(rho)
            snnum[~crit_mask] = ellipj(((ffac*Ir)[~crit_mask]-fobs), k)[0]
            snsqr = snnum**2
            snsqr[eta_mask] = np.nan
            snsqr[np.abs(snsqr)>1.1]=np.nan
            
            r = np.real((r4*r31 - r3*r41*snsqr) / (r31-r41*snsqr))
            
            r[crit_mask] = ((Bgl*cr2 - Agl*cr1) + (Bgl*cr2+Agl*cr1)*cnnum) / ((Bgl-Agl)+(Bgl+Agl)*cnnum)
            r[eta_mask] =np.nan
            r[~Irmask] = np.nan
            rvec = np.nan_to_num(r)
            # rvecs.append(np.nan_to_num(r))
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
            if compute_V:
                vvec = np.dot(np.swapaxes(pspatialfluid,1,2), bvec).T[0]
            else:
                vvec = zeros
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
            # ivec = np.sqrt(qvec**2+uvec**2)
            # rpmask = np.abs(r-rp) < 0.01*rp
            # ivec[rpmask]=0.
            # qvec[rpmask]=0.
            # uvec[rpmask]=0.
            # vvec[rpmask]=0.
            # ivec = np.real(np.nan_to_num(ivec))
            qvec = np.real(np.nan_to_num(qvec))
            uvec = np.real(np.nan_to_num(uvec))
            if compute_V:
                vvec = np.real(np.nan_to_num(vvec))
            redshift = np.real(np.nan_to_num(redshift))
            # if adap_fac > 1 and nmax>0:
            #     rvec = rescale(rvec.reshape((xdim,xdim)),adap_fac,order=1).flatten()
            #     # ivec = rescale(ivec.reshape((xdim,xdim)),adap_fac,order=1).flatten()
            #     qvec = rescale(qvec.reshape((xdim,xdim)),adap_fac,order=1).flatten()
            #     uvec = rescale(uvec.reshape((xdim,xdim)),adap_fac,order=1).flatten()
                
            #     if compute_V:
            #         vvec = rescale(vvec.reshape((xdim,xdim)),adap_fac,order=1).flatten()
            #     else:
            #         vvec = np.zeros(adap_fac**2*xdim**2)
            #     redshift = rescale(redshift.reshape((xdim,xdim)),adap_fac,order=1).flatten()
            ivec = np.sqrt(qvec**2+uvec**2)
        rvecs.append(rvec)
        ivecs.append(ivec)
        qvecs.append(qvec)
        uvecs.append(uvec)
        vvecs.append(vvec)
        redshifts.append(redshift)

    return rvecs, ivecs, qvecs, uvecs, vvecs, redshifts


