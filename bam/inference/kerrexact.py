"""
Implementation of the Kerr toy model for exact computation.

The phi computations are almost direct copies of Andrew Chael's kgeo.

Andrew is a co-author on the paper associated with this code.
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.special import ellipj, ellipk, ellipkinc, ellipeinc
from scipy.interpolate import griddata
from skimage.transform import rescale, resize
from scipy.signal import convolve2d
from bam.inference.model_helpers import get_rho_varphi_from_FOV_npix
from bam.inference.scipy_ellip_binding import ellip_pi_arr

minkmetric = np.diag([-1, 1, 1, 1])

kernel = np.ones((3,3))


def R1_R2(al,phi,j,ret_r2=True): #B62 and B65
    """
    Function by Andrew Chael to compute phi and t integral preliminaries.
    """
    al2 = al**2
    s2phi = np.sqrt(1-j*np.sin(phi)**2)
    p1 = np.sqrt((al2 -1)/(j+(1-j)*al2))
    f1 = 0.5*p1*np.log(np.abs((p1*s2phi+np.sin(phi))/(p1*s2phi-np.sin(phi))))
    nn = al2/(al2-1)
    R1 = (ellip_pi_arr(nn,phi,j) - al*f1)/(1-al2)

    if ret_r2:
        F = ellipkinc(phi,j)
        E = ellipeinc(phi,j)
        R2 = (F - (al2/(j+(1-j)*al2))*(E - al*np.sin(phi)*s2phi/(1+al*np.cos(phi)))) / (al2-1)
        R2 = R2 + (2*j - nn)*R1 / (j + (1-j)*al2)

    else:
        R2=np.NaN

    return (R1,R2)

np.seterr(invalid='ignore')
np.seterr(divide='ignore')
print("KerrBAM is silencing numpy warnings about invalid inputs and division by zero (default: warn, now ignore). To undo, call np.seterr(invalid='warn').")


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

def emissivity_model(rvecs, phivecs, signprs, signpthetas, alphas, betas, lams, etas, a, inc, boost, chi, fluid_eta, iota, compute_V=False):
    """
    Given the r and phi coordinates impacted by photons, evaluate the all-space (that is, pre-envelope) emissivity model for
    Q, U, and V there.
    """

    if fluid_eta is None:
        fluid_eta = chi+np.pi
    bz = np.cos(iota)
    beq = np.sqrt(1-bz**2)
    br = beq*np.cos(fluid_eta)
    bphi = beq*np.sin(fluid_eta)
    
    bvec = np.array([br, bphi, bz])
    ivecs = []
    qvecs = []
    uvecs = []
    vvecs = []
    redshifts = []
    for n in range(len(rvecs)):
        r = rvecs[n]
        phi = phivecs[n]
        signpr = signprs[n]
        signptheta = signpthetas[n]
        alpha = alphas[n]
        beta = betas[n]
        lam = lams[n]
        eta = etas[n]        
        npix = len(r)
        zeros = np.zeros_like(r)
        bigR = R(r, a, lam, eta)
        bigDelta = Delta(r, a)
        bigXi = Xi(r, a, np.pi/2)
        littleomega = omega(r, a, np.pi/2)

        #lowered
        pt_low = -1*np.ones_like(r)
        pr_low = signpr * np.sqrt(bigR)/bigDelta
        pr_low[pr_low>10] = 10
        pr_low[pr_low<-10] = -10
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
        prekappa1 = (pt * kfr - pr * kft) + spin * (pr * kfphi - pphi * kfr)
        prekappa2 = (rs**2 + spin**2) * (pphi * kftheta - ptheta * kfphi) - spin * (pt * kftheta - ptheta * kft)
        kappa1 = rs * prekappa1
        kappa2 = -rs * prekappa2
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
        qvec = np.real(np.nan_to_num(qvec))
        uvec = np.real(np.nan_to_num(uvec))
        if compute_V:
            vvec = np.real(np.nan_to_num(vvec))
        redshift = np.real(np.nan_to_num(redshift))
        ivec = np.sqrt(qvec**2+uvec**2)
        ivecs.append(ivec)
        qvecs.append(qvec)
        uvecs.append(uvec)
        vvecs.append(vvec)
        redshifts.append(redshift)
    return ivecs, qvecs, uvecs, vvecs, redshifts

#these should return r, phi, tau, tau_tot

r_o = 1000

def ray_trace_by_case(a, rm, rp, sb, lam, eta, r1, r2, r3, r4, up, um, inc, nmax, case, adap_fac= 1,axisymmetric = True, nmin=0):
    """
    Case 1: r1, r2, r3, r4 are real, r2<rp<r3.
    Case 2: r1, r2, r3, r4 are real and less than rp.
    Case 3: r1, r2 real, r3, r4 complex and conjugate.
    Case 4: All complex, r1,r2 conjugate and r3,r4 conjugate.
    """
    if len(sb) == 0:
        # print("No support in case "+str(case))
        return [[] for n in range(nmin, nmax+1)], [[] for n in range(nmin, nmax+1)], [[] for n in range(nmin, nmax+1)], [[] for n in range(nmin, nmax+1)]
    r21 = r2-r1
    r31 = r3-r1
    r32 = r3-r2
    r42 = r4-r2
    r41 = r4-r1
    r43 = r4-r3

    k = (r32*r41 / (r31*r42))
    urat = up/um
    Kurat = ellipk(urat)
    # print(nmin)
    #option 1
    m = sb.copy()
    m[m>0] = 0
    m+= nmin
    rvecs = []
    phivecs = []
    Irmasks = []
    signprs = []
    Fobs_arg = np.arcsin(np.cos(inc)/np.sqrt(up))
    Fobs = ellipkinc(Fobs_arg, urat)

    if not axisymmetric:
        a2um = a**2*um
        Gph_o = -1/np.sqrt(-a2um) * ellip_pi_arr(up, Fobs_arg, urat)
        # print(np.any(Gph_o<0))
        Gth_o = -1/np.sqrt(-a2um) * Fobs



    # sb = np.sign(beta)
    if case == 1:
        r3142sqrt = np.sqrt(r31*r42)
        if r_o == np.infty:
            x2ro = np.sqrt(r31/r41)
        else:
            x2ro = np.sqrt(r31*(r_o-r4)/(r41*(r_o-r3)))
        I2ro = 2/r3142sqrt * ellipkinc(np.arcsin(x2ro),k)
        Ir_turn = I2ro
        Ir_total = 2*Ir_turn

        # fobs = ellipkinc(np.arcsin(np.sqrt(r31/r41)), k)
        # #note: only outside the critical curve, since nothing inside has a turning point
        # Ir_turn = np.real(2/np.sqrt(r31*r42)*fobs)
        # Ir_total = 2*Ir_turn
    if case == 2:
        r3142sqrt = np.sqrt(r31*r42)
        if r_o == np.infty:
            x2ro = np.sqrt(r31/r41)
        else:
            x2ro = np.sqrt(r31*(r_o-r4)/(r41*(r_o-r3)))
        x2rp = np.sqrt((r31*(rp-r4))/(r41*(rp-r3)))
        I2rp = 2/r3142sqrt*ellipkinc(np.arcsin(x2rp),k)
        I2ro = 2/r3142sqrt*ellipkinc(np.arcsin(x2ro),k)#this is the previous fobs
        Ir_total = I2ro-I2rp      

    if case == 1 or case == 2:

        for n in range(nmin, nmax+1):
            m+= 1
            # print('m',m)
            Ir = 1/np.sqrt(-um*a**2)*(2*m*Kurat - sb*Fobs)
            if case == 1:
                signpr = np.sign(Ir_turn-Ir)
            else:
                signpr = np.ones_like(Ir)
            Irmask = Ir<Ir_total

            X2 = 1/2*r3142sqrt *(-Ir + I2ro)
            snnum, cnnum, dnnum, amnum = ellipj(X2,k)

            snsqr = snnum**2
            r =(r4*r31 - r3*r41*snsqr)/(r31-r41*snsqr)
            r[~Irmask] = np.nan
            rvec = np.nan_to_num(r)
            rvecs.append(rvec)
            signprs.append(signpr)
            Irmasks.append(Irmask)
            if not axisymmetric:      
                tau = Ir
                auxarg = np.arcsin(x2ro)

                rp3 = rp - r3
                rm3 = rm - r3
                rp4 = rp - r4
                rm4 = rm - r4

                dX2dtau = -0.5*r3142sqrt
                dsn2dtau = 2*snnum*cnnum*dnnum*dX2dtau
                drsdtau = -r31*r43*r41*dsn2dtau / ((r31-r41*snsqr)**2)
                Rpot_o = (r_o-r1)*(r_o-r2)*(r_o-r3)*(r_o-r4)
                drsdtau_o = np.sqrt(Rpot_o)
                H = drsdtau / (r - r3) - drsdtau_o/(r_o-r3)
                E = np.sqrt(r31*r42)*(ellipkinc(amnum,k) - ellipeinc(auxarg, k))
                Pi_1 = (2./np.sqrt(r31*r42))*(ellip_pi_arr(r41/r31,amnum,k)-ellip_pi_arr(r41/r31,auxarg,k))
                Pi_p = (2./np.sqrt(r31*r42))*(r43/(rp3*rp4))*(ellip_pi_arr((rp3*r41)/(rp4*r31),amnum,k)-
                                                                 ellip_pi_arr((rp3*r41)/(rp4*r31),auxarg,k))
                Pi_m = (2./np.sqrt(r31*r42))*(r43/(rm3*rm4))*(ellip_pi_arr((rm3*r41)/(rm4*r31),amnum,k)-
                                                                 ellip_pi_arr((rm3*r41)/(rm4*r31),auxarg,k))


                # final integrals
                I1 = r3*(-Ir) + r43*Pi_1 # B48
                I2 = H - 0.5*(r1*r4 + r2*r3)*(-tau) - E # B49
                Ip = Ir/rp3 - Pi_p # B50
                Im = Ir/rm3 - Pi_m # B50
                I_phi = (2*a/(rp-rm))*((rp - 0.5*a*lam)*Ip - (rm - 0.5*a*lam)*Im) # B1


                #finish Gph calculation
                snarg = np.sqrt(-a**2 * um)*(-tau+sb*Gth_o)

                snarg = snarg.astype(float)
                sinPhi_tau = np.zeros_like(snarg)
                Phi_tau = np.zeros_like(snarg)
                jmask = np.abs(snarg)<1e-12
                if np.any(jmask):
                    sinPhi_tau[jmask] = snarg[jmask]
                    Phi_tau[jmask] = snarg[jmask]
                if np.any(~jmask):
                    mk = (urat/(urat-1))[~jmask] # real, in (0,1) since k<0
                    # mk = np.outer(np.ones(1),mk)[~jmask]
                    ellipfuns = ellipj(snarg[~jmask]/np.sqrt(1-mk), mk)
                    #sn(sqrt(1-m)x | k) = sqrt(1-m)*sn(x|m)/dn(x|m)
                    sinPhi_tau[~jmask] = np.sqrt(1-mk) * ellipfuns[0]/ellipfuns[2]
                    #am(sqrt(1-m)x | k) = pi/2 - am(K(m) - x | m for m <=1
                    Phi_tau[~jmask] = 0.5*np.pi-ellipj(ellipk(mk) - snarg[~jmask]/np.sqrt(1-mk), mk)[3]

                Gph = (1/np.sqrt(-a2um)*ellip_pi_arr(up, Phi_tau, urat)-sb*Gph_o)#.astype(float)
                # print(Gph)
                # print('Gph_o',Gph_o)

                phi = I_phi + lam * Gph
                phi[~Irmask] = np.nan
                phivecs.append(np.nan_to_num(phi))

    if case == 3:
        Agl = np.real(np.sqrt(r32*r42))
        Bgl = np.real(np.sqrt(r31*r41))
        # Agl = np.sqrt(np.imag(r4)**2 + (np.real(r4)-r2)**2)
        # Bgl = np.sqrt(np.imag(r4)**2 + (np.real(r4)-r1)**2)
        k3 = ((Agl+Bgl)**2 - (r2-r1)**2)/(4*Agl*Bgl)
        rp1 = rp-r1
        rp2 = rp-r2
        x3rp = (Agl*rp1 - Bgl*rp2)/(Agl*rp1 + Bgl*rp2) # GL19a, B55
        if r_o == np.infty:
            x3ro = (Agl-Bgl)/(Agl+Bgl)
        else:
            ro1 = r_o - r1
            ro2 = r_o - r2
            Aro1 = Agl*ro1
            Bro2 = Bgl*ro2
            x3ro = (Aro1-Bro2)/(Aro1+Bro2)

        if not axisymmetric:
            alp = -1/x3rp
            rm1 = rm - r1
            rm2 = rm - r2
            x3rm = (Agl*rm1 - Bgl*rm2)/(Agl*rm1 + Bgl*rm2) # GL19a, B55
            alm = -1/x3rm
            al0 = -1/x3ro
        pref = 1/np.sqrt(Agl*Bgl)
        auxarg = np.arccos(x3ro)
        Ir_o = pref*ellipkinc(auxarg, k3)
        Ir_p = pref*ellipkinc(np.arccos(x3rp),k3)
        Ir_total = Ir_o - Ir_p
        # al0 = (Agl+Bgl)/(Bgl-Agl)
        # I3r_angle = np.arccos(1/al0)
        # I3r = ellipkinc(I3r_angle, k3) / np.sqrt(Agl*Bgl)
        # I3rp_angle = np.arccos((Agl*(rp-r1)-Bgl*(rp-r2))/(Agl*(rp-r1)+Bgl*(rp-r2)))
        # I3rp = ellipkinc(I3rp_angle, k3) / np.sqrt(Agl*Bgl)    
        

        # Ir_total = I3r - I3rp
        signpr = np.ones_like(Agl)
        for n in range(nmin, nmax+1):
            m += 1
            Ir = 1/np.sqrt(-um*a**2)*(2*m*Kurat - sb*Fobs)
            Irmask = Ir<Ir_total

            X3 = np.sqrt(Agl*Bgl)*(-Ir + Ir_o)

            snnum, cnnum, dnnum, amnum = ellipj(X3, k3)
            signptheta = (-1)**m * sb
            ffac = 1 / 2 * np.real(r31 * r42)**(1/2)

            r = ((Bgl*r2 - Agl*r1) + (Bgl*r2+Agl*r1)*cnnum) / ((Bgl-Agl)+(Bgl+Agl)*cnnum)
            r[~Irmask] = np.nan
            rvec = np.nan_to_num(r)
            # rvecs.append(np.nan_to_num(r))
            rvecs.append(rvec)
            signprs.append(signpr)
            Irmasks.append(Irmask)

            if not axisymmetric:
                # pass
                #TODO figure out conversion to Andrew's definitions
                tau = Ir
                #need:
                # al0
                amX3 = amnum
                # auxarg

                # # building blocks of the path integrals
                R1_a_0, R2_a_0 = R1_R2(al0,amX3,k3)
                R1_b_0, R2_b_0 = R1_R2(al0,auxarg,k3)
                R1_a_p, _ = R1_R2(alp,amX3,k3,ret_r2=False)
                R1_b_p, _ = R1_R2(alp,auxarg,k3,ret_r2=False)
                # if a>MINSPIN:
                R1_a_m, _ = R1_R2(alm,amX3,k3,ret_r2=False)
                R1_b_m, _ = R1_R2(alm,auxarg,k3,ret_r2=False)
                # else:
                #     R1_a_m = np.zeros(R1_a_p.shape)
                #     R1_b_m = np.zeros(R1_a_p.shape)

                Pi_1 = ((2*r21*np.sqrt(Agl*Bgl))/(Bgl**2-Agl**2)) * (R1_a_0 - R1_b_0) # B81
                Pi_2 = ((2*r21*np.sqrt(Agl*Bgl))/(Bgl**2-Agl**2))**2 * (R2_a_0 - R2_b_0) # B81
                Pi_p = ((2*r21*np.sqrt(Agl*Bgl))/(Bgl*rp2 - Agl*rp1))*(R1_a_p - R1_b_p) # B82
                Pi_m = ((2*r21*np.sqrt(Agl*Bgl))/(Bgl*rm2 - Agl*rm1))*(R1_a_m - R1_b_m) # B82


                # final integrals
                pref = ((Bgl*r2 + Agl*r1)/(Bgl+Agl))
                I1 = pref*(-tau) + Pi_1 # B78
                I2 = pref**2*(-tau) + 2*pref*Pi_1 + np.sqrt(Agl*Bgl)*Pi_2 # B79
                Ip = -((Bgl+Agl)*(-tau) + Pi_p) / (Bgl*rp2 + Agl*rp1) # B80
                Im = -((Bgl+Agl)*(-tau) + Pi_m) / (Bgl*rm2 + Agl*rm1) # B80
                I_phi = (2*a/(rp-rm))*((rp - 0.5*a*lam)*Ip - (rm - 0.5*a*lam)*Im) # B1

                #finish Gph calculation
                snarg = np.sqrt(-a**2 * um)*(-tau+sb*Gth_o)
                sinPhi_tau = np.zeros_like(snarg)
                Phi_tau = np.zeros_like(snarg)
                jmask = np.abs(snarg)<1e-12
                if np.any(jmask):
                    sinPhi_tau[jmask] = snarg[jmask]
                    Phi_tau[jmask] = snarg[jmask]
                if np.any(~jmask):
                    mk = (urat/(urat-1))[~jmask] # real, in (0,1) since k<0
                    # mk = np.outer(np.ones(snarg.shape[0]),mk)[~jmask]
                    ellipfuns = ellipj(snarg[~jmask]/np.sqrt(1-mk), mk)
                    #sn(sqrt(1-m)x | k) = sqrt(1-m)*sn(x|m)/dn(x|m)
                    sinPhi_tau[~jmask] = np.sqrt(1-mk) * ellipfuns[0]/ellipfuns[2]
                    #am(sqrt(1-m)x | k) = pi/2 - am(K(m) - x | m for m <=1
                    Phi_tau[~jmask] = 0.5*np.pi-ellipj(ellipk(mk) - snarg[~jmask]/np.sqrt(1-mk), mk)[3]

                Gph = (1/np.sqrt(-a2um)*ellip_pi_arr(up, Phi_tau, urat)-sb*Gph_o)



                phi = I_phi + lam * Gph
                phi[~Irmask] = np.nan
                phivecs.append(np.nan_to_num(phi))
    if case ==4:
        pass
    return rvecs, phivecs, Irmasks, signprs

def ray_trace_all(mudists, fov_uas, MoDuas, varphi, inc, a, nmax, adap_fac = 1, axisymmetric=True, nmin=0):
    if np.isclose(a,0):
        a = 1e-6
    ns = range(nmin, nmax+1)
    rho = mudists/MoDuas
    zeros = np.zeros_like(rho)
    npix = len(zeros)
    xdim = int(np.sqrt(npix))
    rp = 1+np.sqrt(1-a**2)
    rm = 1-np.sqrt(1-a**2)
    alpha = rho*np.cos(varphi)
    beta = rho*np.sin(varphi)
    lam, eta = get_lam_eta(alpha,beta, inc, a)
    up, um = get_up_um(lam, eta, a)
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
    # case4 = ~ir1_0 * ~ir2_0 * ~ir3_0 * ~ir4_0 * c12 * c34

    all_signpthetas = [np.ones_like(rho) for n in range(nmin,nmax+1)]
    sb = np.sign(beta)


    # plt.imshow((up/um).reshape((xdim,xdim)))
    # plt.colorbar()
    # plt.show()

    m = sb.copy()
    m[m>0] = 0
    m += nmin 
    for ni in range(len(ns)):
        m+=1
        all_signpthetas[ni] = (-1)**m*sb

    
    #for now, don't raytrace case 4
    rvecs1, phivecs1, Irmasks1, signprs1 = ray_trace_by_case(a,rm,rp,sb[case1],lam[case1],eta[case1],rr1[case1],rr2[case1],rr3[case1],rr4[case1],up[case1],um[case1],inc,nmax,1,adap_fac=adap_fac,axisymmetric=axisymmetric,nmin=nmin)
    rvecs2, phivecs2, Irmasks2, signprs2 = ray_trace_by_case(a,rm,rp,sb[case2],lam[case2],eta[case2],rr1[case2],rr2[case2],rr3[case2],rr4[case2],up[case2],um[case2],inc,nmax,2,adap_fac=adap_fac,axisymmetric=axisymmetric,nmin=nmin)
    rvecs3, phivecs3, Irmasks3, signprs3 = ray_trace_by_case(a,rm,rp,sb[case3],lam[case3],eta[case3],rr1[case3],rr2[case3],r3[case3],r4[case3],up[case3],um[case3],inc,nmax,3,adap_fac=adap_fac,axisymmetric=axisymmetric,nmin=nmin)
    # rvecs4, phivecs4, Irmasks4, signprs4 = ray_trace_by_case(a,rm,rp,sb[case4],lam[case4],eta[case4],r1[case4],r2[case4],r3[case4],r4[case4],up[case4],um[case4],inc,nmax,4,adap_fac=adap_fac,axisymmetric=axisymmetric,nmin=nmin)
   

    #stitch together cases
    all_rvecs = []
    all_phivecs = []
    all_Irmasks = []
    all_signprs = []
    for ni in range(len(ns)):#nmin, nmax+1):
        # n = ns[ni]
        r_all = np.zeros_like(rho)
        phi_all = np.zeros_like(rho)
        Irmask_all = np.ones_like(rho)
        signpr_all = np.ones_like(rho)
        r_all[case1]=rvecs1[ni]
        r_all[case2]=rvecs2[ni]
        r_all[case3]=rvecs3[ni]
        # r_all[case4]=rvecs4[ni]
        all_rvecs.append(r_all)
        if not axisymmetric:
            phi_all[case1]=phivecs1[ni]
            phi_all[case2]=phivecs2[ni]
            phi_all[case3]=phivecs3[ni]
        # phi_all[case4]=phivecs4[ni]
        all_phivecs.append(phi_all)
        Irmask_all[case1]=Irmasks1[ni]
        Irmask_all[case2]=Irmasks2[ni]
        Irmask_all[case3]=Irmasks3[ni]
        # Irmask_all[case4]=Irmasks4[ni]
        all_Irmasks.append(Irmask_all)
        signpr_all[case1]=signprs1[ni]
        signpr_all[case2]=signprs2[ni]
        signpr_all[case3]=signprs3[ni]
        # signpr_all[case4]=signprs4[ni]
        all_signprs.append(signpr_all)

    alphas = [alpha for n in ns]
    betas = [beta for n in ns]
    lams = [lam for n in ns]
    etas = [eta for n in ns]

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
            sub_rvecs, sub_phivecs, sub_signprs, sub_signpthetas, sub_alphas, sub_betas, sub_lams, sub_etas = ray_trace_all(submudists, fov_uas, MoDuas, subvarphi, inc, a, n, axisymmetric=axisymmetric, nmin=n, adap_fac=1)
            # subrvec, subivec, subqvec, subuvec, subvvec, subredshift = raytrace_single_n(submudists, fov_uas, MoDuas, subvarphi, inc, a, n, nmin=n, adap_fac=1)
            newsize = adap_fac**2*xdim**2
            rvec = np.zeros(newsize)
            phivec = np.zeros(newsize)
            signpr = np.zeros(newsize)
            signptheta = np.zeros(newsize)
            lamvec = np.zeros(newsize)
            etavec = np.zeros(newsize)
            alphavec = np.zeros(newsize)
            betavec = np.zeros(newsize)

            rvec[Irmask]=sub_rvecs[0].flatten()
            if not axisymmetric:
                phivec[Irmask]=sub_phivecs[0].flatten()
            signpr[Irmask]=sub_signprs[0].flatten()
            signptheta[Irmask]=sub_signpthetas[0].flatten()
            lamvec[Irmask]=sub_lams[0].flatten()
            etavec[Irmask]=sub_etas[0].flatten()
            alphavec[Irmask]=sub_alphas[0].flatten()
            betavec[Irmask]=sub_betas[0].flatten()


            rvec[rvec>20]=20
            # plt.imshow(rvec.reshape((adap_fac*xdim,adap_fac*xdim)))
            # plt.colorbar()
            # plt.show()
            # plt.imshow(lamvec.reshape((adap_fac*xdim,adap_fac*xdim)))
            # plt.colorbar()
            # plt.show()
            
            
            all_rvecs[ni]=rvec
            all_phivecs[ni]=phivec
            all_signprs[ni]=signpr
            all_signpthetas[ni]=signptheta
            lams[ni] = lamvec
            etas[ni] = etavec
            alphas[ni] = alphavec
            betas[ni] = betavec


    return all_rvecs, all_phivecs, all_signprs, all_signpthetas, alphas, betas, lams, etas

#want to disentangle ray-tracing quantities (mudists, fov_uas, MoDuas, varphi, inc, a, nmax, adap_fac, axisymmetric)
#from fluid properties (boost, chi, fluid_eta, iota)

def kerr_exact(mudists, fov_uas, MoDuas, varphi, inc, a, nmax, boost, chi, fluid_eta, iota, adap_fac = 1, compute_V=False, axisymmetric=True):
    """
    Numerical: get rs from rho, varphi, inc, a, and subimage index n.
    """
    rvecs, phivecs, signprs, signpthetas, alphas, betas, lams, etas = ray_trace_all(mudists, fov_uas, MoDuas, varphi, inc, a, nmax, adap_fac = adap_fac, axisymmetric=axisymmetric, nmin=0)
    ivecs, qvecs, uvecs, vvecs, redshifts = emissivity_model(rvecs, phivecs, signprs, signpthetas, alphas, betas, lams, etas, a, inc, boost, chi, fluid_eta, iota, compute_V=compute_V)

    return rvecs, phivecs, ivecs, qvecs, uvecs, vvecs, redshifts


