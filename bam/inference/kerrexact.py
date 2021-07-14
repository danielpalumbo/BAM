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


def kerr_exact(rho, varphi, inc, a, nmax, boost, chi, thetabz, interp = True, interps = None):
    """
    Numerical: get rs from rho, varphi, inc, a, and subimage index n.
    """
    if interp:
        K_int, Fobs_int, fobs_outer_int, fobs_inner_ints, sn_outer_int, sn_inner_ints = interps
        fobs_inner_int_real, fobs_inner_int_imag = fobs_inner_ints
        sn_xk_int_real, sn_xk_int_imag, cndn_xk_int_real, cndn_xk_int_imag, sn_yk_int_real, sn_yk_int_imag, cndn_yk_int_real, cndn_yk_int_imag = sn_inner_ints

    zeros = np.zeros_like(rho)
    npix = len(zeros)
    rp = 1+np.sqrt(1-a**2)
    alpha = rho*np.cos(varphi)
    beta = rho*np.sin(varphi)
    lam, eta = get_lam_eta(alpha,beta, inc, a)
    lam = np.complex128(lam)
    eta = np.complex128(eta)
    up, um = get_up_um(lam, eta, a)
    urat = np.clip(np.real(up/um),-1, 1)
    # plt.imshow(urat.reshape((100,100)))
    # plt.colorbar()
    # plt.title('urat')
    # plt.show()
    r1, r2, r3, r4 = get_radroots(lam, eta, a)
    crit_mask = np.abs(np.imag(r3))>1e-14
    r31 = r3-r1
    r32 = r3-r2
    r42 = r4-r2
    r41 = r4-r1
    k = r32*r41 / (r31*r42)

    fluid_eta = chi+np.pi
    bz = np.cos(thetabz)
    beq = np.sqrt(1-bz**2)
    br = beq*np.cos(fluid_eta)
    bphi = beq*np.sin(fluid_eta)
    
    bvec = np.array([br, bphi, bz])

    if interp:

        r31_phase = np.angle(r31)
        delta321_phase = np.angle(r32) - np.angle(r31)
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
    redshifts = []
    # plowered = []
    # praised = []
    lam=np.real(lam)
    eta=np.real(eta)
    for n in range(nmax+1):
        m += 1
        if interp:
            Ir = np.real(1/np.sqrt(-um*a**2)*(2*m*np.complex128(K_int(urat)) - np.sign(beta)*Fobs))
        else:
            Ir = np.real(1/np.sqrt(-um*a**2)*(2*m*np.complex128(ef(np.pi/2, up/um)) - np.sign(beta)*Fobs))
        # plt.imshow((Ir-Ir_total).reshape((100,100)))
        # plt.colorbar()
        # plt.title('Ir')
        # plt.show()
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
        
        r = np.real((r4*r31 - r3*r41*snsqr) / (r31-r41*snsqr))
        r[eta<0] =np.nan
        r[Ir>Ir_total] = np.nan
        # plt.imshow(r.reshape((100,100)))
        # plt.colorbar()
        # plt.title('r')
        # plt.show()
        # rs.append(r)
        rvecs.append(np.nan_to_num(r))
        bigR = R(r, a, lam, eta)
        bigDelta = Delta(r, a)
        bigXi = Xi(r, a, inc)
        littleomega = omega(r, a, inc)
        # plt.imshow(bigR.reshape((100,100)))
        # plt.colorbar()
        # plt.title('R')
        # plt.show()
        # plt.imshow(bigDelta.reshape((100,100)))
        # plt.colorbar()
        # plt.title('Delta')
        # plt.show()
        # plt.imshow(bigXi.reshape((100,100)))
        # plt.colorbar()
        # plt.title('Xi')
        # plt.show()
        # plt.imshow(littleomega.reshape((100,100)))
        # plt.colorbar()
        # plt.title('omega')
        # plt.show()

        #lowered
        pt_low = -1*np.ones_like(r)
        pr_low = signpr * np.sqrt(bigR)/bigDelta
        # print(np.sum(np.isnan(pr_low)))
        pphi_low = lam
        ptheta_low = signptheta*np.sqrt(eta)
        plowers = np.array(np.hsplit(np.array([pt_low, pr_low, ptheta_low, pphi_low]),npix))
        # plowered.append([pt_low, pr_low, pphi_low, ptheta_low])

        #raised
        pt = 1/r**2 * (-a*(a-lam) + (r**2+a**2) * (r**2 + a**2 -a * lam) / bigDelta)
        pr = signpr * 1/r**2 * np.sqrt(bigR)
        pphi = 1/r**2 * (-(a-lam)+a/Delta(r,a)*(r**2+a**2 - a*lam))
        ptheta = signptheta*np.sqrt(eta) / r**2
        # praised.append([pt_up, pr_up, pphi_up, ptheta_up])
        # plt.imshow(np.real(pt.reshape((100,100))))
        # plt.colorbar()
        # plt.title('pt')
        # plt.show()
        # plt.imshow(np.real(pr.reshape((100,100))))
        # plt.colorbar()
        # plt.title('pr')
        # plt.show()
        # plt.imshow(np.real(ptheta.reshape((100,100))))
        # plt.colorbar()
        # plt.title('ptheta')
        # plt.show()
        # plt.imshow(np.real(pphi.reshape((100,100))))
        # plt.colorbar()
        # plt.title('pphi')
        # plt.show()


        #now everything to generate polarization
        
        emutetrad = np.array([[1/r*np.sqrt(bigXi/bigDelta), zeros, zeros, littleomega/r*np.sqrt(bigXi/bigDelta)], [zeros, np.sqrt(bigDelta)/r, zeros, zeros], [zeros, zeros, zeros, r/np.sqrt(bigXi)], [zeros, zeros, -1/r, zeros]])
        # print(emutetrad.shape)
        emutetrad = np.transpose(emutetrad,(2,0,1))
        boostmatrix = getlorentzboost(-boost, chi)
        # print(boostmatrix)
        # print(boostmatrix.shape)
        # print(np.matmul(minkmetric, boostmatrix))
        # print(np.matmul(minkmetric, boostmatrix).shape)
        # print(np.matmul())
        # #fluid frame tetrad
        coordtransform = np.matmul(np.matmul(minkmetric, boostmatrix), emutetrad)
        coordtransforminv = np.transpose(np.matmul(boostmatrix, emutetrad), (0,2, 1))
        # print('inv',coordtransforminv.shape)
        # coordtransform = np.matmul(getlorentzboost(boost, chi), emutetrad)
        # coordtransforminv = np.transpose(np.matmul(getlorentzboost(boost, chi), emutetrad))

        # print(coordtransform[0])
        # #lowered momenta at source
        # signpr = 1 if setman==True else getsignpr(b, spin, theta0, varphi, mbar)
        # plowers = np.array([-1, signpr * np.sqrt(RR)/d, np.sign(np.cos(theta0))*((-1)**(mbar+1))*np.sqrt(eta0), lam])

        rs = r
        # print('r',r.shape)
        
        #raised
        # pt = 1 / (rs**2) * (-spin * (spin - lam) + (rs**2 + spin**2) * (rs**2 + spin**2 - spin * lam) / d)
        # pr = signpr * np.sqrt(RR) / rs**2
        # ptheta = np.sign(np.cos(theta0))*((-1)**(mbar+1))*np.sqrt(eta0) / rs**2
        # pphi = 1/(rs**2) * (-(spin -lam) + (spin * (rs**2 + spin**2 - spin * lam)) / d)

        #fluid frame momenta
        # print('plowers',plowers.shape)
        # print(coordtransform.shape)
        # print(coordtransform)
        # print(plowers[0])
        # print(plowers.shape)
        # print('coordtransform',coordtransform.shape)
        # print(coordtransform.T[0])
        # print(plowers[0])
        # print(np.matmul(coordtransform.T[0],plowers[0]))
        # print(coordtransform[:,:,0])
        pupperfluid = np.matmul(coordtransform, plowers)
        # print(pupperfluid.shape)
        redshift = 1 / (pupperfluid[:,0,0])
        # plt.imshow(redshift.reshape((100,100)))
        # plt.colorbar()
        # plt.title('g')
        # plt.show()
        # print('redshift',redshift.shape)
        # print(pupperfluid[1:])
        # print(pupperfluid.shape)
        # plt.imshow(pupperfluid[:,0,0].reshape((100,100)))
        # plt.colorbar()
        # plt.title('kerr kFthat')
        # plt.show()
        # plt.imshow(pupperfluid[:,3,0].reshape((100,100)))
        # plt.colorbar()
        # plt.title('kerr kFzhat')
        # plt.show()
        lp = np.abs(pupperfluid[:,0,0]/pupperfluid[:,3,0])
        # plt.imshow(lp.reshape((100,100)))
        # plt.colorbar()
        # plt.title('lp')
        # plt.show()
        # print('lp',lp.shape)
        #fluid frame polarization
        # print(pupperfluid[1:])
        # print(bvec)
        pspatialfluid = pupperfluid[:,1:]
        # print(pspatialfluid.shape)
        # print(pspatialfluid)
        fupperfluid = np.cross(pspatialfluid, bvec, axisa = 1)
        # print(fupperfluid.shape)
        fupperfluid = np.insert(fupperfluid, 0, 0, axis=2)# / (np.linalg.norm(pupperfluid[1:]))
        # print(fupperfluid.shape)
        fupperfluid = np.swapaxes(fupperfluid, 1,2)
        # print(fupperfluid.shape)
        # print(coordtransforminv.shape)
        #apply the tetrad to get kerr f
        kfuppers = np.matmul(coordtransforminv, fupperfluid)
        # print("Just computed kfuppers")
        # print(kfuppers.shape)
        kft = kfuppers[:,0,0]
        kfr = kfuppers[:,1,0]
        kftheta = kfuppers[:,2,0]
        kfphi = kfuppers[:, 3,0]
        # print(kft.shape)
        # print(kfr.shape)
        # print(kftheta.shape)
        # print(kfphi.shape)

        spin = a
        #kappa1 and kappa2
        AA = (pt * kfr - pr * kft) + spin * (pr * kfphi - pphi * kfr)
        BB = (rs**2 + spin**2) * (pphi * kftheta - ptheta * kfphi) - spin * (pt * kftheta - ptheta * kft)
        kappa1 = rs * AA
        kappa2 = -rs * BB

        #screen appearance
        nu = -(alpha + spin * np.sin(inc))
        # print(nu.shape)

        ealpha = (beta * kappa2 - nu * kappa1) / (nu**2 + beta**2)
        ebeta = (beta * kappa1 + nu * kappa2) / (nu**2 + beta**2)
        # plt.imshow(ealpha.reshape((100,100)))
        # plt.colorbar()
        # plt.title('ealpha')
        # plt.show()
        # plt.imshow(ebeta.reshape((100,100)))
        # plt.colorbar()
        # plt.title('ebeta')
        # plt.show()
        # intensity = (ealpha**2 + ebeta**2) * lp
        # angle = np.arctan2(ebeta, ealpha)
        # qvec = intensity*np.cos(2*angle)
        # uvec = intensity*np.sin(2*angle)
        # ivec = intensity

        qvec = -(ealpha**2 - ebeta**2) * lp
        uvec = -2*ealpha*ebeta * lp
        ivec = np.sqrt(qvec**2+uvec**2)
        
        # plt.imshow(qvec.reshape((100,100)))
        # plt.colorbar()
        # plt.title('q')
        # plt.show()
        # plt.imshow(uvec.reshape((100,100)))
        # plt.colorbar()
        # plt.title('u')
        # plt.show()
        # plt.imshow(ivec.reshape((100,100)))
        # plt.colorbar()
        # plt.title('i')
        # plt.show()
        qvecs.append(np.real(np.nan_to_num(qvec)))
        uvecs.append(np.real(np.nan_to_num(uvec)))
        ivecs.append(np.real(np.nan_to_num(ivec)))
        redshifts.append(np.real(np.nan_to_num(redshift)))

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



def build_Fobs_interpolator(Fobs_angle, urat):
    FF, uu = np.meshgrid(Fobs_angle, urat)
    Fobs = np.real(ef(FF, uu))
    Fobs_int_base = interp2d(Fobs_angle, urat, Fobs)#, bounds_error=False, fill_value=0)
    Fobs_int = lambda x, y: si.dfitpack.bispeu(Fobs_int_base.tck[0], Fobs_int_base.tck[1], Fobs_int_base.tck[2], Fobs_int_base.tck[3], Fobs_int_base.tck[4], x, y)[0]
    return Fobs_int

def build_K_interpolator(urat):
    K = ef(np.pi/2, urat)
    return interp1d(urat, K)

def build_fobs_outer_interpolator(fobs_angle, k):
    ff, kk = np.meshgrid(fobs_angle, k)
    fobs = np.real(ef(ff, kk))
    fobs_int_base = interp2d(fobs_angle, k, fobs)#, bounds_error=False, fill_value=0)
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

def build_all_interpolators(ngrid=50):
    """
    For now, a bunch of magic numbers that give decent interpolators for reasonable FOVs and resolutions.
    """

    k = np.linspace(0,1,ngrid)
    fobs_angle = np.linspace(0, np.pi/2,ngrid)
    print("Building fobs outer interpolator...")
    fobs_outer_int = build_fobs_outer_interpolator(fobs_angle, k)
    
    r31_phase = np.linspace(-np.pi,np.pi,ngrid)
    delta321_phase = np.linspace(-np.pi,np.pi,ngrid)
    print("Building fobs inner interpolators...")
    fobs_inner_ints = build_fobs_inner_interpolators(r31_phase, delta321_phase)
    
    urat = np.linspace(-1,1,ngrid)
    Fobs_angle = np.linspace(0, np.pi/2,ngrid)
    print("Building Fobs interpolator...")

    Fobs_int = build_Fobs_interpolator(Fobs_angle, urat)
    
    print("Building K interpolator...")
    K_int = build_K_interpolator(urat)
    print("Building sn outer interpolator...")
    ffacIr_fobs_diff = np.linspace(-5,10,ngrid)
    sn_outer_int = build_sn_outer_interpolator(ffacIr_fobs_diff, k)
    print("Building sn inner interpolators...")
    A = np.linspace(0,3,ngrid)
    sn_inner_ints = build_sn_inner_interpolators(A, r31_phase, delta321_phase)
    # print("Built sn inner interpolators.")
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
