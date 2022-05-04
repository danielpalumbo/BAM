import numpy as np
from numpy import abs, sqrt, log, angle

def var_sys(var_a, var_b, var_c, var_u0, u):
    return var_a**2 * (u/var_u0)**var_c / (1+(u/var_u0)**(var_b+var_c))


def amp_debias(amp, sigma, force_nonzero=False):
    #amp and sigma must be real
    debsqr = amp**2
    mask = amp<sigma
    if force_nonzero:
        debsqr[amp<sigma] = sigma[mask]**2
    else:
        debsqr[amp<sigma] = 0
    debsqr[~mask] = amp[~mask]**2 - sigma[~mask]**2
    
    return sqrt(debsqr)


def logcamp_debias(log_camp, snr1, snr2, snr3, snr4):
    log_camp_debias = log_camp + 0.5*(1/snr1**2 + 1/snr2**2 - 1/snr3**2 -1/snr4**2)
    return log_camp_debias



"""
    Stealing this from eht-imaging, as long as
    we use the quadrangles from ehtim then
    the definitions of numerators and denominators
    will work out.
"""
def make_log_closure_amplitude(n1amp, n2amp, d1amp, d2amp, n1err, n2err, d1err, d2err, n1syserr, n2syserr, d1syserr, d2syserr, debias=True):

    if debias:
        p1 = amp_debias(n1amp, n1err, force_nonzero=True)
        p2 = amp_debias(n2amp, n2err, force_nonzero=True)
        p3 = amp_debias(d1amp, d1err, force_nonzero=True)
        p4 = amp_debias(d2amp, d2err, force_nonzero=True)
    else:
        p1 = n1amp
        p2 = n2amp
        p3 = d1amp
        p4 = d2amp
        
    snr1 = p1/n1err
    snr2 = p2/n2err
    snr3 = p3/d1err
    snr4 = p4/d2err
    
    ssnr1 = p1/np.sqrt(n1err**2+n1syserr**2)
    ssnr2 = p2/np.sqrt(n2err**2+n2syserr**2)
    ssnr3 = p3/np.sqrt(d1err**2+d1syserr**2)
    ssnr4 = p4/np.sqrt(d2err**2+d2syserr**2)


    logcamp = log(p1)+log(p2)-log(p3)-log(p4)
    logcamp_err = sqrt(1/ssnr1**2 + 1/ssnr2**2 + 1/ssnr3**2 + 1/ssnr4**2)
    if debias:
        logcamp = logcamp_debias(logcamp, snr1, snr2, snr3, snr4)
    
    return logcamp, logcamp_err


def amp_add_syserr(amp, amp_error, fractional=0, additive=0, var_a = 0, var_b=0, var_c=0, var_u0=4e9, u = 0):
    sigma = sqrt(amp_error**2+(fractional*amp)**2+additive**2 + var_sys(var_a, var_b, var_c, var_u0, u))
    return amp, sigma

def amp_get_syserr(amp, amp_error, fractional=0, additive=0, var_a = 0, var_b=0, var_c=0, var_u0=4e9, u = 0):
    sys_err = sqrt((fractional*amp)**2+additive**2 + var_sys(var_a, var_b, var_c, var_u0, u))

def vis_add_syserr(vis, amp_error, fractional=0, additive=0, var_a = 0, var_b=0, var_c=0, var_u0=4e9, u = 0):
    sigma = sqrt(amp_error**2+(fractional*np.abs(vis))**2+additive**2 + var_sys(var_a, var_b, var_c, var_u0, u))
    return vis, sigma


def logcamp_add_syserr(n1amp, n2amp, d1amp, d2amp, n1err, n2err, d1err, d2err, campd1, campd2, campd3, campd4, fractional=0, additive = 0, var_a = 0, var_b=0, var_c=0, var_u0=4e9, debias=True):
    n1syserr = amp_get_syserr(n1amp, n1err, fractional=fractional, additive=additive, var_a=var_a, var_b=var_b, var_c=var_c, var_u0=var_u0, u = campd1)
    n2syserr = amp_get_syserr(n2amp, n2err, fractional=fractional, additive=additive, var_a=var_a, var_b=var_b, var_c=var_c, var_u0=var_u0, u = campd2)
    d1syserr = amp_get_syserr(d1amp, d1err, fractional=fractional, additive=additive, var_a=var_a, var_b=var_b, var_c=var_c, var_u0=var_u0, u = campd3)
    d2syserr = amp_get_syserr(d2amp, d2err, fractional=fractional, additive=additive, var_a=var_a, var_b=var_b, var_c=var_c, var_u0=var_u0, u = campd4)
    return make_log_closure_amplitude(n1amp, n2amp, d1amp, d2amp, n1err, n2err, d1err, d2err, n1syserr, n2syserr, d1syserr, d2syserr, debias=debias)


def make_bispectrum(v1, v2, v3, v1err, v2err, v3err):
    bi = v1*v2*v3
    bisig = abs(bi)*sqrt((v1err/abs(v1))**2+(v2err/abs(v2))**2+(v3err/abs(v3))**2)
    return bi, bisig


def closure_phase_from_bispectrum(bi, bisig):
    cphase = angle(bi)
    cphase_error = bisig / abs(bi)
    return cphase, cphase_error


def bispectrum_add_syserr(v1, v2, v3, v1err, v2err, v3err, bisd1, bisd2, bisd3, fractional=0, additive=0, var_a = 0, var_b=0, var_c=0, var_u0=4e9):
    v1, v1err =vis_add_syserr(v1, v1err, fractional=fractional, additive=additive, var_a=var_a, var_b=var_b, var_c=var_c, var_u0=var_u0, u=bisd1)
    v2, v2err =vis_add_syserr(v2, v2err, fractional=fractional, additive=additive, var_a=var_a, var_b=var_b, var_c=var_c, var_u0=var_u0, u=bisd2)
    v3, v3err =vis_add_syserr(v3, v3err, fractional=fractional, additive=additive, var_a=var_a, var_b=var_b, var_c=var_c, var_u0=var_u0, u=bisd3)
    return make_bispectrum(v1, v2, v3, v1err, v2err, v3err)


def cphase_add_syserr(v1, v2, v3, v1err, v2err, v3err, cphased1, cphased2, cphased3, fractional=0, additive=0, var_a = 0, var_b=0, var_c=0, var_u0=4e9):
    bi, bisig = bispectrum_add_syserr(v1, v2, v3, v1err, v2err, v3err, cphased1, cphased2, cphased3, fractional=fractional, additive=additive, var_a=var_a, var_b=var_b, var_c=var_c, var_u0=var_u0)
    return closure_phase_from_bispectrum(bi, bisig)


def cphase_uvpairs(cphase_data):
    cphaseu1 = cphase_data['u1']
    cphaseu2 = cphase_data['u2']
    cphaseu3 = cphase_data['u3']
    cphasev1 = cphase_data['v1']
    cphasev2 = cphase_data['v2']
    cphasev3 = cphase_data['v3']
    cphaseuv1 = np.vstack([cphaseu1,cphasev1]).T
    cphaseuv2 = np.vstack([cphaseu2,cphasev2]).T
    cphaseuv3 = np.vstack([cphaseu3,cphasev3]).T
    return cphaseuv1, cphaseuv2, cphaseuv3

def cphase_uvdists(cphase_data):
    cphaseu1 = cphase_data['u1']
    cphaseu2 = cphase_data['u2']
    cphaseu3 = cphase_data['u3']
    cphasev1 = cphase_data['v1']
    cphasev2 = cphase_data['v2']
    cphasev3 = cphase_data['v3']
    cphased1 = np.sqrt(cphaseu1**2+cphasev1**2)
    cphased2 = np.sqrt(cphaseu2**2+cphasev2**2)
    cphased3 = np.sqrt(cphaseu3**2+cphasev3**2)
    return cphased1, cphased2, cphased3

def logcamp_uvpairs(logcamp_data):
    campu1 = logcamp_data['u1']
    campu2 = logcamp_data['u2']
    campu3 = logcamp_data['u3']
    campu4 = logcamp_data['u4']
    campv1 = logcamp_data['v1']
    campv2 = logcamp_data['v2']
    campv3 = logcamp_data['v3']
    campv4 = logcamp_data['v4']
    campuv1 = np.vstack([campu1,campv1]).T
    campuv2 = np.vstack([campu2,campv2]).T
    campuv3 = np.vstack([campu3,campv3]).T
    campuv4 = np.vstack([campu4,campv4]).T
    return campuv1, campuv2, campuv3, campuv4

def logcamp_uvdists(logcamp_data):
    campu1 = logcamp_data['u1']
    campu2 = logcamp_data['u2']
    campu3 = logcamp_data['u3']
    campu4 = logcamp_data['u4']
    campv1 = logcamp_data['v1']
    campv2 = logcamp_data['v2']
    campv3 = logcamp_data['v3']
    campv4 = logcamp_data['v4']
    campd1 = np.sqrt(campu1**2+campv1**2)
    campd2 = np.sqrt(campu2**2+campv2**2)
    campd3 = np.sqrt(campu3**2+campv3**2)
    campd4 = np.sqrt(campu4**2+campv4**2)
    return campd1, campd2, campd3, campd4

def get_camp_amp_sigma(obs, logcamp_data):
    data = obs.data
    lmask = np.array([data['time'][i] in logcamp_data['time'] for i in range(len(data['time']))])
    lcsd = data[lmask]
    # t1mask = lcsd['t1'] == t1

    #construct a qas_data object
    #for each quadrangle, it should have 
    #amp12, amp34, amp23, amp14, sigma12, sigma34, sigma23, sigma14

    qas_data = []
    for i in range(len(logcamp_data)):
        sub = logcamp_data[i]
        t1 = sub['t1']
        t2 = sub['t2']
        t3 = sub['t3']
        t4 = sub['t4']
        time = sub['time']
        t1mask = (lcsd['t1']==t1) + (lcsd['t2'] == t1)
        t2mask = (lcsd['t2']==t2) + (lcsd['t1'] == t2)
        t3mask = (lcsd['t1']==t3) + (lcsd['t2'] == t3)
        t4mask = (lcsd['t2']==t4) + (lcsd['t1'] == t4)
        timemask = lcsd['time']==time
        m12 = t1mask*t2mask*timemask
        m34 = t3mask*t4mask*timemask
        m23 = t2mask*t3mask*timemask
        m14 = t1mask*t4mask*timemask
        vis12 = lcsd[m12][0]
        vis34 = lcsd[m34][0]
        vis23 = lcsd[m23][0]
        vis14 = lcsd[m14][0]
        amp12 = np.abs(vis12['vis'])
        sig12 = vis12['sigma']
        amp34 = np.abs(vis34['vis'])
        sig34 = vis34['sigma']
        amp23 = np.abs(vis23['vis'])
        sig23 = vis23['sigma']
        amp14 = np.abs(vis14['vis'])
        sig14 = vis14['sigma']
        qas_data.append([amp12, amp34, amp23, amp14, sig12, sig34, sig23, sig14])
    qas_data = np.array(qas_data)
    return qas_data.T

def get_cphase_vis_sigma(obs, cphase_data):
    data = obs.data
    tas_data = []
    pmask = np.array([data['time'][i] in cphase_data['time'] for i in range(len(data['time']))])
    pcsd = data[pmask]

    for i in range(len(cphase_data)):
        sub = cphase_data[i]
        t1 = sub['t1']
        t2 = sub['t2']
        t3 = sub['t3']
        time = sub['time']
        t1mask = (pcsd['t1']==t1) + (pcsd['t2'] == t1)
        t2mask = (pcsd['t2']==t2) + (pcsd['t1'] == t2)
        t3mask = (pcsd['t1']==t3) + (pcsd['t2'] == t3)
        timemask = pcsd['time']==time
        m12 = t1mask*t2mask*timemask
        m23 = t2mask*t3mask*timemask
        m31 = t1mask*t3mask*timemask
        # m14 = t1mask*t4mask*timemask
        vis12 = pcsd[m12][0]
        vis23 = pcsd[m23][0]
        vis31 = pcsd[m31][0]
        # vis14 = lcsd[m14][0]
        v1 = vis12['vis']
        sig12 = vis12['sigma']
        v2 = vis23['vis']
        sig23 = vis23['sigma']
        v3 = vis31['vis']
        sig31 = vis31['sigma']
        tas_data.append([v1, v2, v3, sig12, sig23, sig31])
    tas_data = np.array(tas_data)
    return tas_data.T
