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
    return sys_err
    
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

def get_minimal_cphases(obs):
    """
    This method is adapted from code written by Dom Pesce (see also Blackburn et al, 2021)
    to find the minimal set of closure quantities at each time in an observation.
    """
    print("Working on getting minimal cphases...")
    fileout='cphases.txt'
    obs.reorder_tarr_snr()
    obs.add_cphase(count='max')

    # organize some info
    time_vis = obs.data['time']
    time_cp = obs.cphase['time']
    t1_vis = obs.data['t1']
    t2_vis = obs.data['t2']

    # Determine the number of timestamps containing a closure triangle
    timestamps_cp = np.unique(time_cp)
    N_times_cp = len(timestamps_cp)

    # loop over all timestamps
    obs_cphase_arr = []
    for i in np.arange(0,N_times_cp,1):

        # get the current timestamp
        ind_here_cp = (time_cp == timestamps_cp[i])
        time_here = time_cp[ind_here_cp]

        # copy the cphase table for this timestamp
        obs_cphase_orig = np.copy(obs.cphase[ind_here_cp])

        # sort by cphase SNR
        snr = 1.0 / ((np.pi/180.0)*obs_cphase_orig['sigmacp'])
        ind_snr = np.argsort(snr)
        obs_cphase_orig = obs_cphase_orig[ind_snr]
        snr = snr[ind_snr]

        # organize the closure phase stations
        cp_ant1_vec = obs_cphase_orig['t1']
        cp_ant2_vec = obs_cphase_orig['t2']
        cp_ant3_vec = obs_cphase_orig['t3']

        # get the number of time-matched baselines
        ind_here_bl = (time_vis == timestamps_cp[i])
        B_here = ind_here_bl.sum()

        # organize the time-matched baseline stations
        bl_ant1_vec = t1_vis[ind_here_bl]
        bl_ant2_vec = t2_vis[ind_here_bl]

        # initialize the design matrix
        design_mat = np.zeros((ind_here_cp.sum(),B_here))

        # fill in each row of the design matrix
        for ii in range(ind_here_cp.sum()):

            # determine which stations are in this triangle
            ant1_here = cp_ant1_vec[ii]
            ant2_here = cp_ant2_vec[ii]
            ant3_here = cp_ant3_vec[ii]
            
            # matrix entry for first leg of triangle
            ind1_here = ((bl_ant1_vec == ant1_here) & (bl_ant2_vec == ant2_here))
            if ind1_here.sum() == 0.0:
                ind1_here = ((bl_ant1_vec == ant2_here) & (bl_ant2_vec == ant1_here))
                val1_here = -1.0
            else:
                val1_here = 1.0
            design_mat[ii,ind1_here] = val1_here
            
            # matrix entry for second leg of triangle
            ind2_here = ((bl_ant1_vec == ant2_here) & (bl_ant2_vec == ant3_here))
            if ind2_here.sum() == 0.0:
                ind2_here = ((bl_ant1_vec == ant3_here) & (bl_ant2_vec == ant2_here))
                val2_here = -1.0
            else:
                val2_here = 1.0
            design_mat[ii,ind2_here] = val2_here
            
            # matrix entry for third leg of triangle
            ind3_here = ((bl_ant1_vec == ant3_here) & (bl_ant2_vec == ant1_here))
            if ind3_here.sum() == 0.0:
                ind3_here = ((bl_ant1_vec == ant1_here) & (bl_ant2_vec == ant3_here))
                val3_here = -1.0
            else:
                val3_here = 1.0
            design_mat[ii,ind3_here] = val3_here

        # determine the expected size of the minimal set
        N_min = np.linalg.matrix_rank(design_mat)

        # print some info
        # print('For timestamp '+str(timestamps_cp[i])+':')

        # get the current stations
        stations_here = np.unique(np.concatenate((cp_ant1_vec,cp_ant2_vec,cp_ant3_vec)))
        # print('Observing stations are '+str([str(station) for station in stations_here]))

        # print('Size of maximal set of closure phases = '+str(ind_here_cp.sum())+'.')
        # print('Size of minimal set of closure phases = '+str(N_min)+'.')
        # print('...')

        ##########################################################
        # start of loop to recover minimal set
        ##########################################################

        # make a mask to keep track of which cphases will stick around
        keep = np.ones(len(obs_cphase_orig),dtype=bool)
        obs_cphase = obs_cphase_orig[keep]

        # remember the original minimal set size
        N_min_orig = N_min

        # initialize the loop breaker
        good_enough = False

        # perform the loop
        count = 0
        ind_list = []
        while good_enough == False:

            # recreate the mask each time
            keep = np.ones(len(obs_cphase_orig),dtype=bool)
            keep[ind_list] = False
            obs_cphase = obs_cphase_orig[keep]

            # organize the closure phase stations
            cp_ant1_vec = obs_cphase['t1']
            cp_ant2_vec = obs_cphase['t2']
            cp_ant3_vec = obs_cphase['t3']

            # get the number of time-matched baselines
            ind_here_bl = (time_vis == timestamps_cp[i])
            B_here = ind_here_bl.sum()

            # organize the time-matched baseline stations
            bl_ant1_vec = t1_vis[ind_here_bl]
            bl_ant2_vec = t2_vis[ind_here_bl]

            # initialize the design matrix
            design_mat = np.zeros((keep.sum(),B_here))

            # fill in each row of the design matrix
            for ii in range(keep.sum()):

                # determine which stations are in this triangle
                ant1_here = cp_ant1_vec[ii]
                ant2_here = cp_ant2_vec[ii]
                ant3_here = cp_ant3_vec[ii]
                
                # matrix entry for first leg of triangle
                ind1_here = ((bl_ant1_vec == ant1_here) & (bl_ant2_vec == ant2_here))
                if ind1_here.sum() == 0.0:
                    ind1_here = ((bl_ant1_vec == ant2_here) & (bl_ant2_vec == ant1_here))
                    val1_here = -1.0
                else:
                    val1_here = 1.0
                design_mat[ii,ind1_here] = val1_here
                
                # matrix entry for second leg of triangle
                ind2_here = ((bl_ant1_vec == ant2_here) & (bl_ant2_vec == ant3_here))
                if ind2_here.sum() == 0.0:
                    ind2_here = ((bl_ant1_vec == ant3_here) & (bl_ant2_vec == ant2_here))
                    val2_here = -1.0
                else:
                    val2_here = 1.0
                design_mat[ii,ind2_here] = val2_here
                
                # matrix entry for third leg of triangle
                ind3_here = ((bl_ant1_vec == ant3_here) & (bl_ant2_vec == ant1_here))
                if ind3_here.sum() == 0.0:
                    ind3_here = ((bl_ant1_vec == ant1_here) & (bl_ant2_vec == ant3_here))
                    val3_here = -1.0
                else:
                    val3_here = 1.0
                design_mat[ii,ind3_here] = val3_here

            # determine the size of the minimal set
            N_min = np.linalg.matrix_rank(design_mat)

            if (keep.sum() == N_min_orig) & (N_min == N_min_orig):
                good_enough = True
            else:
                if N_min == N_min_orig:
                    ind_list.append(count)
                else:
                    ind_list = ind_list[:-1]
                    count -= 1
                count += 1

            if count > len(obs_cphase_orig):
                break

        # print out the size of the recovered set for double-checking
        obs_cphase = obs_cphase_orig[keep]
        if len(obs_cphase) != N_min:
            print('*****************WARNING: minimal set not found*****************')
        else:
            print('Size of recovered minimal set = '+str(len(obs_cphase))+'.')
        print('========================================================================')

        obs_cphase_arr.append(obs_cphase)

    # save an output cphase file
    obs_cphase_arr = np.concatenate(obs_cphase_arr)
    np.savetxt(fileout,obs_cphase_arr,fmt='%26.26s')
    print("Saved cphases to:"+str(fileout))
    return obs_cphase_arr
    # np.savetxt(fileout,obs_cphase_arr,fmt='%26.26s')


def get_minimal_logcamps(obs):
    """
    This method is adapted from code written by Dom Pesce (see also Blackburn et al, 2021)
    to find the minimal set of closure quantities at each time in an observation.
    """
    print("Working on getting minimal logcamps...")
    fileout='logcamps.txt'
    # compute a maximum set of log closure amplitudes
    obs.reorder_tarr_snr()
    obs.add_logcamp(count='max')

    # organize some info
    time_vis = obs.data['time']
    time_lca = obs.logcamp['time']
    t1_vis = obs.data['t1']
    t2_vis = obs.data['t2']

    # Determine the number of timestamps containing a quadrangle
    timestamps_lca = np.unique(time_lca)
    N_times_lca = len(timestamps_lca)

    # loop over all timestamps
    obs_lca_arr = []
    for i in np.arange(0,N_times_lca,1):

        # get the current timestamp
        ind_here_lca = (time_lca == timestamps_lca[i])
        time_here = time_lca[ind_here_lca]

        # copy the logcamp table for this timestamp
        obs_lca_orig = np.copy(obs.logcamp[ind_here_lca])

        # sort by logcamp SNR
        snr = 1.0 / obs_lca_orig['sigmaca']
        ind_snr = np.argsort(snr)
        obs_lca_orig = obs_lca_orig[ind_snr]
        snr = snr[ind_snr]

        # organize the quadrangle stations
        lca_ant1_vec = obs_lca_orig['t1']
        lca_ant2_vec = obs_lca_orig['t2']
        lca_ant3_vec = obs_lca_orig['t3']
        lca_ant4_vec = obs_lca_orig['t4']

        # get the number of time-matched baselines
        ind_here_bl = (time_vis == timestamps_lca[i])
        B_here = ind_here_bl.sum()
        
        # organize the time-matched baseline stations
        bl_ant1_vec = t1_vis[ind_here_bl]
        bl_ant2_vec = t2_vis[ind_here_bl]
        
        # initialize the design matrix
        design_mat = np.zeros((ind_here_lca.sum(),B_here))

        # fill in each row of the design matrix
        for ii in range(ind_here_lca.sum()):

            # determine which stations are in this quadrangle
            ant1_here = lca_ant1_vec[ii]
            ant2_here = lca_ant2_vec[ii]
            ant3_here = lca_ant3_vec[ii]
            ant4_here = lca_ant4_vec[ii]
            
            # matrix entry for first leg of quadrangle
            ind1_here = ((bl_ant1_vec == ant1_here) & (bl_ant2_vec == ant2_here)) | ((bl_ant1_vec == ant2_here) & (bl_ant2_vec == ant1_here))
            design_mat[ii,ind1_here] = 1.0

            # matrix entry for second leg of quadrangle
            ind2_here = ((bl_ant1_vec == ant3_here) & (bl_ant2_vec == ant4_here)) | ((bl_ant1_vec == ant4_here) & (bl_ant2_vec == ant3_here))
            design_mat[ii,ind2_here] = 1.0

            # matrix entry for third leg of quadrangle
            ind3_here = ((bl_ant1_vec == ant1_here) & (bl_ant2_vec == ant4_here)) | ((bl_ant1_vec == ant4_here) & (bl_ant2_vec == ant1_here))
            design_mat[ii,ind3_here] = -1.0

            # matrix entry for fourth leg of quadrangle
            ind4_here = ((bl_ant1_vec == ant2_here) & (bl_ant2_vec == ant3_here)) | ((bl_ant1_vec == ant3_here) & (bl_ant2_vec == ant2_here))
            design_mat[ii,ind4_here] = -1.0

        # determine the expected size of the minimal set
        N_min = np.linalg.matrix_rank(design_mat)

        # print some info
        # print('For timestamp '+str(timestamps_lca[i])+':')

        # get the current stations
        stations_here = np.unique(np.concatenate((lca_ant1_vec,lca_ant2_vec,lca_ant3_vec,lca_ant4_vec)))
        # print('Observing stations are '+str([str(station) for station in stations_here]))

        # print('Size of maximal set of closure amplitudes = '+str(ind_here_lca.sum())+'.')
        # print('Size of minimal set of closure amplitudes = '+str(N_min)+'.')
        # print('...')

        ##########################################################
        # start of loop to recover minimal set
        ##########################################################

        # make a mask to keep track of which cphases will stick around
        keep = np.ones(len(obs_lca_orig),dtype=bool)
        obs_lca = obs_lca_orig[keep]

        # remember the original minimal set size
        N_min_orig = N_min

        # initialize the loop breaker
        good_enough = False

        # perform the loop
        count = 0
        ind_list = []
        while good_enough == False:

            # recreate the mask each time
            keep = np.ones(len(obs_lca_orig),dtype=bool)
            keep[ind_list] = False
            obs_lca = obs_lca_orig[keep]

            # organize the quadrangle stations
            lca_ant1_vec = obs_lca['t1']
            lca_ant2_vec = obs_lca['t2']
            lca_ant3_vec = obs_lca['t3']
            lca_ant4_vec = obs_lca['t4']

            # get the number of time-matched baselines
            ind_here_bl = (time_vis == timestamps_lca[i])
            B_here = ind_here_bl.sum()
            
            # organize the time-matched baseline stations
            bl_ant1_vec = t1_vis[ind_here_bl]
            bl_ant2_vec = t2_vis[ind_here_bl]
            
            # initialize the design matrix
            design_mat = np.zeros((keep.sum(),B_here))

            # fill in each row of the design matrix
            for ii in range(keep.sum()):

                # determine which stations are in this quadrangle
                ant1_here = lca_ant1_vec[ii]
                ant2_here = lca_ant2_vec[ii]
                ant3_here = lca_ant3_vec[ii]
                ant4_here = lca_ant4_vec[ii]
                
                # matrix entry for first leg of quadrangle
                ind1_here = ((bl_ant1_vec == ant1_here) & (bl_ant2_vec == ant2_here)) | ((bl_ant1_vec == ant2_here) & (bl_ant2_vec == ant1_here))
                design_mat[ii,ind1_here] = 1.0

                # matrix entry for second leg of quadrangle
                ind2_here = ((bl_ant1_vec == ant3_here) & (bl_ant2_vec == ant4_here)) | ((bl_ant1_vec == ant4_here) & (bl_ant2_vec == ant3_here))
                design_mat[ii,ind2_here] = 1.0

                # matrix entry for third leg of quadrangle
                ind3_here = ((bl_ant1_vec == ant1_here) & (bl_ant2_vec == ant4_here)) | ((bl_ant1_vec == ant4_here) & (bl_ant2_vec == ant1_here))
                design_mat[ii,ind3_here] = -1.0

                # matrix entry for fourth leg of quadrangle
                ind4_here = ((bl_ant1_vec == ant2_here) & (bl_ant2_vec == ant3_here)) | ((bl_ant1_vec == ant3_here) & (bl_ant2_vec == ant2_here))
                design_mat[ii,ind4_here] = -1.0

            # determine the size of the minimal set
            N_min = np.linalg.matrix_rank(design_mat)

            if (keep.sum() == N_min_orig) & (N_min == N_min_orig):
                good_enough = True
            else:
                if N_min == N_min_orig:
                    ind_list.append(count)
                else:
                    ind_list = ind_list[:-1]
                    count -= 1
                count += 1

            if count > len(obs_lca_orig):
                break

        # print out the size of the recovered set for double-checking
        obs_lca = obs_lca_orig[keep]
        if len(obs_lca) != N_min:
            print('*****************WARNING: minimal set not found*****************')
        # else:
        #     print('Size of recovered minimal set = '+str(len(obs_lca))+'.')
        # print('========================================================================')
        
        obs_lca_arr.append(obs_lca)

    # save an output logcamp file
    obs_lca_arr = np.concatenate(obs_lca_arr)
    np.savetxt(fileout,obs_lca_arr,fmt='%26.26s')
    print("Saved logcamps to:"+str(fileout))
    return obs_lca_arr