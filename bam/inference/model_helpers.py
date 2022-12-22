import numpy as np
try:
    from collections.abc import Iterable
except ImportError: 
    from collections import Iterable

Gpercsq = 6.67e-11 / (3e8)**2
M87_ra = 12.513728717168174
M87_dec = 12.39112323919932
M87_dist = 16.8 * 3.086e22
M87_mass = 6.5e9 * 2e30
M87_inc = 17*np.pi/180
M87_PA = (288-360)*np.pi/180
M87_MoD = Gpercsq*M87_mass / M87_dist
RADPERUAS = np.pi/180/60/60/1e6
M87_MoDuas = M87_MoD/RADPERUAS

SgrA_mass = 4.1e6 * 2e30
SgrA_dist = 8*3.086e19
SgrA_MoD = Gpercsq*SgrA_mass / SgrA_dist
SgrA_MoDuas = SgrA_MoD/RADPERUAS

from skimage.transform import rescale, resize


def rescale_veclist(veclist,mode='edge',order=1,anti_aliasing=True):
    """
    Given a list of flattened arrays which are
    ordered by size, use the last array to rescale all
    the others.
    """
    ref = veclist[-1]
    xdim = int(np.sqrt(len(ref)))
    outlist = []
    for i in range(len(veclist)-1):
        subxdim = int(np.sqrt(len(veclist[i])))
        outlist.append(resize(veclist[i].reshape((subxdim,subxdim)), (xdim,xdim), mode=mode, order=order,anti_aliasing=anti_aliasing).flatten())
    outlist.append(ref)
    return outlist


def get_rho_varphi_from_FOV_npix(fov_uas, npix, adap_fac=1, nmax=0):
    """
    Given a field of view, npix, adap_fac, and nmax, compute a sequence of grids of rho and varphi for use in ray-tracing.
    """
    pxi = (np.arange(npix)+0.5)/npix-0.5
    pxj = (np.arange(npix)+0.5)/npix-0.5
    # get angles measured north of west
    PXI,PXJ = np.meshgrid(pxi,pxj)
    varphi = np.arctan2(-PXJ,PXI)# - np.pi/2
    varphi[varphi==0]=np.min(varphi[varphi>0])/10
    varphivec = varphi.flatten()
    
    # #get grid of angular radii in uas
    # mui = pxi*self.fov_uas
    # muj = pxj*self.fov_uas
    # MUI,MUJ = np.meshgrid(mui,muj)
    MUI = PXI*fov_uas
    MUJ = PXJ*fov_uas
    rho_uas = np.sqrt(np.power(MUI,2.)+np.power(MUJ,2.))
    rho_uas = rho_uas.flatten()
    if adap_fac == 1:
        if nmax == 0:
            return rho_uas, varphivec
        else:
            return [rho_uas for n in range(nmax+1)],[varphivec for n in range(nmax+1)]
    else:
        rhos = [rho_uas]
        varphis = [varphivec]
        for n in range(1, nmax+1):
            subr, subv = get_rho_varphi_from_FOV_npix(fov_uas, npix*adap_fac**n)
            rhos.append(subr)
            varphis.append(subv)
        return rhos, varphis

# def rho_grid_from_FOV_npix(fov_uas, npix):
#     mui = pxi*fov_uas
#     muj = pxj*self.fov_uas
#     MUI,MUJ = np.meshgrid(mui,muj)
#     MUDISTS = np.sqrt(np.power(MUI,2.)+np.power(MUJ,2.))
#     self.MUDISTS = MUDISTS.flatten()

# def varphi_grid_from_npix(npix):
#     pxi = (np.arange(npix)-0.01)/npix-0.5
#     pxj = np.arange(npix)/npix-0.5
#     # get angles measured north of west
#     PXI,PXJ = np.meshgrid(pxi,pxj)
#     varphi = np.arctan2(-PXJ,PXI)# - np.pi/2
#     varphi[varphi==0]=np.min(varphi[varphi>0])/10
#     return varphi.flatten()

def rho_conv(r, phi, inc):
    rhosq = r**2 * (1-np.sin(inc)**2*np.sin(phi)**2) + 2*r*(1+np.sin(inc)**2 * np.sin(phi)**2 + 2*np.sin(inc)*np.sin(phi))
    rho = np.sqrt(rhosq)
    return rho

def varphi_conv(phi, inc,mode):
    # if mode == 'model':
    #     atf = tt_arctan2
    # else:
    #     atf = np.arctan2
    # varphi= atf(np.sin(phi)*np.cos(inc), np.cos(phi))
    varphi = phi + np.tan(phi)/(2*(1+np.tan(phi)**2))*inc**2 + (5*np.tan(phi)-np.tan(phi)**3)/(24*(1+np.tan(phi)**2)**2)*inc**4
    return varphi

def emission_coordinates(rho, varphi, inc):
    phi = np.arctan2(np.sin(varphi),np.cos(varphi)*np.cos(inc))
    sinprod = np.sin(inc)*np.sin(phi)
    numerator = 1+rho**2 - (-3+rho**2)*sinprod+3*sinprod**2 + sinprod**3 
    denomenator = (-1+sinprod)**2 * (1+sinprod)
    sqq = np.sqrt(numerator/denomenator)
    r = (1-sqq + sinprod*(1+sqq))/(sinprod-1)
    return r, phi

def isiterable(object):
    '''
    Checks if an object is iterable.
    '''
    return isinstance(object, Iterable)

def quadsum(u, v):
    """ Returns the quadrature sum of arrays u and v.
    """
    return np.sqrt(u**2 + v**2)

def delta_t(t):
    """
    Given a time series, returns the time series shifted by the first time label, i.e. 1.5, 2.3 -> 0, 0.8
    """
    return t - np.min(t)


def quadratic(quad, lin, constant, t):
    """
    Returns the value of a quadratic expression evaluated over delta_t(t)
    """
    return 0.5 * quad * delta_t(t)**2 + lin * delta_t(t) + constant


def apply_diagonalization(model_data, times, T_matrices, quantity):
    """
    A meta-meta-model function for working with diagonalized closure quantities. Using a data product produced
    from telescope combinations as given, produces diagonalized quantities according to
    given T_matrices. Data product should be the output of either log_closure_amplitudes or closure_phases.
    """
    diagonalized_quantities = []
    for diagonal_index in range(len(times)):
        UTC_slice = times.loc[diagonal_index]
        UTC_mask = np.array(times.values == UTC_slice)
        model_subdata = model_data[UTC_mask]
        matrix = np.array(T_matrices[UTC_mask].values[0])
        diagonalized_quantities += np.dot(matrix, model_subdata)
    return np.array(diagonalized_quantities)

