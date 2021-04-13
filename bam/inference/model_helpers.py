import numpy as np
from collections.abc import Iterable
import pandas as pd
from theano.tensor.basic import arctan2 as tt_arctan2

Gpercsq = 6.67e-11 / (3e8)**2
M87_ra = 12.513728717168174
M87_dec = 12.39112323919932
M87_dist = 16.9 * 3.086e22
M87_mass = 6.2e9 * 2e30
M87_inc = 17*np.pi/180
M87_PA = (288-360)*np.pi/180


SgrA_mass = 4.1e6 * 2e30
SgrA_dist = 8*3.086e19

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



def amplitudes(model_func, theta, u, v, t=0, modeling=True):
    """
    This is a meta-model function. Takes a model function and parameters, and evaluates is
    at the u and v points, returning Stokes I amplitude.
    """
    real, imag = model_func(theta, u, v, t=t, modeling=True)
    amp = quadsum(real, imag)
    return amp


def log_closure_amplitudes(model_func, theta, u1, v1, u2, v2, u3, v3, u4, v4, t=0, modeling=True):
    """
    This is a meta-model function. Takes a model function and parameters, evaluates it
    at four arrays of uv points, and returns a log closure amplitude (array).

    A note on convention: for consistency with Chael et al 2018, baselines 1, 2, 3, and 4
    correspond to baselines 1-2, 3-4, 2-3, and 1-4 respectively.
    """
    real12, imag12 = model_func(theta, u1, v1, t=t, modeling=modeling)
    real34, imag34 = model_func(theta, u2, v2, t=t, modeling=modeling)
    real23, imag23 = model_func(theta, u3, v3, t=t, modeling=modeling)
    real14, imag14 = model_func(theta, u4, v4, t=t, modeling=modeling)
    amp12 = quadsum(real12, imag12)
    amp34 = quadsum(real34, imag34)
    amp23 = quadsum(real23, imag23)
    amp14 = quadsum(real14, imag14)
    logcamp = np.log(amp12) + np.log(amp34) - np.log(amp23) - np.log(amp14)
    return logcamp


def closure_phases(model_func, theta, u1, v1, u2, v2, u3, v3, t=0, modeling=True):
    """
    This is a meta-model function. Takes a model function and parameters, evaluates it
    at three arrays of uv points, and returns a closure phase (the phase of the bispectrum) array.
    """
    real12, imag12 = model_func(theta, u1, v1, t=t, modeling=modeling)
    real23, imag23 = model_func(theta, u2, v2, t=t, modeling=modeling)
    real31, imag31 = model_func(theta, u3, v3, t=t, modeling=modeling)
    if modeling:
        atan2 = tt_arctan2
    else:
        atan2 = np.arctan2
    phase12 = atan2(imag12, real12)
    phase23 = atan2(imag23, real23)
    phase31 = atan2(imag31, real31)
    closure_phase = phase12 + phase23 + phase31
    return closure_phase


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


def generate_prediction(model_func, theta, observation, quantity='amplitude', modeling=True):
    """
    Added a wrapper to simplify the process of fitting particular quantities
    by smartly selecting which products to unpack.
    """
    if quantity == 'amplitude':
        return amplitudes(model_func, theta, observation.u, observation.v,
                          t=observation.UTC, modeling=modeling)
    elif quantity == 'logcamp':
        return log_closure_amplitudes(model_func, theta, observation.u1,
                                      observation.v1, observation.u2, observation.v2,
                                      observation.u3, observation.v3, observation.u4,
                                      observation.v4, t=observation.UTC, modeling=modeling)
    elif quantity == 'cphase':
        return closure_phases(model_func, theta, observation.u1, observation.v1,
                              observation.u2, observation.v2, observation.u3, observation.v3,
                              t=observation.UTC, modeling=modeling)



def make_synthetic_data_like(model_func, theta, old_data):
    """
    Returns synthetic data with the same telescope and time stamps as old_data.
    Samples the function at the same Fourier coefficients and applies the same uncertainties.
    """
    return make_synthetic_data(model_func, theta, old_data['u'], old_data['v'], old_data['Isigma'], old_data['UTC'], old_data['t1'], old_data['t2'])


def unpack_baseline(data, t1, t2):
    """
    Unpacks all data for which the constituent telescopes match t1 and t2.
    Data does not contain conjugates, so check both
    """
    data1 = data[(data.t1.values == t1) * (data.t2.values == t2)]
    if len(data1) == 0:
        return data[(data.t1.values == t2) * (data.t2.values == t1)]
    return data1


def get_complex_Stokes_I(data):
    """
    Given data with Iamp and Iphase fields, returns Iamp * exp(i Iphase)
    """
    return data['Iamp'] * np.exp(1j * np.radians(data['Iphase']))


###################################################################################
# Helper functions for making synthetic data
###################################################################################


def extend_copy(element, length):
    """
    Returns a numpy array containing length copies of element
    """
    return np.array([element for i in range(length)])


def make_synthetic_data(model_func, theta, u, v, uncertainty, UTC=0, t1='?', t2='?'):
    """
    Returns data with uncertainties (assumed to be IID with the same variance uncertainty**2, where uncertainties are in Jy)

    If no time labels or telescope labels are provided, fills in 0 for UT time and '?' for telescope names

    To make data with time and telescope tags, use make_synthetic_data_like
    """

    # create arrays for unfilled parameters
    observation_number = len(u)
    if type(UTC) == int:
        UTC = extend_copy(UTC, observation_number)
    if type(t1) == str:
        t1 = extend_copy(t1, observation_number)
    if type(t2) == str:
        t2 = extend_copy(t2, observation_number)
    prediction_real, prediction_imag = model_func(
        theta, u, v, t=UTC, modeling=False)
    prediction = prediction_real + prediction_imag * 1j
    # corrupt amplitudes by random amount drawn from gaussian uncertainties
    amplitude = np.abs(np.random.normal(np.abs(prediction), uncertainty))
    phase = np.degrees(np.angle(prediction))
    data = pd.DataFrame({'u': u,
                         'v': v,
                         'Iamp': amplitude,
                         'Isigma': uncertainty,
                         'Iphase': phase,
                         'UTC': UTC,
                         't1': t1,
                         't2': t2})
    return data

# the debiasing methods are here to reproduce EHT processing output; we never use them directly except when generating synthetic data

def debias_Stokes_I_amplitude(data):
    """
    Measured interferometric visibility amplitudes are uniformly biased upwards by correlation of uncertainties
    See Thomson, Moran and Swanson for details. Returns a new copy of the data with modified amplitudes.
    """
    new_data = data.copy()
    debiased_squared_amplitude = np.abs(data['Iamp'])**2 - data['Isigma']**2
    negative_mask = debiased_squared_amplitude <= 0
    debiased_squared_amplitude[negative_mask] = data['Isigma'][negative_mask]**2
    new_data['Iamp'] = np.sqrt(debiased_squared_amplitude)
    return new_data


def debias_logcamp(logcamp, snr1, snr2, snr3, snr4):
    """
    Given the amplitude SNR on four baselines, debiases the log closure amplitude.
    """
    return logcamp + 0.5 * (1. / (snr1**2) + 1. / (snr2**2) - 1. / (snr3**2) - 1. / (snr4**2))


def get_Stokes_I_snr(data):
    """
    Given data with Iamp and Isigma fields, returns the SNR
    """
    return data['Iamp'] / data['Isigma']


def debiased_complex_I_snr_from_timeslice(data, t1, t2):
    """
    Given data that refer only to a single time slice and a pair of telescopes,
    unpack the given baseline, debias amplitudes, compute the SNR,
    and return the visibility and snr.
    """
    data12 = unpack_baseline(data, t1, t2)
    debiased12 = debias_Stokes_I_amplitude(data12)
    snr12 = get_Stokes_I_snr(debiased12).values[0]
    vis12 = get_complex_Stokes_I(debiased12).values[0]
    return (vis12, snr12)


def logcamp_error(snr1, snr2, snr3, snr4):
    """
    Returns the quadrature inverse sum of the snr on 4 baselines.
    """
    return np.sqrt(1. / (snr1**2) + 1. / (snr2**2) + 1. / (snr3**2) + 1. / (snr4**2))


def make_log_closure_amplitudes_like(vis_data, logcamp_data):
    """
    Given complex visibility data vis_data, creates closure quadrangles according to the telescope labels in log closure amplitude logcamp_data.
    """

    # given non-linear nature of the problem, use a for loop over the set of quadrangle timestamps
    # typical max index is less than 100 for scan-averaged data, and this is done only once per dataset
    log_closure_amplitudes = []
    logcamp_errors = []
    for quadrangle_index in range(len(logcamp_data)):
        sub_logcamp_data = logcamp_data.loc[quadrangle_index]
        UTC = sub_logcamp_data['UTC']
        # select the time slice in the visibility data, with some leeway for rounding
        sub_vis_data = vis_data[np.isclose(vis_data['UTC'], UTC)]
        # find the four constituent sites
        t1 = sub_logcamp_data['t1']
        t2 = sub_logcamp_data['t2']
        t3 = sub_logcamp_data['t3']
        t4 = sub_logcamp_data['t4']

        vis12, snr12 = debiased_complex_I_snr_from_timeslice(
            sub_vis_data, t1, t2)
        vis34, snr34 = debiased_complex_I_snr_from_timeslice(
            sub_vis_data, t3, t4)
        vis23, snr23 = debiased_complex_I_snr_from_timeslice(
            sub_vis_data, t2, t3)
        vis14, snr14 = debiased_complex_I_snr_from_timeslice(
            sub_vis_data, t1, t4)
        # produce closure amplitudes
        numerator = np.abs(vis12 * vis34)
        denominator = np.abs(vis23 * vis14)
        logcamp = np.log(numerator) - np.log(denominator)
        debiased_logcamp = debias_logcamp(logcamp, snr12, snr34, snr23, snr14)
        error = logcamp_error(snr12, snr34, snr23, snr14)
        log_closure_amplitudes.append(debiased_logcamp)
        logcamp_errors.append(error)
    # build a new dataframe starting with a copy of the old
    new_logcamp_data = logcamp_data.copy()
    new_logcamp_data['logcamp'] = np.array(log_closure_amplitudes)
    new_logcamp_data['logcamp_sigma'] = np.array(logcamp_errors)
    return new_logcamp_data
