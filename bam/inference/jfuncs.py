import numpy as np
from numpy import exp, log, cos, sin
import matplotlib.pyplot as plt


def ring_jfunc(r, jargs):
    peak_r = jargs[0]
    thickness = jargs[1]
    return exp(-4.*log(2)*((r-peak_r)/thickness)**2)

def power_law_jfunc(r, jargs):
    alpha = jargs[0]
    out = r**(-alpha)
    out[np.isinf(out)]=0
    return out

def ring_plus_power_law_jfunc(r, jargs):
    peak_r = jargs[0]
    thickness = jargs[1]
    alpha = jargs[2]
    ring_frac = jargs[3]
    return ring_frac*ring_jfunc(r,[peak_r,thickness]) + (1-ring_frac)*power_law_jfunc(r,[alpha])

def double_power_law_jfunc(r, jargs):
    r0, a, b = jargs
    return (r/r0)**a / (1+(r/r0)**(a+b))



#temporarily removed with recent switch to axisymmetry

# def ring_with_gaussian_jfunc(r, phi, jargs):
# 	peak_r = jargs[0]
# 	thickness = jargs[1]
# 	ring_frac = jargs[2]
# 	gauss_r = jargs[3]
# 	gauss_phi = jargs[4]
# 	gauss_sigma = jargs[5]
# 	gauss_x = gauss_r * np.cos(gauss_phi)
# 	gauss_y = gauss_r * np.sin(gauss_phi)
# 	ring_im = exp(-4*log(2)*((r-peak_r)/thickness)**2)
# 	ring_em = np.sum(ring_im)
# 	BLy = r*sin(phi)
# 	BLx = r*cos(phi)
# 	gauss_im = 1/(2*np.pi*gauss_sigma**2) * exp(-((BLx-gauss_x)**2+(BLy-gauss_y)**2)/(2*gauss_sigma**2))
# 	gauss_em = np.sum(gauss_im)
# 	ring_im *= ring_frac/ring_em
# 	gauss_im *= (1-ring_frac)/gauss_em
# 	# print(np.sum(ring_im))
# 	# print(np.sum(gauss_im))
# 	# print(np.min(ring_im))
# 	# print(np.min(gauss_im))
# 	return ring_frac*ring_im/ring_em + (1-ring_frac)*gauss_im/gauss_em


