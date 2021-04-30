import numpy as np
from numpy import exp, log, cos, sin



def ring_jfunc(r, phi, jargs):
    peak_r = jargs[0]
    thickness = jargs[1]
    return exp(-4.*log(2)*((r-peak_r)/thickness)**2)



def ring_with_gaussian_jfunc(r, phi, jargs):
	peak_r = jargs[0]
	thickness = jargs[1]
	ring_frac = jargs[2]
	gauss_r = jargs[3]
	gauss_phi = jargs[4]
	gauss_sigma = jargs[5]
	gauss_x = gauss_r * np.cos(gauss_phi)
	gauss_Y = gauss_r * np.sin(gauss_phi)
	ringim = ring_frac*exp(-4*log(2)*((r-peak_r)/thickness)**2)
	BLy = r*sin(phi)
	BLx = r*cos(phi)
	gaussim = (1-ring_frac)*1/(2*np.pi*gauss_sigma**2) * exp(-((BLx-gauss_x)**2+(BLy-gauss_y)**2)/(2*gauss_sigma**2))
	return ringim+gaussim


