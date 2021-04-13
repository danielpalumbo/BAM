import bam
from bam.inference.bam import Bam
from bam.inference.model_helpers import M87_mass, M87_dist, M87_inc
import ehtim as eh
import matplotlib.pyplot as plt
import numpy as np
import pymc3 as pm

def example_fixed_jfunc(r, phi, jargs):
    peak_r = jargs[0]
    thickness = jargs[1]
    return np.exp(-4.*np.log(2)*((r-peak_r)/thickness)**2)


fov = 80*eh.RADPERUAS
npix = 80
jfunc = example_fixed_jfunc
jarg_names = ['midr','thick']
jargs = [6., 5]
M = 6.2 #in billions of solar masses now
D = M87_dist
inc = M87_inc
zbl = 1.
PA = 0.
nmax = 1
thetabz = np.pi/2
chi = -np.pi/2
beta = 0.5
b = Bam(fov, npix, jfunc, jarg_names, jargs, M, D, inc, zbl, PA=PA, nmax = nmax, thetabz = thetabz, beta=beta, chi=chi)


im = b.make_image()
im.display()

for n in range(nmax+1):
	im = b.make_image(n=n)
	im.display()


# plt.imshow(b.ivec)#.reshape((80,80)))
modelb = Bam(fov, npix, jfunc, jarg_names, jargs, M, D, inc, [0.5, 1.5], PA=PA)
obs = eh.obsdata.load_uvfits('SR1_M87_2017_101_lo_hops_netcal_StokesI.uvfits')
obs.add_scans()
obs_sa = obs.avg_coherent(0., scan_avg=True)
to_fit = b.observe_same(obs_sa)


#let's fit!
modelb.setup(to_fit, data_types=['vis'], nlive=250)


modelb.run_nested()






