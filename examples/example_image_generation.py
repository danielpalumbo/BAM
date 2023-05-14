import numpy as np
import matplotlib.pyplot as plt
from bam.inference.model_helpers import M87_MoDuas
from bam.inference.jfuncs import double_power_law_jfunc as jf
from bam.inference.kerrbam import KerrBam
import ehtim as eh
from dynesty import utils as dyfunc


ntrue = 1
fov =80*eh.RADPERUAS
npix = 32
jfunc = jf
jarg_names = ['radius','inner_power','outer_power']
jargs = [4.5,5,5]
MoDuas = M87_MoDuas
inc = 17/180*np.pi
zbl = 0.6
PA = 288/180*np.pi
chi = -90/180*np.pi
eta = None
beta = 0.5
a = -0.5
iota = np.pi/3
adap_fac = 4
spec=1
az = 1

b = KerrBam(fov, npix, jfunc, jarg_names, jargs, MoDuas, a, inc, zbl, PA=PA,  
	chi=chi,eta=eta, nmax=ntrue, beta=beta, iota=iota, spec=spec,alpha_zeta = az,
	polflux=True, adap_fac=adap_fac, compute_P=True, interp_order=1)
im = b.make_rotated_image()

im.display(plotp=True)