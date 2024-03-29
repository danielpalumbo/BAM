import numpy as np
import matplotlib.pyplot as plt
from bam.inference.kerrbam import KerrBam
import ehtim as eh


#this example emissivity envelope could also have been imported with from bam.inference.jfuncs import ring_jfunc
def example_jfunc(r, jargs):
    peak_r = jargs[0]
    thickness = jargs[1]
    return np.exp(-4.*np.log(2)*((r-peak_r)/thickness)**2)

def example_phi_jfunc(r,phi,jargs):
    peak_r = jargs[0]
    thickness = jargs[1]
    peak_phi = jargs[2]
    return np.exp(-4.*np.log(2)*((r-peak_r)/thickness)**2)*(1+np.sin(phi-peak_phi))



fov =60*eh.RADPERUAS
npix = 120
jfunc = example_phi_jfunc
jarg_names = ['peak_r','thickness']
jargs = [4.5, 2, np.pi/2]
MoDuas = 3.8 # M over D in uas
inc = 5/180*np.pi
zbl = 0.6
PA = 288/180*np.pi#270/180*np.pi# [0, 2*np.pi]
nmax=1
chi = -135/180*np.pi
eta = None
beta = 0.5
a = -0.99
iota=np.pi/2

b = KerrBam(fov, npix, jfunc, jarg_names, jargs, MoDuas, a, inc, zbl, PA=PA,  chi=chi,eta=eta, nmax=nmax, beta=beta, iota=iota, polflux=True, axisymmetric=False)

#look at the frame dragging by plotting the BL phi grid
prims = b.get_primitives()
phi = prims[1][0]%(2*np.pi)
plt.imshow(phi.reshape((npix,npix)))
plt.colorbar()
plt.show()

im = b.make_rotated_image()
im.display(plotp=True)
