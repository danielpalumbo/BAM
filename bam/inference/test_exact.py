# from schwarzschildexact import *
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import numpy as np
import matplotlib.pyplot as plt
# import numpy as np
from gradients import unscaled_model_vis, scaled_model_vis
from blimage import Blimage
# r = 4.5
# theta = 0/180*np.pi
# phi = np.pi/2
# n = 2

# b, varphi = getscreencoords(r, phi, theta, n)
# psin = getpsin(b, theta, phi, n)
# alphan = getalphan(b, r, theta, psin)
# wa = getwindangle(b,r,phi, theta, n)
# print(alphan)
# print(wa)
# print(getsignpr(b,r,theta,psin))

# print(unscaled_model_vis(theta, 0, 0))
#first, without vectorization
u = np.array([0,1e9,2e9,3e9])
v = np.array([0, -1e9, -2e9, -3e9])
import time 
unvecs = []
vecs = []
anas = []
npixs = []
for nr in [2, 4, 8, 16, 32, 64, 128, 256]:
    for nphi in [2, 4, 8, 16, 32, 64, 128, 256]:

        # M, D, inc, j, zbl, PA, nmax, beta, chi, spec, rvec, phivec = theta  
        startTime = time.time()

        r_lims  = (4,10)
        phi_lims = (0, 2*np.pi)

        npixs.append(nphi*nr)
        r = np.sqrt(np.linspace(r_lims[0]**2, r_lims[1]**2, nr+1)[:-1])
        phi = np.linspace(phi_lims[0],phi_lims[1], nphi+1)[:-1]
        # self.r = r
        # self.phi = phi
        r_grid, phi_grid = np.meshgrid(r,phi)
        rvec = r_grid.flatten()
        phivec = phi_grid.flatten()

        M = 6.2e9 * 2e30
        D = 16.9 * 3.086e22
        inc = 17 / 180 * np.pi
        j = 1600 * np.ones_like(rvec)
        j[3]=0
        zbl = 0.6
        PA = 0
        nmax = 0
        beta = 0
        chi = 0
        spec = 1
        theta = [M, D, inc, j, zbl, PA, nmax, beta, chi, spec, rvec, phivec]

        # startTime = time.time()
        # print(scaled_model_vis(theta, u, v,vec=False))
        # endTime = time.time()
        # unvecs.append(endTime-startTime)
        # # print("Unvectorized exact time: ", endTime-startTime)

        vecvis = scaled_model_vis(theta, u, v,vec=True)
        endTime = time.time()
        vecs.append(endTime-startTime)
        print("Vectorized exact time: ", endTime-startTime)




        startTime = time.time()
        blim = Blimage(r_lims,phi_lims,nr,nphi,M,D,inc,1600,zbl,beta=beta,chi=chi, nmax=nmax)
        blim.blixels[3] = 0
        anavis = blim.vis(u, v)
        endTime = time.time()
        anas.append(endTime-startTime)
        print('Analytical time: ',endTime-startTime)
        # print("Analytic time: ", endTime-startTime)
# plt.plot(npixs, unvecs,'.', label='unvectorized exact')
plt.plot(npixs, vecs,'.', label = 'vectorized exact')
plt.plot(npixs, anas,'.', label='analytic')
plt.xlabel('number of blixels')
plt.ylabel('seconds of runtime')
plt.title('n=0+n=1 visibilities at 4 u-v points')
plt.yscale('log')
plt.xscale('log')
plt.legend()
plt.savefig('benchmarking.png',bbox_inches='tight')
plt.show()


# # starttime = time.time()
# # coords, radius = largestEmptyCircle(u,v)

# # endtime = time.time()

