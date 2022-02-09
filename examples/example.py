import numpy as np
import matplotlib.pyplot as plt
from bam.inference.kerrbam import KerrBam
import ehtim as eh


#this example emissivity envelope could also have been imported with from bam.inference.jfuncs import ring_jfunc
def example_jfunc(r, jargs):
    peak_r = jargs[0]
    thickness = jargs[1]
    return np.exp(-4.*np.log(2)*((r-peak_r)/thickness)**2)

obs = eh.obsdata.load_uvfits('SR1_M87_2017_101_lo_hops_netcal_StokesI.uvfits')
obs.add_scans()
obs_sa = obs.avg_coherent(0., scan_avg=True)

fov =60*eh.RADPERUAS
npix = 40
jfunc = example_jfunc
jarg_names = ['peak_r','thickness']
jargs = [4.5, 2.]#, np.pi/2]
MoDuas = 3.8 # M over D in uas
inc = 17/180*np.pi
zbl = 0.6
PA = 288/180*np.pi#270/180*np.pi# [0, 2*np.pi]
nmax=0
chi = -135/180*np.pi
eta = None
beta = 0.5
a = -0.5
iota=np.pi/2

b = KerrBam(fov, npix, jfunc, jarg_names, jargs, MoDuas, a, inc, zbl, PA=PA,  chi=chi,eta=eta, nmax=nmax, beta=beta, iota=iota, polflux=True)

im = b.make_image()
im.rf = obs_sa.rf
b.make_rotated_image().display(plotp=True)#, nvec=40)

to_fit = im.observe_same(obs_sa, ampcal=False,phasecal=False)
to_fit.plotall('u','v',conj=True)

frac = 0
sig = 0
to_fit = to_fit.add_fractional_noise(frac*1e-2)
to_fit.data['sigma'] = (to_fit.data['sigma']**2+(sig*1e-3)**2)**0.5
to_fit.plotall('uvdist','amp')

jargs_to_fit = jargs#.5,2.5]]#jargs#[[6,10],[0.5,4]]#[0.5,6]]#,np.pi/2]
MoDuas_to_fit = [1,5]
a_to_fit = [-0.99, -0.01]#, 0.99]
inc_to_fit =inc#[1*np.pi/180, 30*np.pi/180]
beta_to_fit =beta#[0.,0.9]
chi_to_fit = chi#[-np.pi, 0]
eta_to_fit = eta
PA_to_fit = PA#288/180*np.pi
zbl_to_fit = zbl#im.total_flux()
iota_to_fit = iota#[np.pi/4,np.pi/2]
truths = [a]
nmax_to_fit = nmax
# jargs_to_fit = jargs
# M_to_fit=M
# zbl = [0.5, 1.5]
modelb = KerrBam(fov, npix, jfunc, jarg_names, jargs_to_fit, MoDuas_to_fit, a_to_fit, inc_to_fit, zbl_to_fit, PA=PA_to_fit, chi=chi_to_fit, nmax=nmax_to_fit, beta=beta_to_fit, iota = iota_to_fit)

#let's fit!
# pool = Pool(processes=8)
dtypes = ['logcamp','cphase']

#first, try to find the MAP with simulated annealing
#you can skip this if you want to get right to the posteriors
MAP_Bam = modelb.annealing_MAP(to_fit,data_types=dtypes)
MAP_Bam.make_rotated_image().display()


modelb.setup(to_fit, data_types=dtypes, nlive=250,dynamic=True)#, pool=pool, queue_size=8)


modelb.run_nested_default()

outname = 'selffita0.5_truen'+str(nmax)+'_fitn'+str(nmax_to_fit)+'_'+'_'.join(dtypes)

# if nd:
#   outname += '_nd'
outname+='_frac'+str(frac)+'_sig'+str(sig)

modelb.traceplot(save=outname+'_trace.png', show=False)
truths = [a]#[M,a] + [2]
modelb.cornerplot(save=outname+'_corner.png', show=False,truths=truths)
#display the Mean of Posterior
MOP_Bam = modelb.MOP_Bam()
MOP_Bam.make_rotated_image().display(export_pdf=outname+'_MOP.png')

randomim = modelb.random_sample_Bam().make_rotated_image()
randomim.display(export_pdf=outname+'_random.png',label_type='scale',has_cbar=False,has_title=False, show=False)
randomim.save_fits(outname+'_random.fits')

modelb.save_posterior(outname=outname)

print("Here are a few chisqs!")
chisqfile = open(outname+'_params+chisqs.txt','w')
chisqfile.write(str(modelb.all_names)+'\n')
chisqfile.write(str(modelb.all_params)+'\n')
for i in range(10):
    d = modelb.random_sample_Bam().all_chisqs(to_fit)
    print(d)
    chisqfile.write(str(d)+'\n')

chisqfile.close()

