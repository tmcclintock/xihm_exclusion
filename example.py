import numpy as np
import xihm_exclusion
import fastcorr
from matplotlib import pyplot as plt
from matplotlib.pyplot import rcParams
rcParams['text.usetex'] = True


# Set cosmology
from colossus.cosmology import cosmology
from classy import Class


# First set up cosmology to get power spectrum
Omega_b = 0.049
Omega_m = 0.318
Omega_cdm = Omega_m - Omega_b
h = 0.6704 #H0/100
sigma8 = 0.835
n_s = 0.962

# Fox cosmology for CLASS
foxclass = {'output': 'mPk',
            'non linear': 'halofit',
            'Omega_b': Omega_b,
            'Omega_cdm': Omega_cdm,
            'h': h,
            'sigma8': sigma8,
            'n_s': n_s,
            'P_k_max_1/Mpc': 1000,
            'z_max_pk': 10.  # Default value is 10
                }

classcosmo = Class()
classcosmo.set(foxclass)
classcosmo.compute()

# Fox cosmology for Colossus
foxcolossus = {'flat': True,
               'H0': h * 100 ,
               'Om0': Omega_m,
               'Ob0': Omega_b,
               'sigma8': sigma8,
               'ns': n_s}

cosmology.addCosmology('fox', foxcolossus)
cosmology.setCosmology('fox')
cosmo = cosmology.setCosmology('fox')

Om = Omega_m


# Power spectrum
k = np.logspace(-4, 2, num=1000)  # 1/Mpc
z = 0.
Plin = np.array([classcosmo.pk_lin(ki, z) for ki in k])
Pnl = np.array([classcosmo.pk(ki, z) for ki in k])
# NOTE: You will need to convert these to h/Mpc and (Mpc/h)^3
# to use in the toolkit. To do this you would do:
k /= foxclass['h']
Plin *= foxclass['h'] ** 3
Pnl *= foxclass['h'] ** 3
# End of set cosmology


# Read xihm data
################# Mbin 3e12-5e12 Msun/h #################
xihm = np.loadtxt('data/xihm1_3e12_5e12_z0.0_down10')
covxihm = np.loadtxt('data/covxihm1_3e12_5e12_z0.0_down10')
r = np.loadtxt('data/r')

# Parameters
rt = 1.780
m = 12.63
c = 10.50
beta = 0.2538
bias = 0.8836
ma = 12.39
ca = 0.0444
mb = 13.17
cb = 1.239

'''
################# Mbin 2e14-5e14 Msun/h #################
xihm = np.loadtxt('data/xihm1_2e14_5e14_z0.0_down10')
covxihm = np.loadtxt('data/covxihm1_2e14_5e14_z0.0_down10')
r = np.loadtxt('data/r')

# Parameters
rt = 1.717
m = 14.43
c = 6.658
beta = 0.4966
bias = 2.762
ma = 14.439
ca = 0.1210
mb = 12.51
cb = 0.0318
'''

# Get 2halo term
# xilin = fastcorr.calc_corr(r, k, Plin, N=500000, h=1e-4)
xinl = fastcorr.calc_corr(r, k, Pnl, N=500000, h=1e-4)
xi2h = xinl


# Get model
model = xihm_exclusion.xihm_model(r, rt, m, c, beta, bias, ma, ca, mb, cb, xi2h, Om)


# Plot against data
plt.errorbar(r, r * r * xihm, yerr=r * r * np.sqrt(covxihm.diagonal()), marker='o', linewidth=0, markersize=3, elinewidth=1, capsize=2, label=r'data', zorder=1)
plt.loglog(r, r * r * model, linewidth=2, label=r'best fit', zorder=2)
plt.xlabel(r'$r$', fontsize=24)
plt.ylabel(r'$r^2 \xi_{hm}$', fontsize=24)
plt.legend()
plt.savefig('figs/xihm.png')
plt.clf()

plt.axhline(0., color='black')
plt.semilogx(r, xihm/model - 1.)
plt.fill_between(r, -np.sqrt(covxihm.diagonal())/xihm, np.sqrt(covxihm.diagonal())/xihm, color='grey',alpha=0.2, label=r'data', zorder=1)
plt.ylim(-0.13, 0.05)
plt.xlabel(r'$r$', fontsize=24)
plt.ylabel(r'fractional difference', fontsize=24)
plt.savefig('figs/fracdif_xihm.png')