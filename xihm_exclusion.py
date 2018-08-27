import numpy as np
import cluster_toolkit
import fastcorr
from colossus.halo.mass_so import M_to_R
from scipy.special import sici
from scipy.special import erfc
from matplotlib import pyplot as plt
from matplotlib.pyplot import rcParams
rcParams['text.usetex'] = True


########################################################
########################################################
########################################################
########################################################
########################################################
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
rhocrit = 2.77536627e+11
rhom = Om * rhocrit


# End of set cosmology
########################################################
########################################################
########################################################
########################################################
########################################################


# Functions
def xihm_model(r, rt, m, c, beta, bias, ma, ca, mb, cb, ximm):
    # Get xi1h
    xi1h = 1. + cluster_toolkit.xi.xi_nfw_at_R(r, 10**m, c, Om)
    xi1h *= theta_errfunc(r, rt, beta)

    # Get xi2h
    xi2h = bias * ximm

    # Get correction
    C = - bias * utconvthetae(r, rt, 10**m, 10**ma, ca) * ximm - utconvthetae(r, rt, 10**m, 10**mb, cb)

    # Full xi
    xi = xi1h + xi2h + C

    return xi


def re(r1, r2, scheme):
    if scheme == 1:
        re = max(r1, r2)
    elif scheme == 2:
        re = (r1**3 + r2**3)**(1/3)
    elif scheme == 3:
        re = r1 + r2

    return re


def utconvthetae(r, rt, mass1, mass2, conc2):
    # Get proportions
    r1 = M_to_R(mass1, z=0., mdef='200m') / 1000.
    r2 = M_to_R(mass2, z=0., mdef='200m') / 1000.
    rt1 = rt
    constant = rt1 / r1
    rt2 = constant * r2

    # Get ut(k)
    # Analytic expression
    k = np.logspace(-4, 4, num=1000, base=10)
    rc = rt2 / conc2
    mu = k * rc
    delta200 = 200. / 3. * conc2 ** 3 / (np.log(1 + conc2) - conc2 / (1 + conc2))
    Put = 3 * delta200 / 200. / conc2 ** 3 * (np.cos(mu) * (sici(mu + mu * conc2)[1] - sici(mu)[1]) +
                                              np.sin(mu) * (sici(mu + mu * conc2)[0] - sici(mu)[0])
                                              - np.sin(mu * conc2) / (mu + mu * conc2))

    # Get Pthetae
    Re = re(rt1, rt2, scheme=1)
    Pthetae = 4 * np.pi * (np.sin(k * Re) - k * Re * np.cos(k * Re)) / k ** 3

    # Product
    PutPthetae = Put * Pthetae

    # Transform to real space
    low = r < Re
    high = r >= Re
    integral1 = fastcorr.calc_corr(r[low], k, PutPthetae, N=8000, h=1e-6)
    integral2 = fastcorr.calc_corr(r[high], k, PutPthetae, N=7000, h=1e-5)

    integral = np.concatenate((integral1, integral2))

    integral1 = fastcorr.calc_corr(r[low], k, PutPthetae, N=8800, h=1e-6)
    integral2 = fastcorr.calc_corr(r[high], k, PutPthetae, N=7700, h=1e-5)

    #integrall = np.concatenate((integral1, integral2))

    #plt.semilogx(r, integral)
    #plt.show()

    #plt.semilogx(r, (integrall+1.)/(integral+1.) - 1.)
    #plt.show()
    #exit()

    return integral


def theta_errfunc(r, rt, beta):
    return 0.5 * erfc((r-rt)/beta/np.sqrt(2.))



#############################################################
#############################################################
#############################################################
#############################################################
#############################################################
#############################################################


# Read xihm data
down=10
xihm = np.loadtxt('data/xihm1_3e12_5e12_z0.0_down10')
covxihm = np.loadtxt('data/covxihm1_3e12_5e12_z0.0_down10')
r = np.loadtxt('data/r')

# Get 2halo term
#xilin = fastcorr.calc_corr(r, k, Plin, N=500000, h=1e-4)
xinl = fastcorr.calc_corr(r, k, Pnl, N=500000, h=1e-4)
xi2h = xinl


rt = 1.780
m = 12.63
c = 10.50
beta = 0.2538
bias = 0.8836
ma = 12.39
ca = 0.0444
mb = 13.17
cb = 1.239

model = xihm_model(r, rt, m, c, beta, bias, ma, ca, mb, cb, xi2h)

plt.errorbar(r, r * r * xihm, yerr=r * r * np.sqrt(covxihm.diagonal()), marker='o', linewidth=0, markersize=3, elinewidth=1, capsize=2, label=r'data', zorder=1)
plt.loglog(r, r * r * model, linewidth=2, label=r'best fit', zorder=2)
plt.xlabel(r'$r$', fontsize=24)
plt.ylabel(r'$r^2 \xi_{hm}$', fontsize=24)
plt.legend()
plt.savefig('figs/xihm.png')
plt.clf()

plt.axhline(0., color='black')
plt.semilogx(r, model/xihm - 1.)
plt.fill_between(r, -np.sqrt(covxihm.diagonal())/xihm, np.sqrt(covxihm.diagonal())/xihm, color='grey',alpha=0.2, label=r'data', zorder=1)
plt.ylim(-0.05, 0.13)
plt.xlabel(r'$r$', fontsize=24)
plt.ylabel(r'fractional difference', fontsize=24)
plt.savefig('figs/fracdif_xihm.png')