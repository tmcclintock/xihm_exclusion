import numpy as np
import cluster_toolkit
import fastcorr
from colossus.halo.mass_so import M_to_R
from scipy.special import sici
from scipy.special import erfc


# Functions
def xihm_model(r, rt, m, c, beta, bias, ma, ca, mb, cb, ximm, Om):
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


def theta_errfunc(r, rt, beta):
    return 0.5 * erfc((r-rt)/beta/np.sqrt(2.))


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

    #integral1 = fastcorr.calc_corr(r[low], k, PutPthetae, N=8800, h=1e-6)
    #integral2 = fastcorr.calc_corr(r[high], k, PutPthetae, N=7700, h=1e-5)

    #integrall = np.concatenate((integral1, integral2))

    #plt.semilogx(r, integral)
    #plt.show()

    #plt.semilogx(r, (integrall+1.)/(integral+1.) - 1.)
    #plt.show()
    #exit()

    return integral