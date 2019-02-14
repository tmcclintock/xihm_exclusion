import numpy as np
import cluster_toolkit
import fastcorr
from colossus.halo.mass_so import M_to_R
from colossus.halo.concentration import concentration
from scipy.special import sici
from scipy.special import erfc
from scipy.interpolate import interp1d
from scipy.integrate import quad
from scipy.integrate import simps


rhocrit = 2.77536627e+11


# Functions
def xihm_model(r, rt, m, c, alpha, bias, ma, mb, xi2h, Om):
    """Our model for the halo-mass correlation function.

    Args:
        r (float or array like): 3d distances from halo center in Mpc/h comoving.
        rt (float): Truncation radius. The boundary of the halo.
        m (float): Mass in Msun/h.
        c (float): Concentration.
        alpha (float): Width parameter of the exclusion function.
        bias (float): Large scale halo bias.
        ma (float): Mass of the first correction term in Msun/h.
        mb (float): Mass of the second correction term in Msun/h.
        xi2h (float or array like): 2-halo term at r.
        Om (float): Omega_matter.

    Returns:
        float or arraylike: Halo-mass correlation function.

    """

    rhom = Om * rhocrit

    # Get xi1h
    xi1h = 1. + cluster_toolkit.xi.xi_nfw_at_R(r, m, c, Om)
    xi1h *= thetat(r, rt, alpha)

    # Get concentrations
    ca = concentration(M=ma, mdef='200m', z=0.)
    cb = concentration(M=mb, mdef='200m', z=0.)

    # Substract 1halo term
    r1 = M_to_R(m, z=0., mdef='200m')
    rb = M_to_R(mb, z=0., mdef='200m')
    rb = rt / r1 * rb
    xi2h = xi2h - mb / rhom * utconvut(r, rb, cb)

    # Get xi2h
    xi2h = bias * xi2h

    # Get correction
    C = - utconvthetae(r, rt, m, alpha, ma, ca, Om) * xi2h - utconvthetae(r, rt, m, alpha, mb, cb, Om)

    # Full xi
    xi = xi1h + xi2h + C

    return xi


def thetat(r, rt, alpha):
    """Exclusion function. Error function.

        Args:
            r (float or array like): 3d distances from halo center in Mpc/h comoving.
            rt (float): Truncation radius. The boundary of the halo.
            alpha (float): Width parameter of the exclusion function.

        Returns:
            float or arraylike: Exclusion function.

    """

    return 0.5 * erfc((r-rt)/alpha/rt/np.sqrt(2.))


def re(r1, r2, scheme=1):
    """Exclusion radius given a percolation scheme. Default Rockstar is 1.

            Args:
                r1 (float): 3d distance from halo center in Mpc/h comoving.
                r2 (float): 3d distance from halo center in Mpc/h comoving.
                scheme (integer): Percolation scheme.

            Returns:
                float: Exclusion radius in Mpc/h comoving.

    """

    if scheme == 1:
        re = max(r1, r2)
    elif scheme == 2:
        re = (r1**3 + r2**3)**(1/3)
    elif scheme == 3:
        re = r1 + r2

    return re


def utconvthetae(r, rt1, mass1, alpha, mass2, conc2, Om):
    """Function that goes into the two correction terms. $(u_t \star \theta_e) (r | m1, m2)$.

        Args:
            r (float or array like): 3d distances from halo center in Mpc/h comoving.
            rt1 (float): Truncation radius. The boundary of the halo.
            mass1 (float): Mass in Msun/h.
            alpha (float): Width parameter of the exclusion function.
            mass2 (float): Mass in Msun/h.
            conc2 (float): Concentration.
            Om (float): Omega_matter.

        Returns:
            float or arraylike: $(u_t \star \theta_e) (r | m1, m2)$

    """

    rhom = Om * rhocrit

    # Get proportions
    r1 = M_to_R(mass1, z=0., mdef='200m')
    r2 = M_to_R(mass2, z=0., mdef='200m')
    rt2 = rt1 / r1 * r2

    # Get ut(r)
    rm = np.logspace(-3, 1.5, num=100, base=10)
    ut_r = 1. + cluster_toolkit.xi.xi_nfw_at_R(rm, mass2, conc2, Om)
    rho = interp1d(rm, 4 * np.pi * rm ** 2 * rhom * ut_r)
    aux = quad(rho, rm[0], rt2, limit=200, epsrel=1e-3)
    mass2t = aux[0]
    ut_r *= thetat(rm, rt2, alpha) / mass2t * rhom
    ut_r_interp = interp1d(rm, ut_r, fill_value=0, bounds_error=False)

    # Get thetaet(r)
    Re = re(rt1, rt2, scheme=1)
    thetat_r = thetat(rm, Re, alpha)
    thetat_r_interp = interp1d(rm, thetat_r, fill_value=(1, 0), bounds_error=False)

    # Double integral
    integral = np.zeros(np.size(r))
    u = np.linspace(-1, 1, num=10)
    g_at_u = np.zeros_like(u)
    for j, ri in enumerate(r):
        for i, ui in enumerate(u):
            def f(y):
                return y * y * ut_r_interp(y) * thetat_r_interp(np.sqrt(ri * ri + y * y - 2 * ri * y * ui))

            g_at_u[i] = simps(f(rm), rm, even='first')
        integral[j] = 2 * np.pi * simps(g_at_u, u, even='first')

    return integral


def utconvut(r, rt, conc):
    """1-halo contribution to be substracted from the 2-halo term (ximm).

            Args:
                r (float or array like): 3d distances from halo center in Mpc/h comoving.
                rt (float): Truncation radius. The boundary of the halo.
                conc (float): Concentration.

            Returns:
                float or arraylike: 1-halo contribution.

    """

    # Get ut(k)
    # Analytic expression
    k = np.logspace(-4, 4, num=1000, base=10)
    rc = rt / conc
    mu = k * rc
    Put = (np.cos(mu) * (sici(mu + mu * conc)[1] - sici(mu)[1])
           + np.sin(mu) * (sici(mu + mu * conc)[0] - sici(mu)[0])
           - np.sin(mu * conc) / (mu + mu * conc)) \
          / (np.log(1 + conc) - conc / (1 + conc))

    # Product
    PutPut = Put * Put

    # Transform to real space
    integral = fastcorr.calc_corr(r, k, PutPut, N=500, h=1e-2)

    return integral
