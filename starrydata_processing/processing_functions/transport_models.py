from fdint import fdk  # function that implements Fermi-Dirac integrals
from scipy.optimize import minimize

import numpy as np

'''
this module implements a model for the conductivity and seebeck coefficient

sigma_E_0 quantifies electronic performance independant of Fermi level and
temperature, similar to weighted mobility. sigma_E_0 can be extracted from
conductivity and seebeck data.

the model used to extract sigma_E_0 assumes that the underlying transport
function is a powerlaw with energy, where sigma_E_0 is the powerlaw prefactor.

sigma = integral(sigma_E_0 * (E/kT)^s  * (-df/dE) dE)

sigma_E_0 (S/m) is the powerlaw prefactor
s (unitless) determines the transport mechanism

for inorganic materials dominated by deformation potential scattering, s=1

more information on this model: https://www.nature.com/articles/nmat4784
'''

constant = {'e': 1.60217662e-19,  # physical constants
            'k': 1.38064852e-23}


def model_conductivity(cp, sigma_E_0, s=1):
    '''
    returns the electrical conductivity in S/m

    Args:
      cp: (float/ndarray) reduced chemical potential (mu/kT), unitless
      sigma_E_0: (float) powerlaw prefactor, S/m
      s: (int|half-integer) energy exponent, unitless

    Returns: (float/ndarray) conductivity, S/m
    '''

    if s == 0:  # s=0 requires analytic simplification
        return sigma_E_0 / (1. + np.exp(-cp))
    else:
        return sigma_E_0 * s * fdk(s - 1, cp)


def model_seebeck(cp, s=1):
    '''
    returns the seebeck coeficient in V/K

    Args:
      cp: (float/ndarray) reduced chemical potential (mu/kT), unitless
      s: (int|half-integer) energy exponent, unitless

    Returns: (float/ndarray) seebeck coefficient, V/K
    '''

    if s == 0:  # s=0 requires analytic simplification
        return constant['k'] / constant['e'] * (((1. + np.exp(-cp)) *
                                                 fdk(0, cp)) - cp)
    else:
        return constant['k'] / constant['e'] * (((s + 1.) * fdk(s, cp) / s /
                                                 fdk(s - 1, cp)) - cp)


def model_lorentz_number(cp, s=1):
    '''
    returns the lorentz number in SI units

    Args:
        cp: (float/ndarray) reduced chemical potential (mu/kT), unitless
        s (int|half-integer) energy exponent, unitless

    Returns: (float/ndarray) lorentz number, SI units
    '''

    if s == 0:  # s=0 requires analytic simplification
        raise ValueError('s=0 not supported when calculating lorentz number!')
    else:
        return (constant['k'] / constant['e']) ** 2. *\
            (
                ((s + 2.) * fdk(s + 1., cp) / s / fdk(s - 1., cp)) -
                ((s + 1.) * fdk(s, cp) / s / fdk(s - 1., cp)) ** 2.
        )


def model_sigma_E_0(seebeck, conductivity, temperature, s=1):
    '''
    returns the transport function prefactor (sigma_E_0) in S/m

    the transport exponent (s) for inorganic semiconductors dominated by
    deformation potential scattering is 1 (a linear transport function)

    Args:
      seebeck (float) Seebeck coefficient, V/K
      conductivity (float) electrical conductivity, S/m
      temperature (float) absolute temperature, K
      s (int|half-integer) energy exponent, unitless


    Returns: (float) the transport function prefactor (sigma_E_0), S/m
    '''
    cp = minimize(
        lambda cp: np.abs(model_seebeck(cp, s) - np.abs(seebeck)),
        method='Nelder-Mead', x0=[0.]).x[0]
    return minimize(lambda sigma_E_0: np.abs(
        model_conductivity(cp, sigma_E_0, s) - conductivity),
        method='Nelder-Mead', x0=[0.]).x[0]


def model_kappa_l(seebeck, conductivity, temperature,
                  thermal_conductivity, s=1):
    '''
    returns the lattice portion of the thermal conductivity in W/mK

    Args:
        seebeck (float) Seebeck coefficient, V/K
        conductivity (float) electrical conductivity, S/m
        temperature (float) absolute temperature, K
        thermal_conductivity (float) the total thermal conductivity, W/mK
        s (int|half-integer) energy exponent, unitless
    '''
    cp = minimize(
        lambda cp: np.abs(model_seebeck(cp, s) - np.abs(seebeck)),
        method='Nelder-Mead', x0=[0.]).x[0]
    lorentz_number = model_lorentz_number(cp, s)
    return thermal_conductivity - (lorentz_number * conductivity * temperature)


def model_quality_factor(seebeck, conductivity, temperature,
                         thermal_conductivity, s=1):
    '''
    returns the thermoelectric quality factor (B)

    Args:
        seebeck (float) Seebeck coefficient, V/K
        conductivity (float) electrical conductivity, S/m
        temperature (float) absolute temperature, K
        thermal_conductivity (float) the total thermal conductivity, W/mK
        s (int|half-integer) energy exponent, unitless

    returns (float) the quality factor, unitless
    '''

    sigma_E_0 = model_sigma_E_0(
        seebeck, conductivity, temperature, s)
    kappa_l = model_kappa_l(
        seebeck, conductivity, temperature, thermal_conductivity, s)
    return (constant['k'] / constant['e']) ** 2 *\
        sigma_E_0 * temperature / kappa_l


if __name__ == '__main__':

    import matplotlib.pyplot as plt

    cps = np.arange(-10, 10)
    seebeck = np.array([model_seebeck(cp, s=1) for cp in cps])

    plt.plot(cps, seebeck * 1e6, 'k-')
    plt.xlabel(r'$\eta$', fontsize=15)
    plt.ylabel(r'Seebeck ($\mu V / K$)', fontsize=15)
    plt.show()
