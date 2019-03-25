#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created: 2018
Author: A. P. Naik
Description: File containing prior function 'lnprior', to be fed to emcee
sampler. Also various supplementary functions relating halo scaling relations
from Moster et al, 2013 (arxiv:1205.5807), and Dutton & Maccio, 2014
(arxiv:1402.7073).
"""
import numpy as np
from scipy.constants import G
from scipy.constants import parsec as pc

# physical constants and cosmology; all SI units.
delta = 93.6
h = 0.7
H0 = h*100*1000/(1e+6*pc)
Msun = 1.989e+30


def SHM(M_halo):
    """
    Stellar mass / halo mass function from Moster et al. 2013,
    (arxiv:1205.5807). For given halo mass, returns stellar mass.

    Parameters
    ----------
    M_halo : float or 1D numpy.ndarray
        Halo virial mass. UNITS: kg

    Returns
    -------
    M_star : float or 1D numpy.ndarray, same shape as M_halo
        Total stellar mass. UNITS: kg
    """

    # parameters from Moster et al. (2013)
    M1 = 10**(11.59)*Msun
    N = 0.0351
    beta = 1.376
    gamma = 0.608

    # calculate M_star
    X = M_halo/M1
    denom = X**(-beta) + X**gamma
    M_star = 2*N*M_halo/denom
    return M_star


def err_SHM(M_halo):
    """
    Error on stellar mass / halo mass function from Moster et al. 2013,
    (arxiv:1205.5807). For given halo mass, returns error on log10(m_stellar).

    Parameters
    ----------
    M_halo : float or 1D numpy.ndarray
        Halo virial mass. UNITS: kg

    Returns
    -------
    err_M_star : float or 1D numpy.ndarray, same shape as M_halo
        Error on log10(stellar mass)
    """

    # parameters and errors from Moster et al. 2013
    M1 = 10**(11.59)*Msun
    N = 0.0351
    beta = 1.376
    gamma = 0.608
    sig_M1 = M1*np.log(10)*0.236
    sig_N = 0.0058
    sig_beta = 0.153
    sig_gamma = 0.059

    # calculate stellar mass
    Ms = SHM(M_halo)

    # fractional errors on parameters
    err_N = sig_N*Ms/N
    err_M1 = sig_M1*Ms/M1
    err_beta = sig_beta*Ms/beta
    err_gamma = sig_gamma*Ms/gamma

    # add errors in quadrature for total error
    err_M_star = np.sqrt(err_N**2 + err_M1**2 + err_beta**2 + err_gamma**2)

    # convert to logspace
    err_M_star = err_M_star / (Ms*np.log(10))
    return err_M_star


def CMR(M_halo):
    """
    Concentration / halo mass relation from Dutton & Maccio 2014,
    (arxiv:1402.7073). For given halo mass, returns log concentration.

    Parameters
    ----------
    M_halo : float or 1D numpy.ndarray
        Halo virial mass. UNITS: kg

    Returns
    -------
    logc : float or 1D numpy.ndarray, same shape as M_halo
        log10(halo concentration)
    """

    # parameters from Dutton & Maccio 2014
    a = 1.025
    b = -0.097

    # calculate log(concentration)
    logc = a + b*np.log10(M_halo/(1e+12*Msun/h))
    return logc


def lnprior(theta, theta_dict, galaxy, **kwargs):
    """
    For given fit parameters (contained in 'theta') returns log-prior. If
    'baryon_bound' is switched on, huge negative value is returned for
    parameter values for which baryon fraction is super-cosmic. If
    'scaling_priors' is switched on, then scaling relations from Moster et al,
    2013 (arxiv:1205.5807), and Dutton & Maccio, 2014 (arxiv:1402.7073), are
    used as priors. Otherwise flat priors with bounds from Katz et al., 2016
    (arxiv:1605.05971) for all priors except fR0 and sigma_g. For fR0, bounds
    are 1e-9 and 2e-6. For sigma_g, bounds are 0 and twice the maximum
    observational error for galaxy in question.

    Parameters
    ----------
    theta : numpy.ndarray, shape (ndim,)
        Parameter values for which to calculate prior.
    theta_dict : dict
        Keys are names of free parameters, and values are indices. Indices are
        used, for example, in the stored Markov chains.
    galaxy : spam.data.SPARCGalaxy
        Instance of class spam.data.SPARCGalaxy, containing galaxy to be fit.
    **kwargs :
        Same as kwargs for spam.fit.GalaxyFit constructor. See documentation
        therein. Additionally, prior_bounds_lower and prior_bounds_upper, which
        give the bounds of the priors on all parameters, set in GalaxyFit.

    Returns
    -------
    lnprior : float
        log-prior lnP(theta).
    """

    # get parameter bounds
    lb = kwargs['prior_bounds_lower']
    ub = kwargs['prior_bounds_upper']
    if not theta.shape == lb.shape == ub.shape:
        raise ValueError("Theta does not have same shape as theta bounds")

    # check if parameters are within bounds, otherwise -inf prior
    if (theta < ub).all() and (theta > lb).all():

        # determine mass-to-light ratio
        if kwargs['upsilon'] == 'fixed':
            ML = 0.5
        else:
            ML = 10**theta[theta_dict['ML_disc']]

        # calculate stellar and halo masses
        V_vir = 10**theta[theta_dict['V_vir']]
        c_vir = 10**theta[theta_dict['c_vir']]
        M_halo = (V_vir**3)/(np.sqrt(delta/2)*G*H0)
        M_star = ML*1e+9*galaxy.luminosity_tot*Msun

        # reject if M_baryon/M_DM > 0.2, if switch is on
        if kwargs['baryon_bound']:
            M_gas = (4/3)*galaxy.HI_mass
            if (M_star+M_gas)/M_halo > 0.2:
                return -1e+20

        # implement SHM and CMR scaling relation priors if switch is on
        if kwargs['scaling_priors']:

            # SHM
            y = np.log10(M_star)
            mu = np.log10(SHM(M_halo))
            sig = err_SHM(M_halo)
            sig += 0.2  # f(R) broadening
            g1 = ((y-mu)/sig)**2

            # CMR
            if kwargs['halo_type'] == 'DC14':  # convert DC14 c_vir to NFW
                X = np.log10(M_star/M_halo)
                exponent = 3.4*(X+4.5)
                c_NFW = c_vir/(1+1e-5*np.exp(exponent))  # Katz typo corrected
                y = np.log10(c_NFW)
            else:
                y = np.log10(c_vir)
            mu = CMR(M_halo)
            sig = 0.11  # from Dutton et al.
            sig += 0.1  # f(R) broadening
            g2 = ((y-mu)/sig)**2

            lnp = -0.5*g1*g2
            return lnp
        else:
            return 0
    else:
        return -np.inf
