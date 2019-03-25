#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created: 2018
Author: A. P. Naik
Description: File containing likelihood function 'lnlike', to be fed to
emcee sampler.
"""
import numpy as np
from .rotcurve import v_model


def lnlike(theta, theta_dict, galaxy, **kwargs):
    """
    For given fit parameters (contained in 'theta') and data (contained in
    'galaxy'), returns a Gaussian log-likelihood. This function is fed to emcee
    sampler. If 'infer_errors' is switched on, then additional error term is
    included.

    Parameters
    ----------
    theta : numpy.ndarray, shape (ndim,)
        Parameter values for which to calculate likelihood.
    theta_dict : dict
        Keys are names of free parameters, and values are indices. Indices are
        used, for example, in the stored Markov chains.
    galaxy : spam.data.SPARCGalaxy
        Instance of class spam.data.SPARCGalaxy, containing galaxy to be fit.
    **kwargs :
        Same as kwargs for spam.fit.GalaxyFit constructor. See documentation
        therein.

    Returns
    -------
    lnlike : float
        log-likelihood lnL(data | theta).
    """

    # calculate rotation curve model
    model = v_model(theta, theta_dict, galaxy, **kwargs)

    # calculate Gaussian likelihood with or without additional error component
    if kwargs['infer_errors']:
        sigma = np.sqrt(galaxy.v_err**2 + theta[theta_dict['sigma_gal']]**2)
        lnlike = -np.sum(0.5*((galaxy.v - model)/sigma)**2 + np.log(sigma))
    else:
        lnlike = -0.5*np.sum(((galaxy.v - model)/galaxy.v_err)**2)

    return lnlike
