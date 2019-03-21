#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created:
Author: A. P. Naik
Description:
"""
import numpy as np
from .rotcurve import v_model


def lnlike(theta, theta_dict, galaxy, **kwargs):
    """
    Gaussian log-likelihood function

    """

    model = v_model(theta, theta_dict, galaxy, **kwargs)

    if kwargs['infer_errors']:
        sigma = np.sqrt(galaxy.v_err**2 + theta[theta_dict['sigma_gal']]**2)
        lnlike = -np.sum(0.5*((galaxy.v - model)/sigma)**2 + np.log(sigma))
    else:
        lnlike = -0.5*np.sum(((galaxy.v - model)/galaxy.v_err)**2)

    return lnlike
