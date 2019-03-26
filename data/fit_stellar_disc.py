#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created: 2018
Author: A. P. Naik
Description: Code to fit galactic disc component for SPARC galaxies, and store
parameters in text file
"""
import spam
from scipy.optimize import curve_fit
import numpy as np
from scipy.constants import G
from scipy.special import i0, i1, k0, k1
from scipy.constants import parsec as pc
Msun = 1.989e+30
kpc = 1e+3*pc


def v_disc_anal(R, sigma_0, R_d):
    """
    Analytic expression for rotation speed of exponential disc, from Binney and
    Tremaine.
    sigma_0 in kg/m^2, R_d and R in m. Returns v in km/s
    """

    const = 4*np.pi*G*sigma_0*R_d
    y = R/(2*R_d)
    bessel_term = i0(y)*k0(y) - i1(y)*k1(y)
    v = 1e-3*np.sqrt(np.abs(const * (y**2) * bessel_term))

    return v


# text file to store disc parameters
fitfile = open("SPARCData/stellar_disc_parameters.txt", 'w')

# loop over galaxies
for name in spam.data.names_full:

    galaxy = spam.data.SPARCGalaxy(name)

    R_d = galaxy.disc_scale  # metres
    sigma_0 = Msun*galaxy.disc_SB  # kg/m^2

    # fit
    bounds = ((0.1*sigma_0, 0.1*R_d), (10*sigma_0, 10*R_d))
    popt, pcov = curve_fit(v_disc_anal, galaxy.R*kpc, galaxy.v_disc,
                           p0=(0.5*sigma_0, R_d), bounds=bounds)
    fitfile.write(galaxy.name+'\t'+str(popt[0])+'\t'+str(popt[1])+'\n')

fitfile.close()
