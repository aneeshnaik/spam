#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created: 2018
Author: A. P. Naik
Description: Code to fit Hernquist profile to v_bulge; store Hernquist
parameters for all galaxies to SPARCData/hernquist_parameters.txt
"""
import spam.data
import numpy as np
from scipy.constants import G
from scipy.optimize import curve_fit
from scipy.constants import parsec as pc
kpc = 1e+3*pc


def v_hernquist(r, log10rho, a):
    """
    Circular velocity of hernquist profile, in km/s. r in m, log10rho is
    log10 of density in kg/m^3, as in m.
    """

    rho_0 = np.power(10, log10rho)

    v_sq = 2*np.pi*G*rho_0*a*r/(1+r/a)**2
    v_circ = 1e-3*np.sqrt(v_sq)

    return v_circ


# store hernquist parameters in text file
fitfile = open("SPARCData/hernquist_parameters.txt", 'w')

# loop over galaxies
for name in spam.data.names_full:

    gal = spam.data.SPARCGalaxy(name)

    # check galaxy has bulge-disc decomaposition; if no then empty line
    if not gal.StellarBulge:
        fitfile.write(gal.name+'\n')
        continue

    # fit v_bulge to hernquist profile
    bounds = ([-21, 0], [-14, 50*kpc])
    popt, pcov = curve_fit(v_hernquist, gal.R*kpc, gal.v_bul, maxfev=2000,
                           p0=(-20.0, 10*kpc), bounds=bounds)
    fitfile.write(gal.name+'\t'+str(10**popt[0])+'\t'+str(popt[1])+'\n')

fitfile.close()
