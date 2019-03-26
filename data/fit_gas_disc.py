#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created: 2018
Author: A. P. Naik
Description: Code to fit exponential disc models to SPARC galaxy gas profiles,
and create a file containing best fit disc radii for each galaxy.
"""
import spam
import numpy as np
from scipy.constants import G
from scipy.constants import parsec as pc
from scipy.special import i0, i1, k0, k1
from scipy.optimize import curve_fit
kpc = 1e+3*pc


class GalaxyDiscFit:

    def __init__(self, galaxy):

        self.HI_mass = galaxy.HI_mass
        self.R = galaxy.R*kpc
        self.v_gas = galaxy.v_gas*1e+3
        self.R_d = galaxy.disc_scale

        return

    def v_circ_sq(self, R, R_d):
        """
        Circular velocity, calculated according to Eq 2-169 of Binney+Tremaine.
        I and K are modified Bessel functions of the first and second kind, as
        given in the appendix 1.C-7 of Binney+Tremaine.
        """

        sigma_0 = 2*self.HI_mass/(3*np.pi*R_d**2)
        const = 4*np.pi*G*sigma_0*R_d

        y = R/(2*R_d)

        bessel_term = i0(y)*k0(y) - i1(y)*k1(y)
        v_sq = const * (y**2) * bessel_term

        return v_sq


# text file in which to store gas disc radii
fitfile = open('SPARCData/gas_radii.txt', 'w')

# loop over galaxies
for name in spam.data.names_full:

    galaxy = spam.data.SPARCGalaxy(name)

    # create data structure
    fitclass = GalaxyDiscFit(galaxy=galaxy)

    # fit
    bounds = ([0.1*fitclass.R_d], [5*fitclass.R_d])
    popt, pcov = curve_fit(fitclass.v_circ_sq, fitclass.R, fitclass.v_gas**2,
                           p0=2*fitclass.R_d, bounds=bounds)
    fitfile.write(name+'\t'+str(popt[0])+'\n')

fitfile.close()
