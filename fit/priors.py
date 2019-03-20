#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created: 2nd December 2018
Author: A. P. Naik
Description: Priors based on M_star/M_halo mass and c/M_halo scaling relations
based on abundance matching.
"""
import numpy as np
from scipy.constants import G
from scipy.constants import parsec as pc
delta = 93.6
h = 0.7
H0 = h*100*1000/(1e+6*pc)
Msun = 1.989e+30


# Moster parameters, with errors
M1 = 10**(11.59)*Msun
sig_M1 = M1*np.log(10)*0.236
N = 0.0351
sig_N = 0.0058
beta = 1.376
sig_beta = 0.153
gamma = 0.608
sig_gamma = 0.059


# Dutton parameters, and error on log10c
a = 1.025
b = -0.097
err_logc = 0.11


def SHM(M_halo):

    X = M_halo/M1
    denom = X**(-beta) + X**gamma
    M_star = 2*N*M_halo/denom
    return M_star


def err_SHM(M_halo):

    Ms = SHM(M_halo)
    err_N = sig_N*Ms/N
    err_M1 = sig_M1*Ms/M1
    err_beta = sig_beta*Ms/beta
    err_gamma = sig_gamma*Ms/gamma

    err_M_star = np.sqrt(err_N**2 + err_M1**2 + err_beta**2 + err_gamma**2)
    return err_M_star


def CMR(M_halo):

    logc = a + b*np.log10(M_halo/(1e+12*Msun/h))
    return logc


# log prior; flat in logspace, bounds from Katz et al
def lnprior(theta, theta_dict, lb, ub, galaxy, halo_type, ML_ratio,
            CosmicBaryonBound, scaling_priors):

    assert theta.shape == lb.shape == ub.shape

    if (theta < ub).all() and (theta > lb).all():

        if ML_ratio == 'fixed':
            ML = 0.5
        else:
            ML = 10**theta[theta_dict['ML_disc']]

        V_vir = 10**theta[theta_dict['V_vir']]
        c_vir = 10**theta[theta_dict['c_vir']]

        M_halo = (V_vir**3)/(np.sqrt(delta/2)*G*H0)
        M_star = ML*1e+9*galaxy.luminosity_tot*Msun

        # reject if M_baryon/M_DM > 0.2, if CosmicBaryonBound flag is set
        if CosmicBaryonBound:
            M_gas = (4/3)*galaxy.HI_mass
            if (M_star+M_gas)/M_halo > 0.2:
                return -1e+20

        if scaling_priors:
            y = np.log10(M_star)
            mu = np.log10(SHM(M_halo))
            sig = err_SHM(M_halo)/(SHM(M_halo)*np.log(10))
            sig += 0.2  # f(R) broadening
            g1 = ((y-mu)/sig)**2

            if halo_type == 'DC14':  # convert DC14 c_vir to NFW c_vir
                X = np.log10(M_star/M_halo)
                exponent = 3.4*(X+4.5)
                c_NFW = c_vir/(1+1e-5*np.exp(exponent))  # Katz typo corrected
                y = np.log10(c_NFW)
            else:
                y = np.log10(c_vir)
            mu = CMR(M_halo)
            sig = err_logc
            sig += 0.1  # f(R) broadening
            g2 = ((y-mu)/sig)**2

            lnp = -0.5*g1*g2
            return lnp
        else:
            return 0
    else:
        return -np.inf
