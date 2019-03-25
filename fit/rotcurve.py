#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created: 2018
Author: A. P. Naik
Description: File containing two routines: 'v_model', which gives the model
rotation curve for a given set of fit parameters, and 'mg_acceleration', which
calculates the 5th force contribution to the model rotation curve.
"""
import numpy as np
from .solvers import Grid1D, Grid2D
from . import models
from scipy.constants import parsec as pc
from scipy.constants import c as clight

# physical constants and cosmology
kpc = 1e+3*pc
Mpc = 1e+6*pc
h = 0.7
omega_m = 0.308


def v_model(theta, theta_dict, galaxy, component_split=False, **kwargs):
    """
    For parameter set contained in theta and SPARC galaxy contained in
    'galaxy', returns model rotation curve. Model hyperparameters are contained
    in kwargs.

    Parameters
    ----------
    theta : numpy.ndarray, shape (ndim,)
        Parameter values for which to calculate likelihood.
    theta_dict : dict
        Keys are names of free parameters, and values are indices. Indices are
        used, for example, in the stored Markov chains.
    galaxy : spam.data.SPARCGalaxy
        Instance of class spam.data.SPARCGalaxy, containing galaxy to be fit.
    component_split : bool
        If True, then function returns v_c, v_gas, v_disc, v_bulge, v_DM and
        v_5, rather than merely v_c.
    **kwargs :
        Same as kwargs for spam.fit.GalaxyFit constructor. See documentation
        therein.

    Returns
    -------
    v_c : numpy.ndarray, shape (number of data points)
        Model rotation curve. UNITS: km/s
    [v_gas, v_disc, v_bulge, v_DM, v_5] : numpy.ndarrays, same shapes as v_c
        Returned if component_split is True. Individual component contributions
        tot he rotation curve model. UNITS: km/s
    """

    # determine mass-to-light ratios
    if kwargs['upsilon'] == 'fixed':
        ML_disc = 0.5
        ML_bulge = 0.7
    else:
        ML_disc = 10**theta[theta_dict['ML_disc']]
        ML_bulge = 10**theta[theta_dict['ML_bulge']]

    # convert radii to metres
    R = galaxy.R*kpc

    # get baryonic components in m/s
    v_g = 1e+3*galaxy.v_gas
    v_d = 1e+3*galaxy.v_disc
    v_b = 1e+3*galaxy.v_bul

    # DM contribution; m/s
    v_DM = models.halo_v_circ(R, theta, theta_dict, galaxy,
                              halo_type=kwargs['halo_type'],
                              upsilon=kwargs['upsilon'])

    # total Newtonian acceleration
    a_N = (v_DM**2 + v_g**2 + ML_bulge*v_b**2 + ML_disc*v_d**2)/R

    # fifth force
    if kwargs['fR']:
        a_5 = mg_acceleration(theta, theta_dict, galaxy, **kwargs)
    else:
        a_5 = np.zeros_like(a_N)

    # calculate model rotation curve in km/s
    v_c = 1e-3*np.sqrt((a_N+a_5)*R)
    if component_split:
        v_DM = 1e-3*v_DM
        v_gas = 1e-3*v_g
        v_disc = 1e-3*np.sqrt(ML_disc*v_d**2)
        v_bulge = 1e-3*np.sqrt(ML_bulge*v_b**2)
        v_5 = 1e-3*np.sqrt(a_5*R)
        return v_c, v_gas, v_disc, v_bulge, v_DM, v_5
    else:
        return v_c


def mg_acceleration(theta, theta_dict, galaxy, **kwargs):
    """
    For parameter set contained in theta and SPARC galaxy contained in
    'galaxy', returns acceleration due to fifth force evaluated at radii of
    observed rotation curve.

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
    a_5 : numpy.ndarray, shape (number of data points)
        Acceleration due to fifth force at rotation curve radii. UNITS: m/s^2
    """

    # check whether fR0 free or fixed
    if kwargs['fR_parameter'] == 'free':
        fR0 = 10**theta[theta_dict['fR0']]
    elif kwargs['fR_parameter'] == 'fixed':
        fR0 = 10**kwargs['log10fR0']
    else:
        raise KeyError

    # set up scalar field solver grid
    dim = kwargs['MG_grid_dim']
    if dim == 2:
        grid = Grid2D(ngrid=175, rmin=0.05*kpc, rmax=5*Mpc)
        grid.set_cosmology(h=h, omega_m=omega_m, redshift=0)
        grid.rho = np.zeros((grid.ngrid, grid.nth), dtype=np.float64)
    elif dim == 1:
        grid = Grid1D(ngrid=175, rmin=0.05*kpc, rmax=5*Mpc)
        grid.set_cosmology(h=h, omega_m=omega_m, redshift=0)
        grid.rho = np.zeros((grid.ngrid,), dtype=np.float64)
    else:
        raise KeyError

    # add density profiles from mass models
    grid.rho += models.gas_disc(galaxy, grid)
    u = kwargs['upsilon']
    ht = kwargs['halo_type']
    if not kwargs['stellar_screening']:
        grid.rho += models.stellar_disc(theta, theta_dict, galaxy, u, grid)
        grid.rho += models.stellar_bulge(theta, theta_dict, galaxy, u, grid)
    grid.rho += models.DM_halo(theta, theta_dict, galaxy, ht, u, grid)

    # add uniform sphere to account for external potential from scr. map
    if kwargs['env_screening']:
        fR0_vals = np.linspace(np.log10(1.563e-8), np.log10(2e-6), 20)
        lfR0 = np.log10(fR0)
        phi_ext = np.interp(lfR0, fR0_vals, galaxy.ext_potential)
        phi_ext = (10**phi_ext)*clight**2
        l_compton = 32*10**(0.5*(lfR0+4))*Mpc
        grid.rho += models.top_hat(l_compton, phi_ext, grid)

    grid.drho = grid.rho - grid.rhomean
    grid.DensityFlag = True

    # solve for scalar field
    grid.iter_solve(niter=1000000, F0=-fR0)

    # calculate acceleration on grid
    dfdr = np.zeros(grid.ngrid+1)
    if dim == 2:
        ind = grid.disc_ind
        dfdr[1:-1] = np.diff(grid.fR[:, ind])/(grid.rout[:-1]*grid.dx)
    else:
        dfdr[1:-1] = np.diff(grid.fR)/(grid.rout[:-1]*grid.dx)

    # calculate acceleration at RC radii
    a_5 = np.interp(galaxy.R, np.append(grid.rin[0], grid.rout)/kpc, -dfdr)
    a_5 = 0.5*clight**2*a_5
    return a_5
