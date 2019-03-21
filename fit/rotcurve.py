#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created: Summer 2018
Author: A. P. Naik
Description: Routine to calculate circular velocity profile in SPARC galaxy for
given set of parameters. This file also contains the DM halo model, and the
gaseous and stellar disc models.
"""
import numpy as np
from .solvers import grid_1D, grid_2D
from . import models
from scipy.constants import parsec as pc
from scipy.constants import c as clight
kpc = 1e+3*pc
Mpc = 1e+6*pc
h = 0.7
omega_m = 0.308


def v_model(theta, theta_dict, galaxy, component_split=False, **kwargs):
    """
    Calculate v_circ for parameter set contained in theta, on SPARC galaxy
    contained in 'galaxy' argument. Fifth force is optionally included by
    setting fR=True.
    """

    if kwargs['upsilon'] == 'fixed':
        ML_disc = 0.5
        ML_bulge = 0.7
    else:
        ML_disc = 10**theta[theta_dict['ML_disc']]
        ML_bulge = 10**theta[theta_dict['ML_bulge']]

    R = galaxy.R*kpc  # metres
    v_g = 1e+3*galaxy.v_gas  # m/s
    v_d = 1e+3*galaxy.v_disc
    v_b = 1e+3*galaxy.v_bul

    # NFW profile
    v_DM = models.halo_v_circ(R, theta, theta_dict, galaxy,
                              halo_type=kwargs['halo_type'],
                              upsilon=kwargs['upsilon'])

    # calculating Newtonian and MG accelerations
    a_N = (v_DM**2 + v_g**2 + ML_bulge*v_b**2 + ML_disc*v_d**2)/R

    if kwargs['fR']:
        a_5 = mg_acceleration(theta, theta_dict, galaxy, **kwargs)
    else:
        a_5 = np.zeros_like(a_N)

    # circular velocity
    v_c = 1e-3*np.sqrt((a_N+a_5)*R)  # km/s

    if component_split:
        # converting component velocities to km/s
        v_DM = 1e-3*v_DM
        v_gas = 1e-3*v_g
        v_disc = 1e-3*np.sqrt(ML_disc*v_d**2)
        v_bulge = 1e-3*np.sqrt(ML_bulge*v_b**2)
        v_5 = 1e-3*np.sqrt(a_5*R)
        return v_c, v_gas, v_disc, v_bulge, v_DM, v_5
    else:
        return v_c


def mg_acceleration(theta, theta_dict, galaxy, **kwargs):

    if kwargs['fR_parameter'] == 'free':
        fR0 = 10**theta[theta_dict['fR0']]
    elif kwargs['fR_parameter'] == 'fixed':
        fR0 = 10**kwargs['log10fR0']
    else:
        raise KeyError

    # set up grid
    dim = kwargs['MG_grid_dim']
    if dim == 2:
        grid = grid_2D(ngrid=175, rmin=0.05*kpc, rmax=5*Mpc)
        grid.set_cosmology(h=h, omega_m=omega_m, redshift=0)
        grid.rho = np.zeros((grid.ngrid, grid.nth), dtype=np.float64)
    elif dim == 1:
        grid = grid_1D(ngrid=175, rmin=0.05*kpc, rmax=5*Mpc)
        grid.set_cosmology(h=h, omega_m=omega_m, redshift=0)
        grid.rho = np.zeros((grid.ngrid,), dtype=np.float64)
    else:
        raise KeyError

    # add density profiles
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

    # solve
    grid.iter_solve(niter=1000000, F0=-fR0)

    # calculate acceleration on grid
    dfdr = np.zeros(grid.ngrid+1)
    if dim == 2:
        ind = grid.disc_ind
        dfdr[1:-1] = np.diff(grid.fR[:, ind])/(grid.rout[:-1]*grid.dx)
    else:
        dfdr[1:-1] = np.diff(grid.fR)/(grid.rout[:-1]*grid.dx)
    a_5 = np.interp(galaxy.R, np.append(grid.rin[0], grid.rout)/kpc, -dfdr)
    a_5 = 0.5*clight**2*a_5

    return a_5
