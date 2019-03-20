#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created: 21st September 2018
Author: A. P. Naik
Description: Mass models used in sparc fitting, particularly to feed into the
2D solver, but also the dark matter profile used to calculate Newtonian
contribution.
"""
import MG_solvers as MG
import numpy as np
from scipy.special import hyp2f1
from scipy.constants import G
from scipy.constants import parsec as pc
kpc = 1e+3*pc
delta = 93.6
h = 0.7
H0 = h*100*1000/(1e+6*pc)
omega_m = 0.308
rho_c = 3*H0**2/(8*np.pi*G)
Msun = 1.989e+30


def top_hat(r_grav, phi_ext, grid):

    a = 0.6*r_grav
    rho_0 = phi_ext/(2*np.pi*G*a**2)

    rho = np.zeros_like(grid.rgrid)
    rho[np.where(grid.rgrid < a)] = rho_0

    return rho


def gas_disc(galaxy, grid):

    R_d = galaxy.gas_radius  # metres
    sigma_0 = 2*galaxy.HI_mass/(3*np.pi*R_d**2)  # kg/m^2
    height = 0.1*R_d  # metres

    temp_grid = MG.grid_2D(ngrid=grid.ngrid, rmin=grid.rmin, rmax=grid.rmax)
    radgrid = temp_grid.rgrid*np.sin(temp_grid.thgrid)
    zgrid = temp_grid.rgrid*np.cos(temp_grid.thgrid)

    rho_0 = sigma_0/(2*height)  # kg/m^3
    rho = rho_0*np.exp(-radgrid/R_d)*np.exp(-np.abs(zgrid)/height)

    if type(grid) == MG.grid_1D:
        rho = np.sum(rho*temp_grid.dvol, axis=-1)/grid.dvol

    assert rho.shape == grid.grid_shape
    return rho


def stellar_disc(theta, theta_dict, galaxy, ML_ratio, grid):

    if ML_ratio == 'fixed':
        ML_disc = 0.5
    else:
        ML_disc = 10**theta[theta_dict['ML_disc']]

    sigma_0 = ML_disc*galaxy.stellar_expdisc_sigma_0  # kg/m^2
    R_d = galaxy.stellar_expdisc_R_d  # metres

    height = 0.196 * (R_d/kpc)**0.633 * kpc  # metres

    temp_grid = MG.grid_2D(ngrid=grid.ngrid, rmin=grid.rmin, rmax=grid.rmax)
    radgrid = temp_grid.rgrid*np.sin(temp_grid.thgrid)
    zgrid = temp_grid.rgrid*np.cos(temp_grid.thgrid)

    rho_0 = sigma_0/(2*height)  # kg/m^3
    rho = rho_0*np.exp(-radgrid/R_d)*np.exp(-np.abs(zgrid)/height)

    if type(grid) == MG.grid_1D:
        rho = np.sum(rho*temp_grid.dvol, axis=-1)/grid.dvol

    assert rho.shape == grid.grid_shape
    return rho


def stellar_bulge(theta, theta_dict, galaxy,  ML_ratio, grid):

    if not galaxy.StellarBulge:
        rho = np.zeros(grid.grid_shape, dtype=np.float64)
        return rho

    if ML_ratio == 'fixed':
        ML_bulge = 0.7
    else:
        ML_bulge = 10**theta[theta_dict['ML_bulge']]

    a = galaxy.hernquist_radius
    rho_0 = galaxy.hernquist_rho_0
    x = grid.rgrid/a
    rho = ML_bulge*rho_0/(x*(1+x)**3)

    assert rho.shape == grid.grid_shape
    return rho


def DM_halo(theta, theta_dict, galaxy, halo_type, ML_ratio, grid):

    rho_0, R_s = halo_parameters(theta, theta_dict, galaxy,
                                 halo_type, ML_ratio)

    if halo_type == 'NFW':
        r = grid.rgrid/R_s
        rho = rho_0/(r*(1+r)**2)  # kg/m^3
    elif halo_type == 'DC14':
        r = grid.rgrid/R_s
        X = stellar_mass_frac(theta, theta_dict, galaxy, ML_ratio)
        alpha, beta, gamma = DC14_pars(X=X)
        rho = rho_0/((r**gamma)*((1+r**alpha)**((beta-gamma)/alpha)))

    assert rho.shape == grid.grid_shape
    return rho


def halo_parameters(theta, theta_dict, galaxy, halo_type, ML_ratio):

    assert halo_type in ['NFW', 'DC14']

    V_vir = 10**theta[theta_dict['V_vir']]
    c_vir = 10**theta[theta_dict['c_vir']]

    M_vir = (V_vir**3)/(np.sqrt(delta/2)*G*H0)
    R_vir = (3*M_vir/(4*np.pi*delta*rho_c))**(1/3)

    if halo_type == 'NFW':
        alpha, beta, gamma = 1, 3, 1
    elif halo_type == 'DC14':
        X = stellar_mass_frac(theta, theta_dict, galaxy, ML_ratio)
        alpha, beta, gamma = DC14_pars(X)

    R_s = R_vir/(c_vir*((2-gamma)/(beta-2))**(1/alpha))

    if halo_type == 'NFW':
        denom = 4*np.pi*R_s**3*(np.log(1+c_vir) - (c_vir/(1+c_vir)))
    elif halo_type == 'DC14':
        denom = 4*np.pi*R_vir*hypergeom_A1(R_vir, R_s, alpha, beta, gamma)

    rho_0 = M_vir/denom

    return rho_0, R_s


def stellar_mass_frac(theta, theta_dict, galaxy, ML_ratio):

    if ML_ratio == 'fixed':
        ML = 0.5
    else:
        ML = 10**theta[theta_dict['ML_disc']]
    V_vir = 10**theta[theta_dict['V_vir']]

    M_vir = (V_vir**3)/(np.sqrt(delta/2)*G*H0)
    M_star = ML*1e+9*galaxy.luminosity_tot*Msun

    X = np.log10(M_star/M_vir)

    return X


def halo_v_circ(R, theta, theta_dict, galaxy, halo_type, ML_ratio):

    assert halo_type in ['NFW', 'DC14']
    rho_0, R_s = halo_parameters(theta, theta_dict, galaxy,
                                 halo_type, ML_ratio)

    if halo_type == 'NFW':
        M_s = 4*np.pi*rho_0*R_s**3
        M_enc = M_s*(np.log((R_s+R)/R_s) - (R/(R_s+R)))
    elif halo_type == 'DC14':
        X = stellar_mass_frac(theta, theta_dict, galaxy, ML_ratio)
        alpha, beta, gamma = DC14_pars(X)
        M_enc = 4*np.pi*rho_0*R*hypergeom_A1(R, R_s, alpha, beta, gamma)

    v_DM = np.sqrt(G*M_enc/R)

    return v_DM


def DC14_pars(X):

    if isinstance(X, np.ndarray):
        X_copy = X.copy()
        X_copy[np.where(X > -1.3)] = -1.3
        X_copy[np.where(X < -4.1)] = -4.1
    else:
        X_copy = X
        if X > -1.3:
            X_copy = -1.3
        elif X < -4.1:
            X_copy = -4.1

    Y1 = 10**(X_copy+2.33)
    Y2 = 10**(X_copy+2.56)
    alpha = 2.94 - np.log10(Y1**-1.08+Y1**2.29)
    beta = 4.23+1.34*X_copy+0.26*X_copy**2
    gamma = -0.06 + np.log10(Y2**-0.68+Y2)
    return alpha, beta, gamma


def hypergeom_A1(R, R_s, alpha, beta, gamma):

    x1 = (3-gamma)/alpha
    x2 = (beta-gamma)/alpha
    x3 = (3+alpha-gamma)/alpha
    z = -(R/R_s)**alpha

    A1 = -R**(2-gamma)*R_s**gamma*hyp2f1(x1, x2, x3, z)/(gamma-3)

    return A1
