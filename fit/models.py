#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created: 2018
Author: A. P. Naik
Description: File containing mass models used in sparc fitting, particularly
to feed into the 2D solver, but also the dark matter profile used to calculate
Newtonian contribution to rotation curve models.

Functions
---------
top_hat :
    Top hat density profile for environmental contribution to scalar field.
gas_disc :
    Exponential profile for gas disc.
stellar_disc :
    Exponential profile for stellar disc.
stellar_bulge :
    Hernquist profile for stellar bulge.
DM_halo :
    NFW or DC14 profile for dark matter halo.
halo_parameters :
    For given c_vir/v_vir, calculate halo parameters rho_0/R_s.
stellar_mass_frac :
    For given parameters, calculate stellar mass fraction of the galaxy.
halo_v_circ :
    Calculate DM halo contribution to rotation curve, v_DM.
DC14_pars :
    For given stellar mass fraction, calculate DC14 alpha, beta, gamma.
hypergeom_A1 :
    'A1' function from appendix of Katz et al. 2018 (arxiv.org/abs/1808.00971)
"""
import numpy as np
from scipy.special import hyp2f1
from scipy.constants import G, parsec as pc
from .solvers import Grid1D, Grid2D

# physical constants and cosmology; all SI units
kpc = 1e+3*pc
delta = 93.6
h = 0.7
H0 = h*100*1000/(1e+6*pc)
omega_m = 0.308
rho_c = 3*H0**2/(8*np.pi*G)
Msun = 1.989e+30


def top_hat(radius, phi_ext, grid):
    """
    Returns a top-hat density profile for environmental contribution to scalar
    field. Density calculated from given radius and potential according to
    equation 2.43 in Binney+Tremaine, 2nd ed. Density is then calculated on
    grid for scalar field solver.

    Parameters
    ----------
    radius : float
        Radius of top hat profile. UNITS: metres.
    phi_ext : float
        Gravitational potential of top hat. UNITS: m^2/s^2
    grid : .solvers.grid_1D or grid_2D
        Discretised grid on which density is to be computed.

    Returns
    -------
    rho : numpy.ndarray (shape equal to 'grid.grid_shape')
        Density of structure. UNITS: kg/m^3
    """
    # density of profile; kg/m^3
    rho_0 = phi_ext/(2*np.pi*G*radius**2)

    # calculate density on grid
    rho = np.zeros_like(grid.rgrid)
    rho[np.where(grid.rgrid < radius)] = rho_0

    # check shape
    if not rho.shape == grid.grid_shape:
        raise ValueError("Incorrect shape for rho")

    return rho


def gas_disc(galaxy, grid):
    """
    Returns an exponential density profile for gas disc of a given galaxy.
    Scale radius is given by the best fit radius calculated by
    spam.data.fit_gas_disc). Scale height is 0.1 times scale radius. Central
    density calculated from total HI mass galaxy.HI_mass. Density is then
    calculated on grid for scalar field solver.

    Parameters
    ----------
    galaxy : spam.data.SPARCGalaxy
        Instance of class spam.data.SPARCGalaxy for given galaxy.
    grid : .solvers.grid_1D or grid_2D
        Discretised grid on which density is to be computed.

    Returns
    -------
    rho : numpy.ndarray (shape equal to 'grid.grid_shape')
        Density of structure. UNITS: kg/m^3
    """
    # calculate disc parameters
    R_d = galaxy.gas_radius  # metres
    sigma_0 = 2*galaxy.HI_mass/(3*np.pi*R_d**2)  # kg/m^2
    height = 0.1*R_d  # metres

    # create 2D grid on which to calculate density (in case 'grid' is 1D)
    temp_grid = Grid2D(ngrid=grid.ngrid, rmin=grid.rmin, rmax=grid.rmax)
    radgrid = temp_grid.rgrid*np.sin(temp_grid.thgrid)
    zgrid = temp_grid.rgrid*np.cos(temp_grid.thgrid)
    rho_0 = sigma_0/(2*height)  # kg/m^3
    rho = rho_0*np.exp(-radgrid/R_d)*np.exp(-np.abs(zgrid)/height)

    # integrate polar coordinate if grid is 1D
    if type(grid) == Grid1D:
        rho = np.sum(rho*temp_grid.dvol, axis=-1)/grid.dvol

    # check shape
    if not rho.shape == grid.grid_shape:
        raise ValueError("Incorrect shape for rho")

    return rho


def stellar_disc(theta, theta_dict, galaxy, upsilon, grid):
    """
    Returns an exponential density profile for stellar disc of a given galaxy.
    Radius and central density given by best fit parameters from
    spam.fit.fit_stellar_disc. Scale height calculated from scale radius
    according to relation given in section 3.3 of the original SPARC paper,
    i.e. h = 0.196 * (R_d)**0.633. Density is then calculated on grid for
    scalar field solver.

    Parameters
    ----------
    theta : numpy.ndarray, shape (ndim,)
        Parameter values for which to calculate stellar disc.
    theta_dict : dict
        Keys are names of free parameters, and values are indices. Indices are
        used, for example, in the stored Markov chains.
    galaxy : spam.data.SPARCGalaxy
        Instance of class spam.data.SPARCGalaxy for given galaxy.
    upsilon : str, {'single', 'double', 'fixed'}
        Whether to have a single free parameter for the mass-to-light ratio,
        two free parameters, or fixed empirical values.
    grid : .solvers.grid_1D or grid_2D
        Discretised grid on which density is to be computed.

    Returns
    -------
    rho : numpy.ndarray (shape equal to 'grid.grid_shape')
        Density of structure. UNITS: kg/m^3
    """
    # determine mass-to-light ratio
    if upsilon == 'fixed':
        ML_disc = 0.5
    else:
        ML_disc = 10**theta[theta_dict['ML_disc']]

    # get disc parameters
    sigma_0 = ML_disc*galaxy.stellar_expdisc_sigma_0  # kg/m^2
    R_d = galaxy.stellar_expdisc_R_d  # metres
    height = 0.196 * (R_d/kpc)**0.633 * kpc  # metres

    # create 2D grid on which to calculate density (in case 'grid' is 1D)
    temp_grid = Grid2D(ngrid=grid.ngrid, rmin=grid.rmin, rmax=grid.rmax)
    radgrid = temp_grid.rgrid*np.sin(temp_grid.thgrid)
    zgrid = temp_grid.rgrid*np.cos(temp_grid.thgrid)
    rho_0 = sigma_0/(2*height)  # kg/m^3
    rho = rho_0*np.exp(-radgrid/R_d)*np.exp(-np.abs(zgrid)/height)

    # integrate polar coordinate if grid is 1D
    if type(grid) == Grid1D:
        rho = np.sum(rho*temp_grid.dvol, axis=-1)/grid.dvol

    # check shape
    if not rho.shape == grid.grid_shape:
        raise ValueError("Incorrect shape for rho")

    return rho


def stellar_bulge(theta, theta_dict, galaxy, upsilon, grid):
    """
    Returns a Hernquist profile for stellar bulge of a given galaxy. Radius and
    central density given by best fit parameters from
    spam.fit.fit_stellar_bulge. Density is then calculated on grid for scalar
    field solver.

    Parameters
    ----------
    theta : numpy.ndarray, shape (ndim,)
        Parameter values for which to calculate stellar disc.
    theta_dict : dict
        Keys are names of free parameters, and values are indices. Indices are
        used, for example, in the stored Markov chains.
    galaxy : spam.data.SPARCGalaxy
        Instance of class spam.data.SPARCGalaxy for given galaxy.
    upsilon : str, {'single', 'double', 'fixed'}
        Whether to have a single free parameter for the mass-to-light ratio,
        two free parameters, or fixed empirical values.
    grid : .solvers.grid_1D or grid_2D
        Discretised grid on which density is to be computed.

    Returns
    -------
    rho : numpy.ndarray (shape equal to 'grid.grid_shape')
        Density of structure. UNITS: kg/m^3
    """
    # check if stellar bulge present in galaxy
    if not galaxy.StellarBulge:
        rho = np.zeros(grid.grid_shape, dtype=np.float64)
        return rho

    # determine mass-to-light ratio
    if upsilon == 'fixed':
        ML_bulge = 0.7
    else:
        ML_bulge = 10**theta[theta_dict['ML_bulge']]

    # get hernquist parameters
    a = galaxy.hernquist_radius
    rho_0 = galaxy.hernquist_rho_0

    # calculate density
    x = grid.rgrid/a
    rho = ML_bulge*rho_0/(x*(1+x)**3)

    # check shape
    if not rho.shape == grid.grid_shape:
        raise ValueError("Incorrect shape for rho")

    return rho


def DM_halo(theta, theta_dict, galaxy, halo_type, upsilon, grid):
    """
    Returns a density profile, either NFW or DC14, for given parameters and
    given galaxy. Density is calculated on grid for scalar field solver.

    Parameters
    ----------
    theta : numpy.ndarray, shape (ndim,)
        Parameter values for which to calculate stellar disc.
    theta_dict : dict
        Keys are names of free parameters, and values are indices. Indices are
        used, for example, in the stored Markov chains.
    galaxy : spam.data.SPARCGalaxy
        Instance of class spam.data.SPARCGalaxy for given galaxy.
    halo_type : str, {'NFW', 'DC14'}
        Whether DM haloes are modelled as NFW or DC14.
    upsilon : str, {'single', 'double', 'fixed'}
        Whether to have a single free parameter for the mass-to-light ratio,
        two free parameters, or fixed empirical values.
    grid : .solvers.grid_1D or grid_2D
        Discretised grid on which density is to be computed.

    Returns
    -------
    rho : numpy.ndarray (shape equal to 'grid.grid_shape')
        Density of structure. UNITS: kg/m^3
    """
    # calculate density normalisation and scale radius
    rho_0, R_s = halo_parameters(theta, theta_dict, galaxy,
                                 halo_type, upsilon)

    # calculate density
    if halo_type == 'NFW':
        r = grid.rgrid/R_s
        rho = rho_0/(r*(1+r)**2)
    elif halo_type == 'DC14':
        r = grid.rgrid/R_s
        X = stellar_mass_frac(theta, theta_dict, galaxy, upsilon)
        alpha, beta, gamma = DC14_pars(X=X)
        rho = rho_0/((r**gamma)*((1+r**alpha)**((beta-gamma)/alpha)))

    # check shape
    if not rho.shape == grid.grid_shape:
        raise ValueError("Incorrect shape for rho")

    return rho


def halo_parameters(theta, theta_dict, galaxy, halo_type, upsilon):
    """
    For given fit parameters (inc. c_vir and v_vir), returns density
    normalisation rho_0 and scale radius R_s. Details of calculation can be
    found in appendix of Katz et al. 2018 (arxiv.org/abs/1808.00971).

    Parameters
    ----------
    theta : numpy.ndarray, shape (ndim,)
        Parameter values for which to calculate stellar disc.
    theta_dict : dict
        Keys are names of free parameters, and values are indices. Indices are
        used, for example, in the stored Markov chains.
    galaxy : spam.data.SPARCGalaxy
        Instance of class spam.data.SPARCGalaxy for given galaxy.
    halo_type : str, {'NFW', 'DC14'}
        Whether DM haloes are modelled as NFW or DC14.
    upsilon : str, {'single', 'double', 'fixed'}
        Whether to have a single free parameter for the mass-to-light ratio,
        two free parameters, or fixed empirical values.

    Returns
    -------
    rho_0 : float
        Density normalisation of halo profile. UNITS: kg/m^3
    R_s : float
        Scale radius of density profile. UNITS: m
    """

    # get fit halo parameters
    V_vir = 10**theta[theta_dict['V_vir']]
    c_vir = 10**theta[theta_dict['c_vir']]

    # calculate virial mass and radius
    M_vir = (V_vir**3)/(np.sqrt(delta/2)*G*H0)
    R_vir = (3*M_vir/(4*np.pi*delta*rho_c))**(1/3)

    # calculate alpha, beta, gamma parameters for either halo type
    if halo_type == 'NFW':
        alpha, beta, gamma = 1, 3, 1
    elif halo_type == 'DC14':
        X = stellar_mass_frac(theta, theta_dict, galaxy, upsilon)
        alpha, beta, gamma = DC14_pars(X)

    # calculate scale radius
    R_s = R_vir/(c_vir*((2-gamma)/(beta-2))**(1/alpha))

    # calculate rho_0
    if halo_type == 'NFW':
        denom = 4*np.pi*R_s**3*(np.log(1+c_vir) - (c_vir/(1+c_vir)))
    elif halo_type == 'DC14':
        denom = 4*np.pi*R_vir*hypergeom_A1(R_vir, R_s, alpha, beta, gamma)
    rho_0 = M_vir/denom

    return rho_0, R_s


def stellar_mass_frac(theta, theta_dict, galaxy, upsilon):
    """
    For given fit parameters (theta) and given galaxy, returns stellar mass
    fraction, i.e. log_10(M_star/M_halo).

    Parameters
    ----------
    theta : numpy.ndarray, shape (ndim,)
        Parameter values for which to calculate stellar disc.
    theta_dict : dict
        Keys are names of free parameters, and values are indices. Indices are
        used, for example, in the stored Markov chains.
    galaxy : spam.data.SPARCGalaxy
        Instance of class spam.data.SPARCGalaxy for given galaxy.
    upsilon : str, {'single', 'double', 'fixed'}
        Whether to have a single free parameter for the mass-to-light ratio,
        two free parameters, or fixed empirical values.

    Returns
    -------
    X : float
        Stellar mass fraction log_10 (M_star / M_halo)
    """

    # determine mass-to-light ratio
    if upsilon == 'fixed':
        ML = 0.5
    else:
        ML = 10**theta[theta_dict['ML_disc']]

    # calculate halo (virial) mass
    V_vir = 10**theta[theta_dict['V_vir']]
    M_vir = (V_vir**3)/(np.sqrt(delta/2)*G*H0)

    # calculate stellar mass according to total stellar luminosity
    M_star = ML*1e+9*galaxy.luminosity_tot*Msun

    # calculate stellar mass fraction
    X = np.log10(M_star/M_vir)

    return X


def halo_v_circ(R, theta, theta_dict, galaxy, halo_type, upsilon):
    """
    For given fit parameters, returns v_DM, i.e. dark matter contribution to
    rotation curve. To calculate at same radii as observed rotation curve,
    set R equal to galaxy.R (after converting kpc to metres).

    Parameters
    ----------
    R : float or one-dimensional numpy.ndarray
        Radius or radii at which to calculate v_DM. UNITS: m
    theta : numpy.ndarray, shape (ndim,)
        Parameter values for which to calculate stellar disc.
    theta_dict : dict
        Keys are names of free parameters, and values are indices. Indices are
        used, for example, in the stored Markov chains.
    galaxy : spam.data.SPARCGalaxy
        Instance of class spam.data.SPARCGalaxy for given galaxy.
    halo_type : str, {'NFW', 'DC14'}
        Whether DM haloes are modelled as NFW or DC14.
    upsilon : str, {'single', 'double', 'fixed'}
        Whether to have a single free parameter for the mass-to-light ratio,
        two free parameters, or fixed empirical values.

    Returns
    -------
    v_DM : float or numpy.ndarray, shape same as R
        Dark matter contribution to rotation curve at radius/radii R. UNITS:
        km/s
    """

    # calculate density normalisation and scale radius
    rho_0, R_s = halo_parameters(theta, theta_dict, galaxy,
                                 halo_type, upsilon)

    # calculate enclosed mass at radius R; in DC14 case this is Eq. A11 in
    # Katz et al 2018, (arxiv.org/abs/1808.00971).
    if halo_type == 'NFW':
        M_s = 4*np.pi*rho_0*R_s**3
        M_enc = M_s*(np.log((R_s+R)/R_s) - (R/(R_s+R)))
    elif halo_type == 'DC14':
        X = stellar_mass_frac(theta, theta_dict, galaxy, upsilon)
        alpha, beta, gamma = DC14_pars(X)
        M_enc = 4*np.pi*rho_0*R*hypergeom_A1(R, R_s, alpha, beta, gamma)

    # calculate v_DM
    v_DM = np.sqrt(G*M_enc/R)
    return v_DM


def DC14_pars(X):
    """
    For given stellar mass fraction X, returns DC14 alpha, beta gamma.

    Parameters
    ----------
    X : float or 1D numpy.ndarray
        Stellar mass fraction log_10 (M_star / M_halo), as given by e.g.
        spam.fit.models.stellar_mass_frac function.

    Returns
    -------
    alpha, beta, gamma: floats or 1D numpy.ndarray objects, same shapes as X
        alpha, beta, and gamma parameters, as defined in original DC14 paper.
    """

    # fix X beyond limits; different treatment for floats and arrays
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

    # calculate alpha, beta, gamma
    Y1 = 10**(X_copy+2.33)
    Y2 = 10**(X_copy+2.56)
    alpha = 2.94 - np.log10(Y1**-1.08+Y1**2.29)
    beta = 4.23+1.34*X_copy+0.26*X_copy**2
    gamma = -0.06 + np.log10(Y2**-0.68+Y2)
    return alpha, beta, gamma


def hypergeom_A1(R, R_s, alpha, beta, gamma):
    """
    As a function of radius R, returns 'A1' function as defined in appendix of
    Katz et al. 2018 (arxiv.org/abs/1808.00971).

    Parameters
    ----------
    R : float or 1D numpy.ndarray
        Radius or radii at which to calculate A1. UNITS: m
    R_s : float
        Halo scale radius. UNITS: m
    alpha, beta, gamma : floats
        alpha, beta, gamma parameters as defined in original DC14 paper.

    Returns
    -------
    A1 : float or 1D numpy.ndarray, same shape as R
        A1 function, as defined in appendix of Katz et al. 2018
        (arxiv.org/abs/1808.00971).
    """

    # arguments for hypergeometric function
    x1 = (3-gamma)/alpha
    x2 = (beta-gamma)/alpha
    x3 = (3+alpha-gamma)/alpha
    z = -(R/R_s)**alpha

    # calculate A1
    A1 = -R**(2-gamma)*R_s**gamma*hyp2f1(x1, x2, x3, z)/(gamma-3)
    return A1
