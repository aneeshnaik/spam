#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created: Summer 2018
Author: A. P. Naik
Description: Routine to calculate circular velocity profile in SPARC galaxy for
given set of parameters. This file also contains the DM halo model, and the
gaseous and stellar disc models.
"""
import MG_solvers as MG
import numpy as np
import sparc_mass_models as models
from scipy.constants import parsec as pc
from scipy.constants import c as clight
kpc = 1e+3*pc
Mpc = 1e+6*pc
h = 0.7
omega_m = 0.308


def v_model(theta, theta_dict, galaxy, fR_parameter, log10fR0, Rscr_fixed,
            halo_type, StellarScreening, EnvScreening, ML_ratio,
            MGGridDim, component_split=False):
    """
    Calculate v_circ for parameter set contained in theta, on SPARC galaxy
    contained in 'galaxy' argument. Fifth force is optionally included by
    setting fR=True.
    """
    fR_options = ['fR0', 'Rscr', 'fixed', 'Rscr_fixed', None]
    assert fR_parameter in fR_options, "unrecognised fR"
    if fR_parameter == 'fixed':
        assert MGGridDim in ['1D', '2D'], "Need to specify MG solver type"
        assert type(log10fR0) == float, "fR0 should be float"
        assert log10fR0 >= -9, "fR0 needs to be greater than 1e-9"
        assert log10fR0 <= np.log10(2e-6), "fR0 needs to be less than 2e-6"
        assert Rscr_fixed is None, "don't fix Rscr if fixing fR0!"
    elif fR_parameter == 'fR0':
        assert MGGridDim in ['1D', '2D'], "Need to specify MG solver type"
        assert log10fR0 is None, "Don't specify fR0 if using varying it!"
        assert Rscr_fixed is None, "don't fix Rscr if varying fR0!"
    elif fR_parameter == 'Rscr_fixed':
        assert log10fR0 is None, "Don't specify fR0 if fixing Rscr!"
        assert MGGridDim is None, "Only specify MGGridDim if fitting fR0"
        assert type(Rscr_fixed) in [float, np.float32, np.float64], "Rscr should be float"
        assert Rscr_fixed >= 0
        assert Rscr_fixed < 1.05*galaxy.R[-1]
    else:
        assert Rscr_fixed is None, "don't fix Rscr if varying fR0!"
        assert log10fR0 is None, "Don't specify fR0 if using varying it!"
        assert MGGridDim is None, "Only specify MGGridDim if fitting fR0"
    assert halo_type in ['NFW', 'DC14'], "unrecognised ML"
    assert ML_ratio in ['fixed', 'single', 'double'], "unrecognised ML"

    if ML_ratio == 'fixed':
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
                              halo_type, ML_ratio)

    # calculating Newtonian and MG accelerations
    a_N = (v_DM**2 + v_g**2 + ML_bulge*v_b**2 + ML_disc*v_d**2)/R
    a_5 = mg_acceleration(a_N=a_N, theta=theta, theta_dict=theta_dict,
                          galaxy=galaxy, fR_parameter=fR_parameter,
                          log10fR0=log10fR0, Rscr_fixed=Rscr_fixed,
                          halo_type=halo_type,
                          StellarScreening=StellarScreening,
                          EnvScreening=EnvScreening,
                          ML_ratio=ML_ratio,
                          MGGridDim=MGGridDim)

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


def mg_acceleration(a_N, theta, theta_dict, galaxy, fR_parameter, log10fR0,
                    Rscr_fixed, halo_type, StellarScreening, EnvScreening,
                    ML_ratio, MGGridDim):

    # set up and 2D or 1D solver
    if fR_parameter in ['fR0', 'fixed']:

        if fR_parameter == 'fR0':
            fR0 = 10**theta[theta_dict['fR0']]
        else:
            fR0 = 10**log10fR0

        # set up grid
        if MGGridDim == '2D':
            grid = MG.grid_2D(ngrid=175, rmin=0.05*kpc, rmax=5*Mpc)
            grid.set_cosmology(h=h, omega_m=omega_m, redshift=0)
            grid.rho = np.zeros((grid.ngrid, grid.nth), dtype=np.float64)
        else:
            grid = MG.grid_1D(ngrid=175, rmin=0.05*kpc, rmax=5*Mpc)
            grid.set_cosmology(h=h, omega_m=omega_m, redshift=0)
            grid.rho = np.zeros((grid.ngrid,), dtype=np.float64)

        # add density profiles
        grid.rho += models.gas_disc(galaxy, grid)
        if not StellarScreening:
            grid.rho += models.stellar_disc(theta, theta_dict, galaxy,
                                            ML_ratio, grid)
            grid.rho += models.stellar_bulge(theta, theta_dict, galaxy,
                                             ML_ratio, grid)
        grid.rho += models.DM_halo(theta, theta_dict, galaxy,
                                   halo_type, ML_ratio, grid)

        # add uniform sphere to account for external potential from scr. map
        if EnvScreening:
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
        if MGGridDim == '2D':
            ind = grid.disc_ind
            dfdr[1:-1] = np.diff(grid.fR[:, ind])/(grid.rout[:-1]*grid.dx)
        else:
            dfdr[1:-1] = np.diff(grid.fR)/(grid.rout[:-1]*grid.dx)
        a_5 = np.interp(galaxy.R, np.append(grid.rin[0], grid.rout)/kpc, -dfdr)
        a_5 = 0.5*clight**2*a_5

    # impose screening radius with analytic (spherical) 5th force profile
    elif fR_parameter in ['Rscr', 'Rscr_fixed']:

        if fR_parameter == 'Rscr':
            Rscr = theta[theta_dict['Rscr']]
        else:
            Rscr = Rscr_fixed

        if Rscr == 0:
            a_scr = 0
        else:
            a_scr = np.interp(Rscr, np.append(0, galaxy.R), np.append(0, a_N))

        a_5 = np.zeros_like(a_N)
        if Rscr < galaxy.R[-1]:
            inds = np.where(galaxy.R > Rscr)[0]
            mass_frac = (a_scr*Rscr**2)/(a_N[inds]*galaxy.R[inds]**2)
            a_5[inds] = (a_N[inds]/3)*(1 - mass_frac)

    else:
        a_5 = np.zeros_like(a_N)

    return a_5
