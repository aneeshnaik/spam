#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created: 20th March 2019
Author: A. P. Naik
Description: init file of 'fit' submodule.
"""
import sys
import time
import emcee
import numpy as np
from scipy.constants import G
from scipy.constants import parsec as pc
from .v_model import v_model
from .priors import lnprior

# physical constants
delta = 93.6
h = 0.7
H0 = h*100*1000/(1e+6*pc)
Msun = 1.989e+30


class GalaxyFit:
    """
    Class which prepares and executes MCMC fits of SPARC galaxies.


    Parameters
    ----------
    galaxy : spam.data.SPARCGalaxy
        Instance of class spam.data.SPARCGalaxy, containing galaxy to be fit.
    nwalkers : int
        Number of MCMC walkers. Default is 30.
    ntemps : int
        Number of MCMC temperatures. Default is 4.
    **model : str, {'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H0'-'H19', 'I0'-'I19'}
        Which preset model to use for fit. All settings below are prechosen.
        See Table 1 in Naik et al. (2019) for details about all models. If
        choosing a model, no other kwargs should be provided.
    **fR : bool
        Whether an f(R) gravity 5th force is included. Default is False.
    **halo_type : str, {'NFW', 'DC14'}
        Whether DM haloes are modelled as NFW or DC14. Default is 'NFW'.
    **upsilon : str, {'single', 'double', 'fixed'}
        Whether to have a single free parameter for the mass-to-light ratio,
        two free parameters, or fixed empirical values. Default is 'single'.
    **baryon_bound : bool
        Whether baryon fraction in galaxies is capped at the cosmic baryon
        fraction. Default is True.
    **scaling_priors : bool
        Whether stellar mass / halo mass and concentration / halo mass
        relations from simulations are used as priors. Otherwise flat priors.
        Default is True.
    **infer_errors : bool
        Whether additional error component is added in quadrature to observed
        error. Additional component is then a free parameter in the fit.
        Default is True.
    **fR_parameter : str, {'free', 'fixed'}
        If fR is True, then whether fR0 is a free parameter, or an imposed
        value. Default is 'free'.
    **log10fR0 : float
        If fR is True and fR_parameter is 'fixed', then the (log10 of the)
        value of the imposed fR0. For example, for F6, log10fR0 would be -6.0.
    **stellar_screening : bool
        If fR is True, then whether stellar screening is included. If True,
        then stars are excluded as sources in the scalar field solver. Default
        is False.
    **env_screening : bool
        If fR is True, then whether environmental screening is included. If
        true, then large scale structure is added to scalar field solver.
        Default is False.
    **MG_grid_dim : int, {1, 2}
        Whether scalar field solver is 1D or 2D. Default is 1D.

    Attributes
    ----------
    All parameters described above are stored as attributes, in addition to:

    name : str
        Name of galaxy.
    ndim : int
        Number of dimensions of parameter space.
    prior_bounds_lower : numpy.ndarray, shape (ndim,)
        Lower bounds of priors on all free parameters, in order given by
        self.theta_dict.
    prior_bounds_upper : numpy.ndarray, shape (ndim,)
        Upper bounds of priors on all free parameters, in order given by
        self.theta_dict.
    theta_dict : dict
        Keys are names of free parameters, and values are indices. Indices are
        used, for example, in the stored Markov chains.
    time_taken : float
        Time elapsed so far on MCMC.

    Methods
    -------
    __init__ :
        Initialise an instance of GalaxyFit.
    read_kwargs :
        Internal method to parse kwargs fed to __init__
    initialise_pos :
        Randomly find initial positions for the MCMC walkers.
    iterate :
        Run the MCMC.

    """
    def __init__(self, galaxy, nwalkers=30, ntemps=4, **kwargs):
        """
        Initialise an instance of GalaxyFit class, see GalaxyFit docstring for
        more info.
        """

        # store parameters
        self.galaxy = galaxy
        self.name = galaxy.name
        self.nwalkers = nwalkers
        self.ntemps = ntemps

        # read and store kwargs
        self.read_kwargs(kwargs)

        # assign number of free parameters ndim, as well as lower and upper
        # bounds on priors lb and ub. Start with 2 NFW, then add 0, 1, or 2
        # parameter(s) for M/L ratio, then 0 or 1 parameter for fR0, finally
        # sigma_g
        # Prior bounds are same as Katz et al except fR0 and sigma_g
        ndim = 2
        lb = np.array([4, 0])
        ub = np.array([5.7, 2])
        theta_dict = {'V_vir': 0, 'c_vir': 1}
        if self.upsilon == 'single':
            ndim += 1
            lb = np.append(lb, -0.52)
            ub = np.append(ub, -0.1)
            theta_dict['ML_disc'] = 2
            theta_dict['ML_bulge'] = 2
        elif self.upsilon == 'double':
            ndim += 2
            lb = np.append(lb, [-0.52, -0.52])
            ub = np.append(ub, [-0.1, -0.1])
            theta_dict['ML_disc'] = 2
            theta_dict['ML_bulge'] = 3
        if 'fR_parameter' in self.__dict__.keys():
            if self.fR_parameter == 'free':
                ndim += 1
                lb = np.append(lb, -9)
                ub = np.append(ub, np.log10(2e-6))
                theta_dict['fR0'] = ndim-1
        if self.infer_errors:
            ndim += 1
            lb = np.append(lb, 0)
            ub = np.append(ub, 2*galaxy.v_err.max())
            theta_dict['sigma_gal'] = ndim-1
        assert lb.shape == ub.shape == (ndim,)
        self.ndim = ndim
        self.prior_bounds_lower = lb
        self.prior_bounds_upper = ub
        self.theta_dict = theta_dict

        # time elapsed on MCMC so far
        self.time_taken = 0

        return

    def read_kwargs(self, kwargs):
        """Internal method to parse kwargs fed to __init__"""

        # check kwargs all understood
        acceptable_kwargs = ['model', 'fR', 'halo_type', 'upsilon',
                             'baryon_bound', 'scaling_priors',
                             'infer_errors', 'fR_parameter',
                             'log10fR0', 'stellar_screening',
                             'env_screening', 'MG_grid_dim']
        for k in kwargs.keys():
            if k not in acceptable_kwargs:
                raise KeyError("Unrecognised key: "+k+" in GalaxyFit")

        # apply settings for preset models
        if 'model' in kwargs.keys():
            if len(kwargs.keys()) > 1:
                raise KeyError("Additional kwargs with preset model")
            self.model = kwargs['model']

            model_list = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
            for i in range(20):
                model_list.append('H'+str(i))
                model_list.append('I'+str(i))
            if self.model not in model_list:
                raise ValueError("Unrecognised model!")

            if self.model in ['A', 'G']:
                self.fR = False
            else:
                self.fR = True
                self.MG_grid_dim = 1

            if self.model not in ['E', 'F']:
                self.upsilon = 'single'
            elif self.model == 'E':
                self.upsilon = 'fixed'
            else:
                self.upsilon = 'double'

            if self.model[0] in ['G', 'I']:
                self.halo_type = 'DC14'
            else:
                self.halo_type = 'NFW'

            if self.fR is True:
                if self.model == 'C':
                    self.env_screening = True
                else:
                    self.env_screening = False

                if self.model == 'D':
                    self.stellar_screening = True
                else:
                    self.stellar_screening = False

                if self.model[0] not in ['H', 'I']:
                    self.fR_parameter = 'free'
                else:
                    self.fR_parameter = 'fixed'
                    ind = int(self.model[1:])
                    vals = np.linspace(np.log10(1.563e-8), np.log10(2e-6), 20)
                    self.log10fR0 = vals[ind]

            self.baryon_bound = True
            self.scaling_priors = True
            self.infer_errors = True

        # read in custom settings if model not preset
        else:

            self.fR = kwargs.get('fR', False)
            self.halo_type = kwargs.get('halo_type', 'NFW')
            if self.halo_type not in ['NFW', 'DC14']:
                raise ValueError("Unrecognised halo type!")
            self.upsilon = kwargs.get('upsilon', 'single')
            if self.upsilon not in ['single', 'fixed', 'double']:
                raise ValueError("Unrecognised ML ratio prescription!")
            self.baryon_bound = kwargs.get('baryon_bound', True)
            self.scaling_priors = kwargs.get('scaling_priors', True)
            self.infer_errors = kwargs.get('infer_errors', True)

            # f(R)-specific settings
            if self.fR:

                self.fR_parameter = kwargs.get('fR_parameter', 'free')
                if self.fR_parameter not in ['fixed', 'free']:
                    raise ValueError("fR_parameter_type should be"
                                     "'fixed' or 'free'")

                if self.fR_parameter == 'fixed':
                    if 'log10fR0' in kwargs.keys():
                        self.log10fR0 = kwargs['log10fR0']
                        if self.log10fR0 > -5.6 or self.log10fR0 < -9:
                            raise ValueError("Invalid value for fR0.")
                    else:
                        raise KeyError("Need 'log10fR0' if fixing fR0")
                else:
                    if 'log10fR0' in kwargs.keys():
                        raise KeyError("No 'log10fR0' if varying fR0")

                self.stellar_screening = kwargs.get('stellar_screening', False)
                self.env_screening = kwargs.get('env_screening', False)
                self.MG_grid_dim = kwargs.get('MG_grid_dim', 1)
                if self.MG_grid_dim not in [1, 2]:
                    raise ValueError("Require 1 or 2 for MG_grid_dim")

            # check no f(R) settings given for LCDM run
            else:
                bad_kwargs = ['fR_parameter', 'log10fR0', 'stellar_screening',
                              'env_screening', 'MG_grid_dim']
                for k in bad_kwargs:
                    if k in kwargs.keys():
                        raise KeyError(k+" specified in LCDM run.")

        return

    def initialise_pos(self):
        """Randomly find initial positions for the MCMC walkers."""

        lb = self.prior_bounds_lower
        ub = self.prior_bounds_upper

        # initial positions, random within prior bounds
        d = ub-lb
        p0 = lb + d*np.random.rand(self.ntemps, self.nwalkers, self.ndim)

        # making sure all initial postions are not in region forbidden by
        # cosmic baryon bound; shifting them if they are
        if self.baryon_bound:
            for i in range(self.ntemps):
                for j in range(self.nwalkers):

                    bound_violated = True
                    while bound_violated:
                        theta = p0[i, j]
                        if self.upsilon == 'fixed':
                            ML = 0.5
                        else:
                            ML = 10**theta[self.theta_dict['ML_disc']]
                        V_vir = 10**theta[self.theta_dict['V_vir']]
                        M_halo = (V_vir**3)/(np.sqrt(delta/2)*G*H0)
                        M_star = ML*1e+9*self.galaxy.luminosity_tot*Msun
                        M_gas = (4/3)*self.galaxy.HI_mass
                        if (M_star+M_gas)/M_halo > 0.2:
                            p0[i, j] = lb + d*np.random.rand(self.ndim)
                        else:
                            bound_violated = False

        return p0

    def iterate(self, niter, threads=1, initial_run=True, print_progress=True):
        """
        Run MCMC.

        Parameters
        ----------
        niter : int
            Number of MCMC iterations per walker.
        threads : int
            Number of CPU threads to run MCMC on. Default is 1.
        initial_run : bool
            Whether the present run is the first of the MCMC, or if it is a
            continuation of a previous run. In the latter case, initial
            positions of the walkers are set to last point on the chain.
        print_progress : bool
            Whether to output percentage completion every 100 iterations
        """

        # kwargs for prior and likelihood functions
        kwargs = {'theta_dict': self.theta_dict, 'halo_type': self.halo_type,
                  'upsilon': self.upsilon, 'infer_errors': self.infer_errors,
                  'galaxy': self.galaxy, 'scaling_priors': self.scaling_priors,
                  'baryon_bound': self.baryon_bound, 'fR': self.fR,
                  'prior_bounds_lower': self.prior_bounds_lower,
                  'prior_bounds_upper': self.prior_bounds_upper}
        if self.fR:
            kwargs['fR_parameter'] = self.fR_parameter
            if self.fR_parameter == 'fixed':
                kwargs['log10fR0'] = self.log10fR0,
            kwargs['stellar_screening'] = self.stellar_screening
            kwargs['env_screening'] = self.env_screening
            kwargs['MG_grid_dim'] = self.MG_grid_dim

        # initialise sampler
        sampler = emcee.PTSampler(self.ntemps, self.nwalkers, self.ndim,
                                  lnlike, lnprior, threads=threads,
                                  loglkwargs=kwargs, logpkwargs=kwargs)

        # either initialise walker positions or continue existing chain
        if initial_run:
            self.niter = niter
            p0 = self.initialise_pos()
        else:
            self.niter += niter
            p0 = self.chain[:, :, -1, :]

        # run MCMC
        t0 = time.time()
        for i, result in enumerate(sampler.sample(p0, iterations=niter)):
            if (i+1) % 100 == 0:
                sys.stdout.write("{0:5.1%}\n".format(float(i+1) / niter))
                sys.stdout.flush()
        t1 = time.time()
        self.time_taken += t1-t0

        # save chain
        if initial_run:
            self.chain = sampler.chain
            self.flatchain = sampler.flatchain
            self.lnprob = sampler.lnprobability
        else:
            self.chain = np.dstack((self.chain, sampler.chain))
            self.flatchain = np.hstack((self.flatchain, sampler.flatchain))
            self.lnprob = np.dstack((self.lnprob, sampler.lnprobability))
        return


def lnlike(theta, **kwargs):
    """
    Gaussian log-likelihood function

    """
    model = v_model(theta=theta, theta_dict=theta_dict, galaxy=galaxy,
                    fR_parameter=fR_parameter, log10fR0=log10fR0,
                    Rscr_fixed=Rscr_fixed,
                    halo_type=halo_type,
                    StellarScreening=StellarScreening,
                    EnvScreening=EnvScreening,
                    ML_ratio=ML_ratio, MGGridDim=MGGridDim)
    if kwargs['infer_errors']:
        sigma = np.sqrt(galaxy.v_err**2 + theta[theta_dict['sigma_gal']]**2)
        lnlike = -np.sum(0.5*((galaxy.v - model)/sigma)**2 + np.log(sigma))
    else:
        lnlike = -0.5*np.sum(((galaxy.v - model)/galaxy.v_err)**2)

    return lnlike
