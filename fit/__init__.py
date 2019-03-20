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
from v_model import v_model
from priors import lnprior
from scipy.constants import G
from scipy.constants import parsec as pc
delta = 93.6
h = 0.7
H0 = h*100*1000/(1e+6*pc)
Msun = 1.989e+30


class GalaxyFit:

    def __init__(self, galaxy, nwalkers=20, ntemps=5, fR=False,
                 halo_type='NFW', upsilon='single', baryon_bound=True,
                 scaling_priors=True, infer_errors=True, **kwargs):
        """
        fR_parameter, log10fR0=None,
                 , StellarScreening=False, EnvScreening=False,
                  MGGridDim=None):
        """
        """
        Initialises an instance of SoloGalFit class, i.e. creates a fit for
        a galaxy, with options specified.
        """

        self.galaxy = galaxy
        #self.name = galaxy.name
        self.nwalkers = nwalkers
        self.ntemps = ntemps
        self.fR = fR

        # check keywords all understood
        acceptable_kwargs = ['fR_parameter_type', 'log10fR0', 'stellar_screening', 'env_screening', 'MG_grid_dim']
        for k in kwargs.keys():
            if k not in acceptable_kwargs:
                raise KeyError("Unrecognised key: "+k+" in GalaxyFit")
        if self.fR:
            if 'fR_parameter_type' in kwargs.keys():
                self.fR_parameter_type == kwargs['fR_parameter_type']
                if self.fR_parameter_type not in ['fixed', 'free']:
                    raise ValueError("fR_parameter_type should be"
                                     "'fixed' or 'free'")
            assert 'stellar_screening' in kwargs.keys()
            assert 'env_screening' in kwargs.keys()
            assert 'MG_grid_dim' in kwargs.keys()

            if fR_parameter_type == 'fixed':
                assert 'log10fR0' in kwargs.keys()
        

# =============================================================================
#         fR_options = ['fR0', 'Rscr', 'fixed', 'Rscr_fixed', None]
#         assert fR_parameter in fR_options, "unrecognised fR"
#         if fR_parameter == 'fixed':
#             assert MGGridDim in ['1D', '2D'], "Need to specify MG solver type"
#             assert type(log10fR0) == float, "fR0 should be float"
#             assert log10fR0 >= -9, "fR0 needs to be greater than 1e-9"
#             assert log10fR0 <= np.log10(2e-6), "fR0 needs to be less than 2e-6"
#             assert Rscr_fixed is None, "don't fix Rscr if fixing fR0!"
#         elif fR_parameter == 'fR0':
#             assert MGGridDim in ['1D', '2D'], "Need to specify MG solver type"
#             assert log10fR0 is None, "Don't specify fR0 varying it!"
#             assert Rscr_fixed is None, "don't fix Rscr if varying fR0!"
#         elif fR_parameter == 'Rscr_fixed':
#             assert log10fR0 is None, "Don't specify fR0 if fixing Rscr!"
#             assert MGGridDim is None, "Only specify MGGridDim if fitting fR0"
#             assert type(Rscr_fixed) in [float, np.float32, np.float64]
#             assert Rscr_fixed >= 0
#             assert Rscr_fixed < 1.05*galaxy.R[-1]
#         else:
#             assert Rscr_fixed is None, "don't fix Rscr if varying fR0!"
#             assert log10fR0 is None, "Don't specify fR0 if using varying it!"
#             assert MGGridDim is None, "Only specify MGGridDim if fitting fR0"
#         if EnvScreening:
#             assert fR_parameter in ['fR0', 'fixed']
#         assert halo_type in ['NFW', 'DC14'], "unrecognised ML"
#         assert ML_ratio in ['fixed', 'single', 'double'], "unrecognised ML"
# 
#         self.galaxy = galaxy
#         self.fR_parameter = fR_parameter
#         self.log10fR0 = log10fR0
#         self.Rscr_fixed = Rscr_fixed
#         self.halo_type = halo_type
#         self.StellarScreening = StellarScreening
#         self.EnvScreening = EnvScreening
#         self.ML_ratio = ML_ratio
#         self.CosmicBaryonBound = CosmicBaryonBound
#         self.scaling_priors = scaling_priors
#         self.infer_errors = infer_errors
#         self.MGGridDim = MGGridDim
#         self.Verbose = Verbose
# 
#         self.ntemps = ntemps
#         self.nwalkers = nwalkers
#         self.threads = threads
#         self.IsConverged = False
# 
#         # assign number of free parameters ndim, as well as lower and upper
#         # bounds on priors lb and ub. Start with 2 NFW, then add 0, 1, or 2
#         # parameter(s) for M/L ratio, then 0 or 1 parameter for fR0/Rscr.
#         # Prior bounds are same as Katz et al except fR0/Rscr
#         ndim = 2
#         lb = np.array([4, 0])
#         ub = np.array([5.7, 2])
#         theta_dict = {'V_vir': 0, 'c_vir': 1}
#         if ML_ratio == 'single':
#             ndim += 1
#             lb = np.append(lb, -0.52)
#             ub = np.append(ub, -0.1)
#             theta_dict['ML_disc'] = 2
#             theta_dict['ML_bulge'] = 2
#         elif ML_ratio == 'double':
#             ndim += 2
#             lb = np.append(lb, [-0.52, -0.52])
#             ub = np.append(ub, [-0.1, -0.1])
#             theta_dict['ML_disc'] = 2
#             theta_dict['ML_bulge'] = 3
#         if fR_parameter == 'fR0':
#             ndim += 1
#             lb = np.append(lb, -9)
#             ub = np.append(ub, np.log10(2e-6))
#             theta_dict['fR0'] = ndim-1
#         elif fR_parameter == 'Rscr':
#             ndim += 1
#             lb = np.append(lb, 0)
#             ub = np.append(ub, 1.05*galaxy.R[-1])
#             theta_dict['Rscr'] = ndim-1
#         if infer_errors:  # extra parameter for modelled error
#             ndim += 1
#             lb = np.append(lb, 0)
#             ub = np.append(ub, 2*galaxy.v_err.max())
#             theta_dict['sigma_gal'] = ndim-1
#         assert lb.shape == ub.shape == (ndim,)
#         self.ndim = ndim
#         self.prior_bounds_lower = lb
#         self.prior_bounds_upper = ub
#         self.theta_dict = theta_dict
#         self.time_taken = 0
# =============================================================================
        return

    def initialise_pos(self):

        lb = self.prior_bounds_lower
        ub = self.prior_bounds_upper

        # initial positions, random within prior bounds
        d = ub-lb
        p0 = lb + d*np.random.rand(self.ntemps, self.nwalkers, self.ndim)

        # making sure all initial postions are not in region forbidden by
        # cosmic baryon bound; shifting them if they are
        if self.CosmicBaryonBound:
            for i in range(self.ntemps):
                for j in range(self.nwalkers):

                    BoundViolated = True
                    while BoundViolated:
                        theta = p0[i, j]
                        if self.ML_ratio == 'fixed':
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
                            BoundViolated = False

        return p0

    def iterate(self, niter, threads=1, initial_run=True, p0=None):
        """
        Perform niter iterations
        """

        loglargs = [self.theta_dict, self.galaxy,
                    self.fR_parameter, self.log10fR0, self.Rscr_fixed,
                    self.halo_type, self.StellarScreening, self.EnvScreening,
                    self.ML_ratio, self.infer_errors, self.MGGridDim,
                    self.Verbose]
        logpargs = [self.theta_dict,
                    self.prior_bounds_lower, self.prior_bounds_upper,
                    self.galaxy, self.halo_type, self.ML_ratio,
                    self.CosmicBaryonBound, self.scaling_priors]
        sampler = emcee.PTSampler(self.ntemps, self.nwalkers, self.ndim,
                                  lnlike, lnprior, threads=threads,
                                  loglargs=loglargs, logpargs=logpargs)

        # either initialise walker positions or continue existing chain
        if initial_run:
            self.niter = niter
            if p0 is None:
                p0 = self.initialise_pos()
            else:
                assert p0.shape == (self.ntemps, self.nwalkers, self.ndim)
        else:
            self.niter += niter
            p0 = self.chain[:, :, -1, :]

        # run MCMC
        t0 = time.time()
        for i, result in enumerate(sampler.sample(p0, iterations=niter)):
            if (i+1) % 10 == 0:
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

    def converge(self, niter):
        """
        Once convergence has been *independently* verified on the chains, this
        function saves maxprob parameter values, maxprob fit, and cuts the
        chain down to niter iterations.
        """
        self.IsConverged = True

        # cut chain
        start = self.niter - niter
        self.chain_converged = self.chain[:, :, start:, :]
        flatshape = (self.ntemps, self.nwalkers*niter, self.ndim)
        self.flatchain_converged = self.chain_converged.reshape(flatshape)
        self.lnprob_converged = self.lnprob[:, :, start:]

        # find maxprob values
        ind = np.argmax(self.lnprob_converged[0])
        theta = self.flatchain_converged[0][ind]
        v_fit = v_model(theta, self.theta_dict, self.galaxy,
                        fR_parameter=self.fR_parameter,
                        log10fR0=self.log10fR0,
                        Rscr_fixed=self.Rscr_fixed,
                        halo_type=self.halo_type,
                        StellarScreening=self.StellarScreening,
                        EnvScreening=self.EnvScreening,
                        ML_ratio=self.ML_ratio,
                        MGGridDim=self.MGGridDim,
                        component_split=True)

        self.maxprob_theta = theta
        self.maxprob_v_circ = v_fit[0]
        self.maxprob_v_gas = v_fit[1]
        self.maxprob_v_disc = v_fit[2]
        self.maxprob_v_bulge = v_fit[3]
        self.maxprob_v_DM = v_fit[4]
        self.maxprob_v_5 = v_fit[5]
        if self.infer_errors:
            sigma_gal = theta[self.theta_dict['sigma_gal']]
            self.maxprob_v_err = np.sqrt(self.galaxy.v_err**2 + sigma_gal**2)

        return


# MCMC log likelihood
def lnlike(theta, theta_dict, galaxy, fR_parameter, log10fR0, Rscr_fixed,
           halo_type, StellarScreening, EnvScreening, ML_ratio, infer_errors,
           MGGridDim, Verbose):

    if Verbose:
        print("Parameters: ", theta)

    model = v_model(theta=theta, theta_dict=theta_dict, galaxy=galaxy,
                    fR_parameter=fR_parameter, log10fR0=log10fR0,
                    Rscr_fixed=Rscr_fixed,
                    halo_type=halo_type,
                    StellarScreening=StellarScreening,
                    EnvScreening=EnvScreening,
                    ML_ratio=ML_ratio, MGGridDim=MGGridDim)
    if infer_errors:
        sigma = np.sqrt(galaxy.v_err**2 + theta[theta_dict['sigma_gal']]**2)
        lnlike = -np.sum(0.5*((galaxy.v - model)/sigma)**2 + np.log(sigma))
    else:
        lnlike = -0.5*np.sum(((galaxy.v - model)/galaxy.v_err)**2)

    return lnlike

if __name__=='__main__':
    g = GalaxyFit(True, fR=True, fR_parameter_type=True, env_screening=True, stellar_screening=True, MG_grid_dim=True)
