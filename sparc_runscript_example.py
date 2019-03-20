#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created: 1st March 2019
Author: A. P. Naik
Description: Example runscript
"""
import sparc_data
import sparc_mcmc as mcmc

# params
gal_name = "CamB"
fR_parameter = "fR0"
log10fR0 = None
Verbose = False
ntemps = 2
threads = 4
nwalkers = 10
niter = 100
filename = '../objects/CamB.obj'
halo_type = "NFW"
StellarScreening = False
EnvScreening = False
ML_ratio = "single"
MGGridDim = "1D"
CosmicBaryonBound = True
scaling_priors = True
infer_errors = True

# load SPARC data
sparc = sparc_data.SPARCGlobal()
ind = sparc.names.index(gal_name)
gal = sparc.galaxies[ind]


# get fit
fit = mcmc.SoloGalFit(gal, fR_parameter=fR_parameter, log10fR0=log10fR0,
                      halo_type=halo_type, StellarScreening=StellarScreening,
                      EnvScreening=EnvScreening, ML_ratio=ML_ratio,
                      Verbose=False, CosmicBaryonBound=CosmicBaryonBound,
                      scaling_priors=scaling_priors, infer_errors=infer_errors,
                      MGGridDim=MGGridDim,
                      ntemps=ntemps, nwalkers=nwalkers, threads=threads)
fit.iterate(niter=niter)
