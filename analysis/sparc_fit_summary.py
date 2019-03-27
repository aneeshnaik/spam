#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created: 18th February 2019
Author: A. P. Naik
Description: Create object containing summary data from fits of entire SPARC
sample
"""
import numpy as np
import pickle
from sparc_data import SPARCGlobal
from os import environ
from os.path import exists


def open_fit(directory, name):

    fitfile = open(directory+name+'.obj', 'rb')
    fit = pickle.load(fitfile)
    fitfile.close()

    return fit


def lnL(fit):

    data = fit.galaxy.v
    model = fit.maxprob_v_circ
    err = fit.maxprob_v_err

    lnL = -0.5*np.sum(((data-model)**2/err**2)+np.log(2*np.pi*err**2))
    return lnL


def BIC(fit):
    """
    BIC = ln(n)k - 2 lnL
    """
    n = fit.galaxy.R.size
    k = fit.ndim

    BIC = np.log(n)*k - 2*lnL(fit)

    return BIC


class SPARCFitSummary(SPARCGlobal):
    def __init__(self, directory, sample='standard',
                 datadir=environ['SPARCDIR']+"/SPARCData"):
        super().__init__(sample=sample, datadir=datadir)

        assert exists(directory)
        for galaxy in self.galaxies:

            fit = open_fit(directory, galaxy.name)

            galaxy.maxprob_theta = fit.maxprob_theta
            galaxy.maxprob_v_circ = fit.maxprob_v_circ
            galaxy.maxprob_v_gas = fit.maxprob_v_gas
            galaxy.maxprob_v_disc = fit.maxprob_v_disc
            galaxy.maxprob_v_bulge = fit.maxprob_v_bulge
            galaxy.maxprob_v_DM = fit.maxprob_v_DM
            galaxy.maxprob_v_5 = fit.maxprob_v_5
            if fit.infer_errors:
                galaxy.maxprob_v_err = fit.maxprob_v_err

            galaxy.lnL = lnL(fit)
            galaxy.BIC = BIC(fit)

        return
