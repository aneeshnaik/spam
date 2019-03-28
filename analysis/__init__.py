#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created: 20th March 2019
Author: A. P. Naik
Description: init file of 'analysis' submodule.
"""
import os as _os
import pickle as _pickle
import spam as _spam
import util as _util
import numpy as _np


def open_summary(model, fitdir=_os.environ['SPAMFITDIR']):

    summfile = open(fitdir+'summaries/MODEL_'+model+'_summary.obj', "rb")
    summ = _pickle.load(summfile)
    summfile.close()

    return summ


def open_fit(model, name, fitdir=_os.environ['SPAMFITDIR']):

    if model == '2D':
        fitfile = open(fitdir+'2D_TEST/'+name+'.obj', 'rb')
    else:
        fitfile = open(fitdir+'MODEL_'+model+'/'+name+'.obj', 'rb')

    fit = _pickle.load(fitfile)
    fitfile.close()
    return fit


def _lnL(data, model, err):
    lnL = -0.5*_np.sum(((data-model)**2/err**2)+_np.log(2*_np.pi*err**2))
    return lnL


def _BIC(n, k, lnL):
    """
    BIC = ln(n)k - 2 lnL
    """
    BIC = _np.log(n)*k - 2*lnL
    return BIC


class FitSummary():
    def __init__(self, model, sample='standard',
                 fitdir=_os.environ['SPAMFITDIR']):

        self.model = model
        self.sample = sample
        if sample == 'standard':
            names = _spam.data.names_standard
        else:
            names = _spam.data.names_full

        # loop over galaxies
        print("Model "+model+": looping over galaxies...")
        self.galaxies = {}
        for name in names:

            # progress bar
            _util.print_progress(names.index(name), len(names))

            # load galaxy data and fit
            galaxy = _spam.data.SPARCGalaxy(name)
            fit = _spam.analysis.open_fit(model, name, fitdir)

            # find maximum posterior parameters
            lnprob = fit.lnprob[0, :, -5000:]
            ind = _np.unravel_index(_np.argmax(lnprob), lnprob.shape)
            chain = fit.chain[0, :, -5000:, :]
            theta = chain[ind]
            galaxy.maxprob_theta = theta

            # calculate velocity for best fit model
            kw = fit.__dict__
            v = _spam.fit.rotcurve.v_model(theta, component_split=True, **kw)
            galaxy.maxprob_v_circ = v[0]
            galaxy.maxprob_v_gas = v[1]
            galaxy.maxprob_v_disc = v[2]
            galaxy.maxprob_v_bulge = v[3]
            galaxy.maxprob_v_DM = v[4]
            galaxy.maxprob_v_5 = v[5]

            # calculate errors if fitting
            if fit.infer_errors:
                sigma_gal = theta[fit.theta_dict['sigma_gal']]
                err = _np.sqrt(galaxy.v_err**2 + sigma_gal**2)
                galaxy.maxprob_v_err = err
                galaxy.lnL = _lnL(data=galaxy.v, model=v[0], err=err)
            else:
                galaxy.lnL = _lnL(data=galaxy.v, model=v[0], err=galaxy.v_err)

            # calculate BIC
            n = fit.galaxy.R.size
            k = fit.ndim
            galaxy.BIC = _BIC(n, k, galaxy.lnL)

            # save galaxy
            self.galaxies[name] = galaxy

        return


__all__ = ['open_fit', 'open_summary', 'FitSummary']
