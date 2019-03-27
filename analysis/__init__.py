#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created: 20th March 2019
Author: A. P. Naik
Description: init file of 'analysis' submodule.
"""
import os as _os
import pickle as _pickle
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


def lnL(fit):

    data = fit.galaxy.v
    model = fit.maxprob_v_circ
    err = fit.maxprob_v_err

    lnL = -0.5*_np.sum(((data-model)**2/err**2)+_np.log(2*_np.pi*err**2))
    return lnL


def BIC(fit):
    """
    BIC = ln(n)k - 2 lnL
    """
    n = fit.galaxy.R.size
    k = fit.ndim

    BIC = _np.log(n)*k - 2*lnL(fit)

    return BIC


__all__ = ['open_fit', 'open_summary']
