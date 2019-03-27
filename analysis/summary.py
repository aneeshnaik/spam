#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created: 2019
Author: A. P. Naik
Description:
"""
import spam
import os


class FitSummary():
    def __init__(self, model, sample='standard',
                 fitdir=os.environ['SPAMFITDIR']):

        self.model = model
        self.sample = sample
        if sample == 'standard':
            names = spam.data.names_standard
        else:
            names = spam.data.names_full

        self.galaxies = {}
        for name in names:

            galaxy = spam.data.SPARCGalaxy(name)
            fit = spam.analysis.open_fit(model, name, fitdir)

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

            self.galaxies[name] = galaxy
        return
