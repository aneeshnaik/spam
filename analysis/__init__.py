#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created: 2018
Author: A. P. Naik
Description: 'analysis' submodule of spam package. See README for details and
usage examples.
"""
import os as _os
import pickle as _pickle
import spam as _spam
import numpy as _np


def open_summary(model, fitdir=_os.environ['SPAMFITDIR']):
    """
    Load pickled spam.analysis.FitSummary object.

    Parameters
    ----------
    model : str, {'A'-'G', 'H0'-'H19', 'I0'-'I19'}
        Which 'model' the fits being summarised belong to. See Table 1 in Naik
        et al., 2019 for details about the models.
    fitdir : str
        Path to directory containing 'summaries' directory, containing the
        relevant summary file. If user sets this as an environment variable,
        then this is used by default, otherwise it must be user specified.
    """
    summfile = open(fitdir+'summaries/MODEL_'+model+'_summary.obj', "rb")
    summ = _pickle.load(summfile)
    summfile.close()

    return summ


def open_fit(model, name, fitdir=_os.environ['SPAMFITDIR']):
    """
    Load pickled spam.fit.GalaxyFit object.

    Parameters
    ----------
    model : str, {'A'-'G', 'H0'-'H19', 'I0'-'I19'}
        Which 'model' the fits being summarised belong to. See Table 1 in Naik
        et al., 2019 for details about the models.
    name : str
        Name of galaxy.
    fitdir : str
        Path to directory containing 'summaries' directory, containing the
        relevant summary file. If user sets this as an environment variable,
        then this is used by default, otherwise it must be user specified.
    """

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
    BIC = _np.log(n)*k - 2*lnL
    return BIC


class FitSummary():
    """
    Object containing summary data for ensemble of MCMC samples across SPARC
    galaxies.

    Parameters
    ----------
    model : str, {'A'-'G', 'H0'-'H19', 'I0'-'I19'}
        Which 'model' the fits being summarised belong to. See Table 1 in Naik
        et al., 2019 for details about the models.
    sample : str, {'standard', 'full'}
        Whether the sample of fits is the 85-strong 'standard' sample analysed
        in Naik et al., 2019, or the 147-strong 'full' sample, which includes
        the galaxies eliminated by the environmental screening cut described
        in that paper.
    fitdir : str
        Path to directory containing 'summaries' directory, containing the
        relevant summary file. If user sets this as an environment variable
        under the name SPAMFITDIR, then this is used by default, otherwise it
        must be user specified.

    Attributes
    ----------
    galaxies : dict
        Each key is the name of a SPARC galaxy, while each value is an instance
        of the class spam.data.SPARCGalaxy with a few extra attributes,
        described below.
    galaxies['name'].maxprob_theta : 1D np.ndarray
        For named galaxy, maximum probability parameter values.
    galaxies['name'].maxprob_v_circ : 1D np.ndarray
        Best fit rotation curve model for named galaxy.
    galaxies['name'].maxprob_v_gas : 1D np.ndarray
        Gas contribution to best fit model.
    galaxies['name'].maxprob_v_disc : 1D np.ndarray
        Disc contribution to best fit model.
    galaxies['name'].maxprob_v_bulge : 1D np.ndarray
        Bulge contribution to best fit model.
    galaxies['name'].maxprob_v_DM : 1D np.ndarray
        DM contribution to best fit model.
    galaxies['name'].maxprob_v_5 : 1D np.ndarray
        Fifth force contribution to best fit model.
    galaxies['name'].maxprob_v_err : 1D np.ndarray
        If inferring errors, then total errors.
    galaxies['name'].lnL : float
        Log-likelihood of best fit model
    galaxies['name'].BIC : float
        Bayesian information criterion of best fit model
    """
    def __init__(self, model, sample='standard',
                 fitdir=_os.environ['SPAMFITDIR']):
        """
        Initialise an instance of FitSummary class. See class dosctring for
        more info.
        """
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
