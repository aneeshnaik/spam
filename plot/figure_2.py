#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created: 2018
Author: A. P. Naik
Description: Figure 2 in Naik et al., 2019.
"""
from matplotlib import rcParams
import matplotlib.pyplot as plt
import spam
rcParams['text.usetex'] = True
rcParams['font.size'] = 8

green1 = '#EDF8E9'
green2 = '#BAE4B3'
green3 = '#10C476'
green4 = '#31A354'
green5 = '#006D2C'

comp = '#A33159'


def figure_2():
    """
    Figure depicting an example fit.

    Single panel: rotation curve of NGC3741, along with Model B fit and
    components.
    """

    name = 'NGC3741'
    fig = plt.figure(figsize=(3.3, 3.3))
    ax = fig.add_axes([0.2, 0.2, 0.75, 0.75])

    # set up axes
    ax.set_xlabel(r"$R\ [\mathrm{kpc}]$")
    ax.set_ylabel(r"$v_\mathrm{circ}\ [\mathrm{km/s}]$")

    # load summary file
    summ_B = spam.analysis.open_summary('B')
    gal = summ_B.galaxies[name]

    # plot data
    ax.errorbar(gal.R, gal.v, gal.v_err,
                fmt='.', ms=3, elinewidth=0.5, capsize=1.5, ecolor='grey',
                mfc='k', mec='k', label=r'$v_\mathrm{data}$')

    # plot RC fits and components
    ax.plot(gal.R, gal.maxprob_v_circ,
            label=r'$v_{f(R)}$', c=green4)
    ax.plot(gal.R, gal.maxprob_v_DM,
            label=r'$v_\mathrm{DM}$', c='grey')
    ax.plot(gal.R, gal.maxprob_v_gas,
            label=r'$v_\mathrm{gas}$', c='grey', ls='-.')
    ax.plot(gal.R, gal.maxprob_v_disc,
            label=r'$v_\mathrm{disc}$', c='grey', ls=':')
    ax.plot(gal.R, gal.maxprob_v_5,
            label=r'$v_\mathrm{5}$', c='grey', ls='--')

    ax.legend(frameon=False)

    return fig
