#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created: 2018
Author: A. P. Naik
Description: 
"""
import matplotlib.pyplot as plt


def figure_2():
    """
    Plots of fits of NGC3741, DDO161 and NGC0289
    """

    name = 'NGC3741'
    fig = plt.figure(figsize=(3.3, 3.3))
    ax = fig.add_axes([0.2, 0.2, 0.75, 0.75])

    # set up axes
    ax.set_xlabel(r"$R\ [\mathrm{kpc}]$")
    ax.set_ylabel(r"$v_\mathrm{circ}\ [\mathrm{km/s}]$")

    # load summary file
    summ_B = open_summary('B')
    gal_B = summ_B.galaxies[summ_B.names.index(name)]

    # plot data
    gal = sparc.galaxies[sparc.names.index(name)]
    ax.errorbar(gal.R, gal.v, gal.v_err,
                fmt='.', ms=3, elinewidth=0.5, capsize=1.5, ecolor='grey',
                mfc='k', mec='k', label=r'$v_\mathrm{data}$')

    # plot RC fits and components
    ax.plot(gal.R, gal_B.maxprob_v_circ,
            label=r'$v_{f(R)}$', c=green4)
    ax.plot(gal.R, gal_B.maxprob_v_DM,
            label=r'$v_\mathrm{DM}$', c='grey')
    ax.plot(gal.R, gal_B.maxprob_v_gas,
            label=r'$v_\mathrm{gas}$', c='grey', ls='-.')
    ax.plot(gal.R, gal_B.maxprob_v_disc,
            label=r'$v_\mathrm{disc}$', c='grey', ls=':')
    ax.plot(gal.R, gal_B.maxprob_v_5,
            label=r'$v_\mathrm{5}$', c='grey', ls='--')

    ax.legend(frameon=False)

    return fig