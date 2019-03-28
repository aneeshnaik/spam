#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created: 2018
Author: A. P. Naik
Description: Figure 3 in Naik et al., 2019.
"""
from matplotlib import rcParams
import matplotlib.pyplot as plt
import spam
import numpy as np
rcParams['text.usetex'] = True
rcParams['font.size'] = 8

green1 = '#EDF8E9'
green2 = '#BAE4B3'
green3 = '#10C476'
green4 = '#31A354'
green5 = '#006D2C'

comp = '#A33159'


def figure_3():
    """
    Figure showing priors used.

    Two panels: Upper shows stellar mass / halo mass relation, while lower
    shows concentration / halo mass relation. 1 sigma and 2 sigma regions are
    shown as coloured regions, while a dashed line also shows previous 1 sigma
    region, before broadening to account for f(R).
    """

    # set up figure
    fig = plt.figure(figsize=(3.5, 7))
    ax1 = fig.add_axes([0.15, 0.45, 0.7, 0.35])
    ax2 = fig.add_axes([0.15, 0.1, 0.7, 0.35])

    # halo masses (x axis in both panels)
    Msun = 1.989e+30
    M_h = np.logspace(9, 14)*Msun
    x = np.log10(M_h/Msun)

    # calculate stellar masses
    y = np.log10(spam.fit.prior.SHM(M_h)/Msun)
    sig_old = spam.fit.prior.err_SHM(M_h)
    sig = sig_old + 0.2  # f(R) broadening

    # plot regions and lines in top panel
    ax1.fill_between(x, y-2*sig, y+2*sig, color=green2)
    ax1.fill_between(x, y-sig, y+sig, color=green4)
    ax1.plot(x, y, c='k')
    ax1.plot(x, y-sig_old, ls='dashed', c='lightgrey')
    ax1.plot(x, y+sig_old, ls='dashed', c='lightgrey')

    # calculate concentrations
    y = spam.fit.prior.CMR(M_h)
    sig_old = 0.11
    sig = sig_old + 0.1  # f(R) broadening

    # plot regions and lines in lower panel
    ax2.fill_between(x, y-2*sig, y+2*sig, color=green2)
    ax2.fill_between(x, y-sig, y+sig, color=green4)
    ax2.plot(x, y, c='k')
    ax2.plot(x, y+sig_old, ls='dashed', c='lightgrey')
    ax2.plot(x, y-sig_old, ls='dashed', c='lightgrey')

    # line up x axis limits in both panels
    ax2.set_xlim(ax1.get_xlim())

    # axis labels
    ax1.set_ylabel(r"$\log_{10} M_*\ [M_\odot]$")
    ax2.set_ylabel(r"$\log_{10} c_\mathrm{vir}$")
    ax2.set_xlabel(r"$\log_{10} M_\mathrm{halo}\ [M_\odot]$")

    return fig
