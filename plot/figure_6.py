#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created: 2018
Author: A. P. Naik
Description: Figure 6 of Naik et al., 2019
"""
from matplotlib import rcParams
from matplotlib.lines import Line2D
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


def figure_6():
    """
    Figure showing log-likelihood comparison of Models A, H, G, and I, as well
    as 3 rotation curves of relevant galaxies.

    4 panels: main top panel shows loglikelihood ratios, while 3 lower panels
    show rotation curves.
    """

    # create figure
    fig = plt.figure(figsize=(7, 7))
    ax_coords = ([0.16, 0.55, 0.68, 0.19],
                 [0.16, 0.26, 0.19, 0.19],
                 [0.405, 0.26, 0.19, 0.19],
                 [0.65, 0.26, 0.19, 0.19],)

    # open summary files
    summ_A = spam.analysis.open_summary('A')
    summ_G = spam.analysis.open_summary('G')
    summs_H = [spam.analysis.open_summary('H'+str(i)) for i in range(20)]
    summs_I = [spam.analysis.open_summary('I'+str(i)) for i in range(20)]

    # TOP PANEL: lnL ratios ####################################
    # loop over galaxies, calculate total likelihood
    lnL_A = 0
    lnL_G = 0
    lnL_H = np.zeros(20)
    lnL_I = np.zeros(20)
    for name in spam.data.names_standard:

        lnL_A += summ_A.galaxies[name].lnL
        lnL_G += summ_G.galaxies[name].lnL
        for j in range(20):
            lnL_H[j] += summs_H[j].galaxies[name].lnL
            lnL_I[j] += summs_I[j].galaxies[name].lnL

    ax = fig.add_axes(ax_coords[0])

    # labels
    l1 = (r'$\frac{\mathrm{NFW,}\ f(R)}'
          r'{\mathrm{NFW,}\ \Lambda\mathrm{CDM}}$')
    l2 = (r'$\frac{\mathrm{DC14,}\ \Lambda\mathrm{CDM}}'
          r'{\mathrm{NFW,}\ \Lambda\mathrm{CDM}}$')
    l3 = (r'$\frac{\mathrm{DC14,}\ f(R)}'
          r'{\mathrm{NFW,}\ \Lambda\mathrm{CDM}}$')

    x = np.linspace(np.log10(1.563e-8), np.log10(2e-6), 20)
    ax.plot(x, lnL_H-lnL_A, lw=2, c=green3, label=l1)
    ax.plot(x, (lnL_G-lnL_A)*np.ones(20), lw=2, c=comp, ls='--', label=l2)
    ax.plot(x, lnL_I-lnL_A, lw=2, c=comp, label=l3)
    ax.plot([x[0], x[-1]], [0, 0], ls=':', c='grey')

    ax.set_xlabel(r"$\log_{10}f_{R0}$")
    ax.set_ylabel("Log-likelihood Ratio")
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)
    ax.set_ylim(-110)

    # BOTTOM PANEL: RC fits ###################################
    names = ['DDO161', 'UGC00891', 'NGC3109']
    for i in range(3):
        # set up axes
        name = names[i]
        ax = fig.add_axes(ax_coords[i+1])
        ax.set_xlabel(r"$R\ [\mathrm{kpc}]$")
        if i == 0:
            ax.set_ylabel(r"$v_\mathrm{circ}\ [\mathrm{km/s}]$")
        ax.set_title(name)

        gal = spam.data.SPARCGalaxy(name)
        ax.errorbar(gal.R, gal.v, gal.v_err,
                    fmt='.', ms=3, elinewidth=0.5, capsize=1.5, ecolor='grey',
                    mfc='k', mec='k', label=r'$v_\mathrm{data}$')

        gal_A = summ_A.galaxies[name]
        gal_G = summ_G.galaxies[name]
        gal_H = summs_H[7].galaxies[name]

        # plot RC fits and components
        ax.plot(gal.R, gal_A.maxprob_v_circ, c=green4, ls='dashed')
        ax.plot(gal.R, gal_G.maxprob_v_circ, c=comp, ls='dashed')
        ax.plot(gal.R, gal_H.maxprob_v_circ, c=green4)
        ax.plot(gal.R, gal_H.maxprob_v_5, c=green4, ls=':')

    handles = [ax.get_legend_handles_labels()[0][0],
               Line2D([0], [0], label=r'NFW, $f(R)$', c=green4),
               Line2D([0], [0], label=r'NFW, $\Lambda$CDM', c=green4, ls='--'),
               Line2D([0], [0], label='DC14, $\Lambda$CDM', c=comp, ls='--'),
               Line2D([0], [0], label=r'$v_5$', c=green4, ls='dotted')]

    ax.legend(handles=handles, loc='center left',
              bbox_to_anchor=(1, 0.5), frameon=False)

    # phantom bounding box to make sure bbox_inches cuts symmetrically
    fig.patches.extend([plt.Rectangle((0.005, 0.26), 0.99, 0.48, lw=0,
                                      fill=False, zorder=1000, edgecolor=None,
                                      transform=fig.transFigure, figure=fig)])

    return fig
