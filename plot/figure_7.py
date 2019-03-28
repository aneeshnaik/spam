#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created: 2018
Author: A. P. Naik
Description: Figure 7 of Naik et al., 2019
"""
from matplotlib import rcParams
from matplotlib.lines import Line2D
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


def figure_7():
    """
    Figure showing galaxies which most prefer f(R) over cored halo.

    Three panels: RCs and fits for UGC11820, NGC2403 and UGC05253.
    """
    summ_G = spam.analysis.open_summary('G')
    summ_B = spam.analysis.open_summary('B')

    fig = plt.figure(figsize=(7, 3.5))
    ax_coords = ([0.12, 0.3, 0.22, 0.44],
                 [0.39, 0.3, 0.22, 0.44],
                 [0.66, 0.3, 0.22, 0.44])

    names = ['UGC11820', 'NGC2403', 'UGC05253']

    for i in range(3):

        # set up axes
        name = names[i]
        ax = fig.add_axes(ax_coords[i])
        ax.set_xlabel(r"$R\ [\mathrm{kpc}]$")
        if i == 0:
            ax.set_ylabel(r"$v_\mathrm{circ}\ [\mathrm{km/s}]$")

        ax.set_title(name)

        gal = spam.data.SPARCGalaxy(name)
        ax.errorbar(gal.R, gal.v, gal.v_err,
                    fmt='.', ms=3, elinewidth=0.5, capsize=1.5, ecolor='grey',
                    mfc='k', mec='k', label=r'$v_\mathrm{data}$')

        gal_G = summ_G.galaxies[name]
        gal_B = summ_B.galaxies[name]

        # plot RC fits
        ax.plot(gal.R, gal_G.maxprob_v_circ, c=comp, ls='dashed')
        ax.plot(gal.R, gal_B.maxprob_v_circ, c=green4)
        ax.plot(gal.R, gal_B.maxprob_v_5, c=green4, ls=':')

        if i == 1:
            rect = plt.Rectangle((1, 90), 4, 35, fill=False)
            ax.add_artist(rect)
            ax_inset = fig.add_axes([0.5, 0.34, 0.1, 0.2])
            plt.errorbar(gal.R, gal.v, gal.v_err, fmt='.', ms=3,
                         elinewidth=0.5, capsize=1.5, ecolor='grey',
                         mfc='k', mec='k', label=r'$v_\mathrm{data}$')
            plt.plot(gal.R, gal_G.maxprob_v_circ, c=comp, ls='dashed')
            plt.plot(gal.R, gal_B.maxprob_v_circ, c=green4)
            ax_inset.set_xlim(1, 5)
            ax_inset.set_ylim(90, 125)
            ax_inset.tick_params(axis='y', labelleft=False, left=False)
            ax_inset.tick_params(axis='x', labelbottom=False, bottom=False)
        elif i == 2:
            rect = plt.Rectangle((-1, 230), 10, 30, fill=False)
            ax.add_artist(rect)
            ax_inset = fig.add_axes([0.77, 0.34, 0.1, 0.2])
            plt.errorbar(gal.R, gal.v, gal.v_err, fmt='.', ms=3,
                         elinewidth=0.5, capsize=1.5, ecolor='grey',
                         mfc='k', mec='k', label=r'$v_\mathrm{data}$')
            plt.plot(gal.R, gal_G.maxprob_v_circ, c=comp, ls='dashed')
            plt.plot(gal.R, gal_B.maxprob_v_circ, c=green4)
            ax_inset.set_xlim(-1, 9)
            ax_inset.set_ylim(230, 260)
            ax_inset.tick_params(axis='y', labelleft=False, left=False)
            ax_inset.tick_params(axis='x', labelbottom=False, bottom=False)

    handles = [ax.get_legend_handles_labels()[0][0],
               Line2D([0], [0], label='DC14', c=comp, ls='dashed'),
               Line2D([0], [0], label=r'$f(R)$', c=green4),
               Line2D([0], [0], label=r'$v_5$', c=green4, ls='dotted')]

    ax.legend(handles=handles, loc='center left',
              bbox_to_anchor=(1, 0.5), frameon=False)

    # phantom bounding box to make sure bbox_inches cuts symmetrically
    fig.patches.extend([plt.Rectangle((0.015, 0.3), 0.97, 0.44, lw=0,
                                      fill=False, zorder=1000, edgecolor=None,
                                      transform=fig.transFigure, figure=fig)])

    return fig
