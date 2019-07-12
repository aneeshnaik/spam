#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created: 2018
Author: A. P. Naik
Description: Figure 10 of Naik et al., 2019
"""
from matplotlib import rcParams
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
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


def figure_10():
    """
    Figure showing different inferred fR0 values for different treatments of
    the mass to light ratio.
    """
    # load summaries for Models B, E, F
    summ_B = spam.analysis.open_summary('B')
    summ_E = spam.analysis.open_summary('E')
    summ_F = spam.analysis.open_summary('F')

    # loop over galaxies, get maxprob fR0
    fR0_B = np.zeros(85)
    fR0_E = np.zeros(85)
    fR0_F = np.zeros(85)
    for i in range(85):
        name = spam.data.names_standard[i]
        fR0_B[i] = summ_B.galaxies[name].maxprob_theta[-2]
        fR0_E[i] = summ_E.galaxies[name].maxprob_theta[-2]
        fR0_F[i] = summ_F.galaxies[name].maxprob_theta[-2]

    # plot
    fig = plt.figure(figsize=(3.3, 3.3))
    fig.add_axes([0.2, 0.2, 0.75, 0.75])

    x = np.arange(85)
    plt.scatter(x, fR0_B, facecolors='k', edgecolors='k', s=6)
    plt.scatter(x, fR0_E, facecolors='darkgrey', edgecolors='darkgrey', s=6)
    plt.scatter(x, fR0_F, facecolors=green5, edgecolors=green5, s=6)

    for i in range(85):
        plt.plot([i, i], [fR0_B[i], fR0_E[i]],
                 ls='dashed', c='grey', lw=0.5)
        plt.plot([i, i], [fR0_B[i], fR0_F[i]],
                 ls='dashed', c='grey', lw=0.5)

    # legend
    handles = [Line2D([0], [0], marker='.', lw=0, label=r"Single $\Upsilon$",
                      mfc='k', mec='k', ms=10),
               Line2D([0], [0], marker='.', lw=0, label=r"Fixed $\Upsilon$",
                      mfc='darkgrey', mec='darkgrey', ms=10),
               Line2D([0], [0], marker='.', lw=0, label=r"Double $\Upsilon$",
                      mfc=green5, mec=green5, ms=10)]
    plt.legend(frameon=False, handles=handles)

    # axis labels
    plt.ylabel(r"$\log_{10}|\bar{f}_{R0}|$")
    plt.xlabel("Galaxy")

    return fig
