#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created: 2018
Author: A. P. Naik
Description: Figure 9 of Naik et al., 2019
"""
from matplotlib import rcParams
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import spam
rcParams['text.usetex'] = True
rcParams['font.size'] = 8

green1 = '#EDF8E9'
green2 = '#BAE4B3'
green3 = '#10C476'
green4 = '#31A354'
green5 = '#006D2C'

comp = '#A33159'


def figure_9():
    """
    Figure showing change to inferred fR0 by introducing stellar screening.
    One panel.
    """
    # loop over galaxies, get inferred fR0 values
    summ_B = spam.analysis.open_summary('B')
    summ_D = spam.analysis.open_summary('D')
    L_star = np.zeros(85)
    fR0_B = np.zeros(85)
    fR0_D = np.zeros(85)
    for i in range(85):
        name = spam.data.names_standard[i]
        L_star[i] = spam.data.SPARCGalaxy(name).luminosity_tot
        fR0_B[i] = summ_B.galaxies[name].maxprob_theta[3]
        fR0_D[i] = summ_D.galaxies[name].maxprob_theta[3]

    # set up figure
    fig = plt.figure(figsize=(3.3, 3.3))
    fig.add_axes([0.2, 0.2, 0.75, 0.75])

    # plot scatters
    plt.xscale('log')
    plt.scatter(L_star, fR0_B, facecolors='none', edgecolors=green5,
                s=6, label="No stellar screening")
    plt.scatter(L_star, fR0_D, facecolors=green5, edgecolors=green5,
                s=6, label="Stellar screening")

    # plot lines
    for i in range(85):
        plt.plot([L_star[i], L_star[i]], [fR0_B[i], fR0_D[i]],
                 ls='dashed', c=green5, lw=0.5)

    # legend
    handles = [Line2D([0], [0], marker='.', lw=0, label='Stellar Screening',
                      mfc=green5, mec=green5, ms=10),
               Line2D([0], [0], marker='.', lw=0, label='No Stellar Screening',
                      mfc='none', mec=green5, ms=10)]
    plt.legend(frameon=False, handles=handles)

    # axis labels
    plt.ylabel(r"$\log_{10}|\bar{f}_{R0}|$")
    plt.xlabel(r"$L\ [10^9 L_\odot]$")

    return fig
