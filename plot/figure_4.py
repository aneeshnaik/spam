#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created: 2018
Author: A. P. Naik
Description: Figure 4 in Naik et al., 2019
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


def figure_4():
    """
    Figure showing CDF of dBIC values. Single panel.
    """
    # get dBIC values
    dBIC = np.array([])
    summ_A = spam.analysis.open_summary('A')
    summ_B = spam.analysis.open_summary('B')
    for name in spam.data.names_standard:
        gal_A = summ_A.galaxies[name]
        gal_B = summ_B.galaxies[name]
        dBIC = np.append(dBIC, gal_B.BIC-gal_A.BIC)

    fig = plt.figure(figsize=(3.3, 3.3))
    ax = fig.add_axes([0.2, 0.15, 0.75, 0.75])

    # histogram
    bins = np.append(-1e+10, np.linspace(-10, 10, 101))
    bins = np.append(bins, 1e+10)
    n, bins, patches = plt.hist(dBIC, bins=bins, cumulative=True,
                                histtype='step', color=green4)

    # fiducial markers
    plt.plot([-10, -6], [n[20], n[20]], ls='dashed', c='grey')
    plt.plot([-10, -2], [n[40], n[40]], ls='dashed', c='grey')
    plt.plot([-10, 0], [n[50], n[50]], ls='dashed', c='grey')
    plt.plot([-10, 2], [n[60], n[60]], ls='dashed', c='grey')
    plt.plot([-10, 6], [n[80], n[80]], ls='dashed', c='grey')
    plt.plot([-6, -6], [0, n[20]], ls='dashed', c='grey')
    plt.plot([-2, -2], [0, n[40]], ls='dashed', c='grey')
    plt.plot([0, 0], [0, n[50]], ls='dashed', c='grey')
    plt.plot([2, 2], [0, n[60]], ls='dashed', c='grey')
    plt.plot([6, 6], [0, n[80]], ls='dashed', c='grey')

    plt.yticks([n[20], n[40], n[50], n[60], n[80]])
    plt.xticks([-10, -6, -2, 0, 2, 6, 10])

    # arrows
    plt.arrow(0.49, 1.025, -0.4, 0, transform=ax.transAxes, clip_on=False,
              width=0.01, facecolor=green5)
    plt.arrow(0.51, 1.025, 0.4, 0, transform=ax.transAxes, clip_on=False,
              width=0.01, facecolor=green5)
    plt.text(0.25, 1.05, r'Prefers $f(R)$', transform=ax.transAxes,
             ha='center')
    plt.text(0.75, 1.05, r'Prefers $\Lambda$CDM', transform=ax.transAxes,
             ha='center')

    plt.xlim(-10, 10)
    plt.ylim(0, 86)
    plt.xlabel(r"$\Delta\mathrm{BIC}$")
    plt.ylabel("Number of Galaxies")

    return fig
