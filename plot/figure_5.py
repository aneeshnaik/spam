#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created: 2018
Author: A. P. Naik
Description: Figure 5 of Naik et al., 2019
"""
from matplotlib import rcParams
from matplotlib import colors
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


def create_cmap(clist):
    cmap = colors.LinearSegmentedColormap.from_list('mycmap', clist)
    return cmap


def figure_5():
    """
    Figure showing spread of inferred fR0 values.

    Two panels: Top panel shows all marginal posteriors for fR0 from Model B.
    Bottom panel shows likelihood ratios for all galaxies for all fR0 values
    for Model H vs. Model A.
    """
    # get histograms and argmaxes for each galaxy
    fR_edges = np.linspace(-9, np.log10(2e-6), 201)
    hists = np.zeros((85, 200))
    argmaxes = np.zeros(85)
    print("Calculating histograms...")
    for i in range(85):
        name = spam.data.names_standard[i]
        fit = spam.analysis.open_fit('B', name)
        chain = fit.chain[0, :, -5000:, -2].flatten()
        hist = np.histogram(chain, fR_edges)[0]
        hists[i] = hist
        argmaxes[i] = np.argmax(hist)

    # set up figure
    fig = plt.figure(figsize=(3.5, 7))
    ax1 = fig.add_axes([0.2, 0.4, 0.6, 0.3])
    ax2 = fig.add_axes([0.2, 0.1, 0.6, 0.3])
    cax1 = fig.add_axes([0.8, 0.4, 0.035, 0.3])
    cax2 = fig.add_axes([0.8, 0.1, 0.035, 0.3])

    # top panel ###############################################################

    # edges of mesh
    fR_edges = np.linspace(-9, np.log10(2e-6), 201)
    gal_edges = np.linspace(-0.5, 84.5, 86)

    # plot colormesh and colorbar
    mesh = ax1.pcolormesh(gal_edges, fR_edges, hists[np.argsort(argmaxes)].T,
                          vmax=2500, vmin=0, cmap='Greys')
    plt.colorbar(mesh, cax1)
    cax1.set_ylabel("Bin count")
    cax1.yaxis.set_label_position('right')

    # get values of best fit fR0
    summ_B = spam.analysis.open_summary('B')
    fR0_bests = np.array([])
    for name in spam.data.names_standard:
        fR0 = summ_B.galaxies[name].maxprob_theta[-2]
        fR0_bests = np.append(fR0_bests, fR0)

    # get list of hubble types and create ordered colour list
    h_types = []
    for name in spam.data.names_standard:
        h_types.append(spam.data.SPARCGalaxy(name).hubble_type)
    h_types = np.array(h_types)
    cdict = {'Sa': green1, 'Sab': green1, 'Sb': green1,
             'Sbc': green2, 'Sc': green2, 'Scd': green2,
             'Sd': green3, 'Sdm': green3,
             'Sm': green4, 'BCD': green4, 'Im': green5}
    clist = np.array([cdict[htype] for htype in h_types[np.argsort(argmaxes)]])

    # markers for legend
    marker1 = Line2D([], [], color=green1, marker='.',
                     markersize=15, mec='k', label='Sa/Sab/Sb', lw=0)
    marker2 = Line2D([], [], color=green2, marker='.',
                     markersize=15, mec='k', label='Sbc/Sc/Scd', lw=0)
    marker3 = Line2D([], [], color=green3, marker='.',
                     markersize=15, mec='k', label='Sd/Sdm', lw=0)
    marker4 = Line2D([], [], color=green4, marker='.',
                     markersize=15, mec='k', label='Sm/BCD', lw=0)
    marker5 = Line2D([], [], color=green5, marker='.',
                     markersize=15, mec='k', label='Im', lw=0)
    mlist = [marker1, marker2, marker3, marker4, marker5]

    # plot scatter of best fit values, coloured by Hubble type
    x = np.arange(85)
    y = fR0_bests[np.argsort(argmaxes)]
    ax1.scatter(x, y, c=clist, s=12, edgecolor='k', linewidths=0.5)
    ax1.legend(handles=mlist, framealpha=0.4, frameon=False)

    # bottom panel ############################################################

    # edges of colormesh
    fR0_vals = np.linspace(np.log10(1.563e-8), np.log10(2e-6), 20)
    dfR0 = np.diff(fR0_vals)[0]
    fR_edges = np.linspace(fR0_vals[0]-0.5*dfR0, fR0_vals[-1]+0.5*dfR0, 21)

    # calculate array of lnL values
    lnL_array = np.zeros((85, 20))
    summ_A = spam.analysis.open_summary('A')
    for j in range(20):
        summ_H = spam.analysis.open_summary('H'+str(j))
        for i in range(85):
            name = spam.data.names_standard[i]
            gal_H = summ_H.galaxies[name]
            gal_A = summ_A.galaxies[name]

            lnL_array[i, j] = gal_H.lnL-gal_A.lnL

    # plot colormesh and colorbar
    cmap = create_cmap([comp, 'white', green5])
    lnL_array = lnL_array[np.argsort(argmaxes)].T
    mesh = ax2.pcolormesh(gal_edges, fR_edges, lnL_array,
                          vmin=-5, vmax=5, cmap=cmap)
    plt.colorbar(mesh, cax2)
    clabel = r"$\ln\mathcal{L}_{f(R)}/\mathcal{L}_{\Lambda\mathrm{CDM}}$"
    cax2.set_ylabel(clabel)
    cax2.yaxis.set_label_position('right')

    # axis labels and limits
    ax1.set_ylim(ax2.get_ylim())
    ax1.set_xlim(ax2.get_xlim())
    ax1.tick_params(axis='x', labelbottom=False)
    ax2.set_xlabel(r"Galaxy \#")
    ax1.set_ylabel(r"$\log_{10} |\bar{f}_{R0}|$")
    ax2.set_ylabel(r"$\log_{10} |\bar{f}_{R0}|$")

    return fig
