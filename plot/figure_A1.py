#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created: 2018
Author: A. P. Naik
Description: Figure A1 of Naik et al., 2019
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


def figure_A1():

    # randomly selected galaxies for 2D test
    names = ['ESO116-G012', 'F574-1',   'NGC0300',  'NGC2841',  'NGC3741',
             'NGC6015',     'UGC00128', 'UGC05918', 'UGC08550', 'UGC10310']
    # bin edges for histogram
    bins = np.linspace(-9, np.log10(2e-6), 201)

    fig = plt.figure(figsize=(7, 7))
    for row in range(2):
        for col in range(5):

            # open fits
            name = names[5*row+col]
            fit_2D = spam.analysis.open_fit('2D', name)
            fit_1D = spam.analysis.open_fit('B', name)

            # add axes
            ax0 = fig.add_axes([0.05+0.18*col, 0.09+0.46*row, 0.18, 0.18])
            ax1 = fig.add_axes([0.05+0.18*col, 0.27+0.46*row, 0.18, 0.18])
            ax1.set_title(name)

            # get chains
            chain_2D = fit_2D.chain[0, :, -500:, -2].flatten()
            chain_1D = fit_1D.chain[0, :, -5000:, -2].flatten()

            # draw histograms
            ax0.hist(chain_2D, bins, color=comp)
            ax1.hist(chain_1D, bins, color=green3)

            # axis limits and labels
            ax0.set_xlim(-9.1, -5.6)
            ax1.set_xlim(-9.1, -5.6)
            ax0.set_xticks([-9, -8, -7, -6])
            ax1.set_xticks([-9, -8, -7, -6])
            ax0.tick_params('y', labelleft=False, left=False)
            ax1.tick_params('y', labelleft=False, left=False)
            ax1.tick_params('x', labelbottom=False)
            if col == 0:
                ax0.set_ylabel("Frequency")
                ax0.yaxis.set_label_coords(-0.05, 1)
            elif col == 2:
                ax0.set_xlabel(r'$\bar{f}_{R0}$')
            elif col == 4:
                ax0.text(1.05, 0.5, '2D', transform=ax0.transAxes, va='center')
                ax1.text(1.05, 0.5, '1D', transform=ax1.transAxes, va='center')

    # phantom bounding box to make sure bbox_inches cuts symmetrically
    fig.patches.extend([plt.Rectangle((0.025, 0.3), 0.95, 0.44, lw=0,
                                      fill=False, zorder=1000, edgecolor=None,
                                      transform=fig.transFigure, figure=fig)])

    return fig
