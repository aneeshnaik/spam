#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created: 2018
Author: A. P. Naik
Description: Figure 8 of Naik et al., 2019
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


def figure_8():
    """
    Figure showing change to fR0 by introducing environmental screening. One
    panel.
    """
    # loop over galaxies from full sample
    fR0_B_inc = np.array([])
    phi_B_inc = np.array([])
    fR0_C_inc = np.array([])
    phi_C_inc = np.array([])
    fR0_B_exc = np.array([])
    phi_B_exc = np.array([])
    fR0_C_exc = np.array([])
    phi_C_exc = np.array([])
    summ_B = spam.analysis.open_summary('B')
    summ_C = spam.analysis.open_summary('C')
    for name in spam.data.names_full:

        # get maxprob fR0 for both
        fR0_B = summ_B.galaxies[name].maxprob_theta[3]
        fR0_C = summ_C.galaxies[name].maxprob_theta[3]

        # interpolate phi_ext for both
        galaxy = spam.data.SPARCGalaxy(name)
        fR0_vals = np.linspace(np.log10(1.563e-8), np.log10(2e-6), 20)
        phi_B = np.interp(fR0_B, fR0_vals, galaxy.ext_potential)
        phi_C = np.interp(fR0_C, fR0_vals, galaxy.ext_potential)

        # append to either standard or full list
        if name in spam.data.names_standard:
            fR0_B_inc = np.append(fR0_B_inc, fR0_B)
            fR0_C_inc = np.append(fR0_C_inc, fR0_C)
            phi_B_inc = np.append(phi_B_inc, phi_B)
            phi_C_inc = np.append(phi_C_inc, phi_C)
        else:
            fR0_B_exc = np.append(fR0_B_exc, fR0_B)
            fR0_C_exc = np.append(fR0_C_exc, fR0_C)
            phi_B_exc = np.append(phi_B_exc, phi_B)
            phi_C_exc = np.append(phi_C_exc, phi_C)

    fig = plt.figure(figsize=(3.3, 3.3))
    fig.add_axes([0.2, 0.2, 0.75, 0.75])

    plt.scatter(phi_B_inc, fR0_B_inc,
                facecolors='none', edgecolors=green3, s=6)
    plt.scatter(phi_C_inc, fR0_C_inc,
                facecolors=green3, edgecolors=green3, s=6)
    plt.scatter(phi_B_exc, fR0_B_exc,
                facecolors='none', edgecolors='grey', s=6)
    plt.scatter(phi_C_exc, fR0_C_exc,
                facecolors='grey', edgecolors='grey', s=6)

    for i in range(85):
        plt.plot([phi_B_inc[i], phi_C_inc[i]], [fR0_B_inc[i], fR0_C_inc[i]],
                 ls='dashed', c=green4, lw=0.5)
    for i in range(62):
        plt.plot([phi_B_exc[i], phi_C_exc[i]], [fR0_B_exc[i], fR0_C_exc[i]],
                 ls='dashed', c='grey', lw=0.5)

    handles = [Line2D([0], [0], marker='.', lw=0, label='Env. Screening',
                      mfc='k', mec='k', ms=10),
               Line2D([0], [0], marker='.', lw=0, label='No Env. Screening',
                      mfc='none', mec='k', ms=10),
               Line2D([0], [0], marker='.', lw=0, label='Included Galaxies',
                      mfc=green3, mec=green3, ms=10),
               Line2D([0], [0], marker='.', lw=0, label='Excluded Galaxies',
                      mfc='grey', mec='grey', ms=10)]

    plt.legend(handles=handles, frameon=False)

    plt.xlim(-9.05, -4.95)
    plt.ylim(-9.75)
    plt.ylabel(r"$\log_{10}f_{R0}$")
    plt.xlabel(r"$\log_{10}\Phi_\mathrm{ext}/c^2$")

    return fig
