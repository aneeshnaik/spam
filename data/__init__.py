#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created: 20th March 2019
Author: A. P. Naik
Description: 'data' submodule of spam package. See README for details and usage
examples.

Attributes
----------
names_full : list of strings, length 147
    List of names of SPARC galaxies in 'full' sample, i.e. 147 galaxies
    remaining after first 4 data cuts described in Naik et al. (2019).
names_standard : list of strings, length 85
    List of names of SPARC galaxies in 'standard' sample, i.e. 85 galaxies
    remaining after all data cuts described in Naik et al. (2019). Difference
    between 'standard' and 'full' samples are that in the 'standard' case,
    environmentally screened galaxies have additionally been cut from the
    sample.
sample_full : list of .data.SPARCGalaxy instances, length 147
    A list of SPARCGalaxy instances for each of the 147 galaxies listed in
    names_full. Each instance contains all SPARC data for the galaxy.
sample_standard : list of .data.SPARCGalaxy instances, length 85
    Same as sample_full above, but for the 'standard' sample.
"""
from .data import names_full, names_standard, sample_full, sample_standard

__all__ = ['names_full', 'names_standard', 'sample_full', 'sample_standard']
