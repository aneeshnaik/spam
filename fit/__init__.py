#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created: 2018
Author: A. P. Naik
Description: 'fit' submodule of spam package. See README for details and usage
examples.

Attributes
----------
GalaxyFit : class
    GalaxyFit is the class that prepares and executes MCMC runs, and stores
    resulting chains.
"""
from .fit import GalaxyFit

__all__ = ["GalaxyFit"]
