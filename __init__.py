#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created: 2018
Author: A. P. Naik
Description: 'spam' package, designed to search for observable imprints of
Hu-Sawicki f(R) gravity in the rotation curves of the SPARC sample. All
functionality is contained within the 'data', 'fit', and 'analysis' submodules.

See submodule documentation and README for details and usage examples.

SPAM is free software made available under the MIT license. For details see
LICENSE.

If you make use of SPAM in your work, please cite our paper, Naik et al. (2019)
"""
from . import data
from . import fit
from . import analysis
from . import plot

__all__ = ["data", "fit", "analysis", "plot"]
