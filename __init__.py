#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created: 20th March 2019
Author: A. P. Naik
Description: init file of spam module. File is empty because everything is
contained with 'data', 'fit' or 'analysis' submodules.
"""
from . import data
from . import fit
from . import analysis

__all__ = ["data", "fit", "analysis"]
