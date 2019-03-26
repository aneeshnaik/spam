#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created: 2018
Author: A. P. Naik
Description: 'data' submodule of spam package. See README for details and usage
examples.

Attributes
----------
SPARCGalaxy : class
    Main object containing all relevant data for each SPARC galaxy.
names_full : list of strings, length 147
    List of names of SPARC galaxies in 'full' sample, i.e. 147 galaxies
    remaining after first 4 data cuts described in Naik et al. (2019).
names_standard : list of strings, length 85
    List of names of SPARC galaxies in 'standard' sample, i.e. 85 galaxies
    remaining after all data cuts described in Naik et al. (2019). Difference
    between 'standard' and 'full' samples are that in the 'standard' case,
    environmentally screened galaxies have additionally been cut from the
    sample.
"""
import os as _os
import numpy as _np
from scipy.constants import parsec as _pc

# physical constants
_kpc = 1e+3*_pc
_Mpc = 1e+6*_pc
_Msun = 1.989e+30


class SPARCGalaxy:
    """
    Class containing all relevant data for a given galaxy. All data come from
    the SPARC database (http://astroweb.cwru.edu/SPARC/), with the following
    exceptions:

    - gas_radius : calculated in spam.data.fit_gas_disc.py
    - hernquist_radius : calculated in spam.data.fit_stellar_bulge.py
    - hernquist_rho_0 : ditto
    - stellar_expdisc_sigma_0 : calculated in spam.data.fit_stellar_disc.py
    - stellar_expdisc_R_d : ditto
    - ext_potential : calculated via the screening map of Desmond et al.
    - ext_potential_lower : ditto
    - ext_potential_upper : ditto

    Parameters
    ----------
    name : str
        Name of galaxy matching name in SPARC database, e.g. F574-1 or CamB.

    Attributes
    ----------
    name : str
        As above.
    hubble_type : str
        Hubble classification of galaxy.
    distance : float
        Distance to galaxy. UNITS: m
    distance_err : float
        Error on distance to galaxy. UNITS: m
    distance_method : int, {1, 2, 3, 4, 5}
        Method used to determine distance to galaxy (see SPARC database for
        meanings of numbers).
    inclination : float
        Inclination of galaxy. UNITS: degrees
    inclination_err : float
        Error on inclination. UNITS: degrees
    luminosity_tot : float
        Total luminosity of galaxy at 3.6mu. UNITS: 10^9 L_sun.
    luminosity_err : float
        Error on total luminosity. UNITS: 10^9 L_sun.
    disc_scale : float
        Scale length of disc fit to photometry data. UNITS: m
    disc_SB : float
        Central surface brightness of disc fit to photometry data. UNITS: m
    HI_mass : float
        Total mass of HI gas. UNITS: kg
    Q_flag : int
        Quality flag (see SPARC database).
    StellarBulge : bool
        Whether a bulge component is detected.
    R : 1D numpy.ndarray
        Radii of rotation curve measurements. UNITS: kpc
    v : 1D numpy.ndarray, shape same as R
        Rotation curve measurements. UNITS: km/s
    v_err : 1D numpy.ndarray, shape same as R
        Errors on rotation curve. UNITS: km/s
    v_gas : 1D numpy.ndarray, shape same as R
        Gas contribution to rotation curve. UNITS: km/s
    v_disc : 1D numpy.ndarray, shape same as R
        Stellar disc contribution to rotation curve, assuming mass-to-light
        ratio of 1 M_sun/L_sun. UNITS: km/s
    v_bul : 1D numpy.ndarray, shape same as R
        Stellar bulge contribution to rotation curve, assuming mass-to-light
        ratio of 1 M_sun/L_sun. Zero everywhere if StellarBulge is False.
        UNITS: km/s
    coords_RA : float
        Right ascension of galaxy. UNITS: degrees
    coords_DEC : float
        Declination of galaxy. UNITS: degrees
    gas_radius : float
        Best fit radius of gas disc, calculated in spam.data.fit_gas_disc.py.
        UNITS: m
    hernquist_radius : float
        Best fit radius of Hernquist bulge, calculated in
        spam.data.fit_stellar_bulge.py. UNITS: m
    hernquist_rho_0 : float
        Best fit central density of Hernquist bulge, calculated in
        spam.data.fit_stellar_bulge.py. UNITS: kg/m^3
    stellar_expdisc_sigma_0 : float
        Best fit central density of stellar disc, calculated in
        spam.data.fit_stellar_disc.py. UNITS: kg/m^2
    stellar_expdisc_R_d : float
        Best fit scale length of stellar disc, calculated in
        spam.data.fit_stellar_disc.py. UNITS: m
    ext_potential : float
        Maximum posterior external potential (specifically, log10(phi/c^2) )
        calculated via the screening map of Desmond et al. (see Naik et al.,
        2019 for details and refs).
    ext_potential_lower : float
        1 sigma lower bound on external potential.
    ext_potential_upper : ditto
        1 sigma upper bound on external potential.
    """
    def __init__(self, name):

        self.name = name
        datadir = _os.path.dirname(_os.path.realpath(__file__))+"/SPARCData"

        # loading metadata
        listfile = open(datadir+"/metadata.txt", 'r')
        data = listfile.readlines()
        listfile.close()

        names = []
        for i in range(len(data)):
            names.append(data[i].split()[0])
        ind = names.index(self.name)

        htypes = {0: 'S0', 1: 'Sa', 2: 'Sab', 3: 'Sb', 4: 'Sbc', 5: 'Sc',
                  6: 'Scd', 7: 'Sd', 8: 'Sdm', 9: 'Sm', 10: 'Im', 11: 'BCD'}
        self.hubble_type = htypes[int(data[ind].split()[1])]
        self.distance = float(data[ind].split()[2])*_Mpc  # metres
        self.distance_err = float(data[ind].split()[3])*_Mpc  # metres
        self.distance_method = int(data[ind].split()[4])
        self.inclination = float(data[ind].split()[5])  # degrees
        self.inclination_err = float(data[ind].split()[6])  # degrees
        self.luminosity_tot = float(data[ind].split()[7])  # 1e+9 Lsun
        self.luminosity_err = float(data[ind].split()[8])  # 1e+9 Lsun
        self.disc_scale = float(data[ind].split()[11])*_kpc  # metres
        self.disc_SB = float(data[ind].split()[12])/_pc**2  # Lsun/m^2
        self.HI_mass = float(data[ind].split()[13])*1e+9*_Msun  # kg
        self.Q_flag = int(data[ind].split()[17])

        # loading main SPARC data
        self.filename = datadir+"/data/"+name+"_rotmod.dat"
        gal_file = open(self.filename, 'r')
        data = gal_file.readlines()
        gal_file.close()
        self.R = _np.zeros((len(data[3:]),))
        self.v = _np.zeros((len(data[3:]),))
        self.v_err = _np.zeros((len(data[3:]),))
        self.v_gas = _np.zeros((len(data[3:]),))
        self.v_disc = _np.zeros((len(data[3:]),))
        self.v_bul = _np.zeros((len(data[3:]),))
        for i in range(len(data[3:])):
            self.R[i] = float(data[3:][i].split()[0])
            self.v[i] = float(data[3:][i].split()[1])
            self.v_err[i] = float(data[3:][i].split()[2])
            self.v_gas[i] = float(data[3:][i].split()[3])
            self.v_disc[i] = float(data[3:][i].split()[4])
            self.v_bul[i] = float(data[3:][i].split()[5])
        if (self.v_bul == 0).all():
            self.StellarBulge = False
        else:
            self.StellarBulge = True

        # loading coords
        coordfile = open(datadir+"/coords.txt", 'r')
        data = coordfile.readlines()[1:]
        coordfile.close()
        assert data[ind].split()[0] == self.name
        self.coords_RA = float(data[ind].split()[2])
        self.coords_DEC = float(data[ind].split()[3])

        # loading gas radius
        gasfile = open(datadir+"/gas_radii.txt", 'r')
        data = gasfile.readlines()
        gasfile.close()
        assert data[ind].split()[0] == self.name
        self.gas_radius = float(data[ind].split()[1])

        # loading hernquist parameters
        if self.StellarBulge:
            hernquistfile = open(datadir+"/hernquist_parameters.txt", 'r')
            data = hernquistfile.readlines()
            hernquistfile.close()
            assert data[ind].split()[0] == self.name
            self.hernquist_rho_0 = float(data[ind].split()[1])
            self.hernquist_radius = float(data[ind].split()[2])
        else:
            self.hernquist_rho_0 = None
            self.hernquist_radius = None

        # loading stellar disc fit parameters
        discparfile = open(datadir+"/stellar_disc_parameters.txt", 'r')
        data = discparfile.readlines()
        discparfile.close()
        assert data[ind].split()[0] == self.name
        self.stellar_expdisc_sigma_0 = float(data[ind].split()[1])  # kg/m^2
        self.stellar_expdisc_R_d = float(data[ind].split()[2])  # metres

        # loading external potential data
        potential_dir = datadir+"/SPARC_potentials"
        col1 = _np.array([], dtype=_np.float64)
        col2 = _np.array([], dtype=_np.float64)
        col3 = _np.array([], dtype=_np.float64)
        for i in range(20):
            file = open(potential_dir+"/SPARC_screen_"+str(i)+".dat", 'r')
            data = file.readlines()
            file.close()
            assert data[ind].split()[0] == self.name
            col1 = _np.append(col1, float(data[ind].split()[1]))
            col2 = _np.append(col2, float(data[ind].split()[2]))
            col3 = _np.append(col3, float(data[ind].split()[3]))
        self.ext_potential_lower = col1
        self.ext_potential = col2
        self.ext_potential_upper = col3

        return


# getting list of galaxy names
names_full = []
names_standard = []
_datadir = _os.path.dirname(_os.path.realpath(__file__))+"/SPARCData"
_namefile = open(_datadir+"/names_full.txt", 'r')
_data = _namefile.readlines()
_namefile.close()
names_full = []
for _i in range(len(_data)):
    names_full.append(_data[_i].split()[0])
_namefile = open(_datadir+"/names_standard.txt", 'r')
_data = _namefile.readlines()
_namefile.close()
names_standard = []
for _i in range(len(_data)):
    names_standard.append(_data[_i].split()[0])

__all__ = ['SPARCGalaxy', 'names_full', 'names_standard']
