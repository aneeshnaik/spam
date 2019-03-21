#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created: 21st March 2019
Author: A. P. Naik
Description: Data file
"""
import os
import numpy as np
from scipy.constants import parsec as pc
kpc = 1e+3*pc
Mpc = 1e+6*pc
Msun = 1.989e+30


class SPARCGalaxy:
    """
    Object containing all available SPARC data for a single SPARC galaxy.
    """
    def __init__(self, name):

        self.name = name

        # loading metadata
        listfile = open(datadir+"/list.txt", 'r')
        data = listfile.readlines()
        listfile.close()

        names = []
        for i in range(len(data)):
            names.append(data[i].split()[0])
        ind = names.index(self.name)
        self.index = ind

        htypes = {0: 'S0', 1: 'Sa', 2: 'Sab', 3: 'Sb', 4: 'Sbc', 5: 'Sc',
                  6: 'Scd', 7: 'Sd', 8: 'Sdm', 9: 'Sm', 10: 'Im', 11: 'BCD'}
        self.hubble_type = htypes[int(data[ind].split()[1])]
        self.distance = float(data[ind].split()[2])*Mpc  # metres
        self.distance_err = float(data[ind].split()[3])*Mpc  # metres
        self.distance_method = int(data[ind].split()[4])
        self.inclination = float(data[ind].split()[5])  # degrees
        self.inclination_err = float(data[ind].split()[6])  # degrees
        self.luminosity_tot = float(data[ind].split()[7])  # 1e+9 Lsun
        self.luminosity_err = float(data[ind].split()[8])  # 1e+9 Lsun
        self.effective_radius = float(data[ind].split()[9])*kpc  # metres
        self.effective_SB = float(data[ind].split()[10])/pc**2  # Lsun/m^2
        self.disc_scale = float(data[ind].split()[11])*kpc  # metres
        self.disc_SB = float(data[ind].split()[12])/pc**2  # Lsun/m^2
        self.HI_mass = float(data[ind].split()[13])*1e+9*Msun  # kg
        self.HI_radius = float(data[ind].split()[14])*kpc
        self.v_flat = float(data[ind].split()[15])  # km/s
        self.v_flat_err = float(data[ind].split()[16])  # km/s
        self.Q_flag = int(data[ind].split()[17])
        self.HI_ref = data[ind].split()[18]

        # loading main SPARC data
        self.filename = datadir+"/data/"+name+"_rotmod.dat"
        gal_file = open(self.filename, 'r')
        data = gal_file.readlines()
        gal_file.close()
        self.R = np.zeros((len(data[3:]),))
        self.v = np.zeros((len(data[3:]),))
        self.v_err = np.zeros((len(data[3:]),))
        self.v_gas = np.zeros((len(data[3:]),))
        self.v_disc = np.zeros((len(data[3:]),))
        self.v_bul = np.zeros((len(data[3:]),))
        self.SB_disc = np.zeros((len(data[3:]),))
        self.SB_bul = np.zeros((len(data[3:]),))
        for i in range(len(data[3:])):
            self.R[i] = float(data[3:][i].split()[0])
            self.v[i] = float(data[3:][i].split()[1])
            self.v_err[i] = float(data[3:][i].split()[2])
            self.v_gas[i] = float(data[3:][i].split()[3])
            self.v_disc[i] = float(data[3:][i].split()[4])
            self.v_bul[i] = float(data[3:][i].split()[5])
            self.SB_disc[i] = float(data[3:][i].split()[6])/pc**2  # Lsun/m^2
            self.SB_bul[i] = float(data[3:][i].split()[7])/pc**2  # Lsun/m^2
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

        # loading bulge/disc decomposition data
        decompfile = open(datadir+"/bulge_disc_decompositions/"+name+".dens")
        data = decompfile.readlines()[1:]
        decompfile.close()
        self.decomp_R = np.zeros((len(data),))
        self.decomp_SB_disc = np.zeros((len(data),))
        self.decomp_SB_bul = np.zeros((len(data),))
        for i in range(len(data)):
            line = data[i].split()
            self.decomp_R[i] = float(line[0])
            self.decomp_SB_disc[i] = float(line[1])/pc**2  # Lsun/m^2
            self.decomp_SB_bul[i] = float(line[2])/pc**2  # Lsun/m^2

        # loading photometry data
        photofile = open(datadir+"/photometric_profiles/"+name+".sfb")
        data = photofile.readlines()[1:]
        photofile.close()
        self.photometry_theta = np.zeros((len(data),))
        self.photometry_mu = np.zeros((len(data),))
        self.photometry_kill = np.zeros((len(data),))
        self.photometry_err = np.zeros((len(data),))
        for i in range(len(data)):
            line = data[i].split()
            self.photometry_theta[i] = float(line[0])
            self.photometry_mu[i] = float(line[1])
            self.photometry_kill[i] = int(line[2])
            self.photometry_err[i] = float(line[3])
        a = np.pi/648000
        self.photometry_R = self.distance*np.tan(a*self.photometry_theta)/kpc

        # loading external potential data
        potential_dir = datadir+"/SPARC_potentials"
        col1 = np.array([], dtype=np.float64)
        col2 = np.array([], dtype=np.float64)
        col3 = np.array([], dtype=np.float64)
        for i in range(20):
            file = open(potential_dir+"/SPARC_screen_"+str(i)+".dat", 'r')
            data = file.readlines()
            file.close()
            assert data[self.index].split()[0] == self.name
            col1 = np.append(col1, float(data[self.index].split()[1]))
            col2 = np.append(col2, float(data[self.index].split()[2]))
            col3 = np.append(col3, float(data[self.index].split()[3]))
        self.ext_potential_lower = col1
        self.ext_potential = col2
        self.ext_potential_upper = col3

        return


datadir = os.path.dirname(os.path.realpath(__file__))+"/SPARCData"

names_full = []
names_standard = []
sample_full = []
sample_standard = []


# getting list of galaxy names
namefile = open(datadir+"/names_full.txt", 'r')
data = namefile.readlines()
namefile.close()
names_full = []
for i in range(len(data)):
    names_full.append(data[i].split()[0])
namefile = open(datadir+"/names_standard.txt", 'r')
data = namefile.readlines()
namefile.close()
names_standard = []
for i in range(len(data)):
    names_standard.append(data[i].split()[0])

# creating list of SPARCGalaxy instances (loading data for each)
for name in names_full:
    sample_full.append(SPARCGalaxy(name))
for name in names_standard:
    sample_standard.append(SPARCGalaxy(name))
