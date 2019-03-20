#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created: October 2018
Author: A. P. Naik
Description: 2D and 1D solvers for Hu-Sawicki f(R) equations of motion.
"""
import numpy as np
from scipy.constants import parsec as pc
from scipy.constants import gravitational_constant as G
from scipy.constants import c
import time
kpc = 1.0e3 * pc
Mpc = 1.0e6 * pc
c2 = c*c
msun = 1.9884430e30


class grid_2D:

    def __init__(self, ngrid=500, rmin=1*kpc, rmax=30*Mpc):
        """
        Sets up the 2D grid. 'ngrid' is number of radial grid points. These are
        log-spaced from ln(rmin) to ln(rmax). Number of angular grid points
        is hard coded below as 'nth'.

        Creates objects such as 'rgrid', which is an ngrid x nth array, i.e.
        rgrid[i][j] gives the radius of the ijth (ith r, jth theta) point.
        """

        self.ngrid = ngrid
        self.rmax = rmax
        self.rmin = rmin

        # radial direction: log spaced grid from ln(rmin) to ln(rmax)
        # xin and xout are the inner and outer edges of each grid cell
        xmax = np.log(rmax)
        xmin = np.log(rmin)
        self.dx = (xmax-xmin)/ngrid
        self.x = xmin + (np.arange(self.ngrid, dtype=np.float64)+0.5)*self.dx
        self.xout = xmin + (np.arange(self.ngrid, dtype=np.float64)+1)*self.dx
        self.xin = xmin + np.arange(self.ngrid, dtype=np.float64)*self.dx

        self.r = np.exp(self.x)

        # theta direction: evenly spaced from 0 to pi; number of points is nth
        # self.nth = 2999  # fixed s.t. r*dtheta is << disc height in galaxy
        self.nth = 101
        self.disc_ind = self.nth//2
        self.dth = np.pi/self.nth
        self.th = (np.arange(self.nth, dtype=np.float64)+0.5)*self.dth
        self.thin = np.arange(self.nth, dtype=np.float64)*self.dth
        self.thout = (np.arange(self.nth, dtype=np.float64)+1)*self.dth
        self.rout = np.exp(self.xout)
        self.rin = np.exp(self.xin)
        self.grid_shape = (self.ngrid, self.nth)

        # set up grid structure; any object, e.g. rgrid, is a ngrid x nth array
        self.rgrid, self.thgrid = np.meshgrid(self.r, self.th, indexing='ij')
        self.ringrid, self.thingrid = np.meshgrid(self.rin, self.thin,
                                                  indexing='ij')
        self.routgrid, self.thoutgrid = np.meshgrid(self.rout, self.thout,
                                                    indexing='ij')
        self.sthgrid = np.sin(self.thgrid)
        self.sthingrid = np.sin(self.thingrid)
        self.sthoutgrid = np.sin(self.thoutgrid)

        # coefficient of each term in the discretised Laplacian
        self.coeff1 = self.routgrid/(self.dx**2*self.rgrid**3)
        self.coeff2 = self.ringrid/(self.dx**2*self.rgrid**3)
        self.coeff3 = self.sthoutgrid/(self.dth**2*self.rgrid**2*self.sthgrid)
        self.coeff4 = self.sthingrid/(self.dth**2*self.rgrid**2*self.sthgrid)

        # dLdu_const gives the final constant term in the Newton Raphson
        # expression for dL/du
        d1 = (self.rgrid**3*self.dx**2)
        d2 = (self.rgrid**2*self.sthgrid*self.dth**2)
        self.dLdu_const = ((self.ringrid+self.routgrid)/d1 +
                           (self.sthingrid+self.sthoutgrid)/d2)

        # dvol is cell volume, full vol is volume of whole sphere
        self.dvol = 2*np.pi*self.rgrid**3*np.sin(self.thgrid)*self.dx*self.dth
        self.fullvol = 4.0/3.0*np.pi*(self.rmax**3 - self.rmin**3)

        self.GuessFlag = False

        return

    def set_cosmology(self, h, omega_m, redshift=0):
        """
        Assigns a background cosmology to the grid
        """

        self.h = h
        self.omega_m = omega_m
        self.omega_l = 1 - omega_m
        self.redshift = redshift

        self.H0 = self.h * 100.0 * 1000.0 / Mpc

        # rhocrit is evaluated today, rhomean at given redshift
        self.rhocrit = 3.0 * self.H0**2 / (8.0 * np.pi * G)
        self.rhomean = (1+self.redshift)**3*self.omega_m * self.rhocrit

        return

    def set_NFW_profile(self, halomass, c_nfw):
        """
        Sets up the density profile for a NFW profile with M200 = halomass,
        and concentration parameter = c_nfw.
        """

        self.halomass = halomass
        self.c_nfw = c_nfw

        # calculate r200 and r_scale
        rhocrit = self.rhomean/self.omega_m
        self.r200c = (halomass/(200*rhocrit)*3/(4*np.pi)) ** (1/3)
        self.rscale = self.r200c/self.c_nfw

        # calculate rho0
        ind = np.where(self.rgrid <= self.r200c)
        ratio = self.rgrid/self.rscale
        weight = self.dvol/(ratio*(1.0+ratio)**2)
        self.rho0 = self.halomass/np.sum(weight[ind])

        # create NFW profile
        dmass = self.rhomean*self.dvol
        dmass += self.rho0*weight
        dmass -= np.sum(self.rho0*weight)/self.fullvol*self.dvol
        self.dmass = dmass

        # adjust slightly to ensure overall mean density is same as cosmic
        self.rho = self.dmass/self.dvol
        self.drho = self.rho - self.rhomean
        self.weight = weight

        return

    def laplacian(self, expu):
        """
        Calculates discretised laplacian of e^u. The coefficients coeff1 etc.
        are initialised in __init__ above.
        """

        # d(e^u)/dx. BCs: vanishes at both boundaries
        deudx = np.zeros((self.ngrid+1, self.nth))

        deudx[1:-1, :] = (expu[1:, :] - expu[:-1, :])

        # d(e^u)/dtheta. BCs: vanishes at both boundaries
        deudth = np.zeros((self.ngrid, self.nth+1))
        deudth[:, 1:-1] = (expu[:, 1:] - expu[:, :-1])

        # discretised laplacian
        D2expu = deudx[1:, :]*self.coeff1 - deudx[:-1, :]*self.coeff2
        D2expu += deudth[:, 1:]*self.coeff3 - deudth[:, :-1]*self.coeff4

        return D2expu

    def newton(self, expu, D2expu):
        """
        Performs the Newton-Raphson step, i.e. du = - L / (dL/du)
        """

        # Newton-Raphson step, as in MG-GADGET paper
        oneoversqrteu = 1/np.sqrt(expu)
        L = D2expu + self.const1*(1.0-oneoversqrteu) - self.const2*self.drho
        dLdu = 0.5*self.const1*oneoversqrteu - expu*self.dLdu_const
        du = - L / dLdu

        return du

    def set_loginterpolated_drho(self, snap_R, snap_drho):

        assert snap_R[0] < self.r[0]
        assert snap_R[-1] > self.r[-1]
        assert np.shape(snap_drho) == (len(snap_R), self.nth)

        drho = np.zeros(np.shape(self.rgrid))

        for i in range(self.ngrid):
            k = np.where(snap_R < self.r[i])[0][-1]
            x1 = np.log(snap_R[k])
            x2 = np.log(snap_R[k+1])
            y1 = snap_drho[k, :]
            y2 = snap_drho[k+1, :]

            x = np.log(self.r[i])
            drho[i, :] = y1 + (y2-y1)*(x-x1)/(x2-x1)

        self.drho = drho

        return

    def iter_solve(self, niter, F0, verbose=False, tol=1e-7):
        """
        Iteratively solves scalar field equation of motion on grid.
        """

        # relevant constants
        msq = self.omega_m*self.H0*self.H0
        self.Ra = 3*msq*((1+self.redshift)**3 + 4*self.omega_l/self.omega_m)
        self.R0 = 3*msq*(1 + 4*self.omega_l/self.omega_m)
        self.F0 = F0
        self.Fa = self.F0*(self.R0/self.Ra)**2
        self.const1 = self.Ra / (3.0 * c2 * self.Fa)
        self.const2 = -8.0*np.pi*G / (3.0 * c2 * self.Fa)

        # u = ln(fR/Fa)
        if self.GuessFlag:
            u = self.u_guess
        else:
            u = np.zeros((self.ngrid, self.nth))

        # main loop
        t0 = time.time()
        for i in np.arange(niter):

            expu = np.exp(u)

            D2expu = self.laplacian(expu)

            du = self.newton(expu, D2expu)

            if((abs(du) > 1).any()):
                ind = np.where(du > 1.0)
                du[ind] = 1.0
                ind = np.where(du < -1.0)
                du[ind] = -1.0
            elif abs(du).max() < tol:
                break

            u += du

            if(verbose and i % 1000 == 0):
                print("iteration", i, ", max(|du|)=", abs(du).max())

        assert i != niter-1, "Solver took too long!"

        t1 = time.time()

        # output results
        expu = np.exp(u)

        self.D2expu = self.laplacian(expu)
        self.u = u
        self.fR = self.Fa*np.exp(u)
        self.dR = self.Ra * (np.sqrt(self.Fa/self.fR) - 1.0)
        self.time_taken = t1-t0
        self.iters_taken = i

        if verbose:
            print("Took ", self.time_taken, " seconds")

        return


class grid_1D:

    def __init__(self, ngrid=500, rmin=1*kpc, rmax=30*Mpc):
        """
        Sets up the 1D grid. 'ngrid' is number of radial grid points. These are
        log-spaced from ln(rmin) to ln(rmax).
        """

        self.ngrid = ngrid
        self.rmax = rmax
        self.rmin = rmin
        self.grid_shape = (self.ngrid,)

        # radial direction: log spaced grid from ln(rmin) to ln(rmax)
        # xin and xout are the inner and outer edges of each grid cell
        xmax = np.log(rmax)
        xmin = np.log(rmin)
        self.dx = (xmax-xmin)/ngrid
        self.x = xmin + (np.arange(self.ngrid, dtype=np.float64)+0.5)*self.dx
        self.xout = xmin + (np.arange(self.ngrid, dtype=np.float64)+1)*self.dx
        self.xin = xmin + np.arange(self.ngrid, dtype=np.float64)*self.dx

        self.rgrid = np.exp(self.x)

        self.rout = np.exp(self.xout)
        self.rin = np.exp(self.xin)

        # coefficient of each term in the discretised Laplacian
        self.coeff1 = self.rout/(self.dx**2*self.rgrid**3)
        self.coeff2 = self.rin/(self.dx**2*self.rgrid**3)

        # dLdu_const gives the final constant term in the Newton Raphson
        # expression for dL/du
        d1 = (self.rgrid**3*self.dx**2)
        self.dLdu_const = (self.rin+self.rout)/d1

        # dvol is cell volume, full vol is volume of whole sphere
        self.dvol = 4.0/3.0*np.pi*(self.rout**3-self.rin**3)
        self.fullvol = 4.0/3.0*np.pi*(self.rmax**3 - self.rmin**3)

        self.GuessFlag = False

        return

    def set_cosmology(self, h, omega_m, redshift=0):
        """
        Assigns a background cosmology to the grid
        """

        self.h = h
        self.omega_m = omega_m
        self.omega_l = 1 - omega_m
        self.redshift = redshift

        self.H0 = self.h * 100.0 * 1000.0 / Mpc

        # rhocrit is evaluated today, rhomean at given redshift
        self.rhocrit = 3.0 * self.H0**2 / (8.0 * np.pi * G)
        self.rhomean = (1+self.redshift)**3*self.omega_m * self.rhocrit

        return

    def laplacian(self, expu):
        """
        Calculates discretised laplacian of e^u. The coefficients coeff1 etc.
        are initialised in __init__ above.
        """

        # d(e^u)/dx. BCs: vanishes at both boundaries
        deudx = np.zeros(self.ngrid+1,)
        deudx[1:-1] = (expu[1:] - expu[:-1])

        # discretised laplacian
        D2expu = deudx[1:]*self.coeff1 - deudx[:-1]*self.coeff2

        return D2expu

    def newton(self, expu, D2expu):
        """
        Performs the Newton-Raphson step, i.e. du = - L / (dL/du)
        """

        # Newton-Raphson step, as in MG-GADGET paper
        oneoversqrteu = 1/np.sqrt(expu)
        L = D2expu + self.const1*(1.0-oneoversqrteu) - self.const2*self.drho
        dLdu = 0.5*self.const1*oneoversqrteu - expu*self.dLdu_const
        du = - L / dLdu

        return du

    def iter_solve(self, niter, F0, verbose=False, tol=1e-7):
        """
        Iteratively solves scalar field equation of motion on grid.
        """

        # relevant constants
        msq = self.omega_m*self.H0*self.H0
        self.Ra = 3*msq*((1+self.redshift)**3 + 4*self.omega_l/self.omega_m)
        self.R0 = 3*msq*(1 + 4*self.omega_l/self.omega_m)
        self.F0 = F0
        self.Fa = self.F0*(self.R0/self.Ra)**2
        self.const1 = self.Ra / (3.0 * c2 * self.Fa)
        self.const2 = -8.0*np.pi*G / (3.0 * c2 * self.Fa)

        # u = ln(fR/Fa)
        if self.GuessFlag:
            u = self.u_guess
        else:
            u = np.zeros(self.ngrid,)

        # main loop
        t0 = time.time()
        for i in np.arange(niter):

            expu = np.exp(u)

            D2expu = self.laplacian(expu)

            du = self.newton(expu, D2expu)

            if((abs(du) > 1).any()):
                ind = np.where(du > 1.0)
                du[ind] = 1.0
                ind = np.where(du < -1.0)
                du[ind] = -1.0
            elif abs(du).max() < tol:
                break

            u += du

            if(verbose and i % 1000 == 0):
                print("iteration", i, ", max(|du|)=", abs(du).max())

        assert i != niter-1, "Solver took too long!"

        t1 = time.time()

        # output results
        expu = np.exp(u)

        self.D2expu = self.laplacian(expu)
        self.u = u
        self.fR = self.Fa*np.exp(u)
        self.dR = self.Ra * (np.sqrt(self.Fa/self.fR) - 1.0)
        self.time_taken = t1-t0
        self.iters_taken = i

        if verbose:
            print("Took ", self.time_taken, " seconds")

        return
