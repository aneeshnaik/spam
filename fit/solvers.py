#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created: 2018
Author: A. P. Naik
Description: File containing two objects: 'grid_1D' and 'grid_2D', which are
respectively numerical solvers for the the Hu-Sawicki f(R) equations of motion,
on discretised grids in one and two dimensions respectively.
"""
import numpy as np
from scipy.constants import parsec as pc
from scipy.constants import gravitational_constant as G
from scipy.constants import c
import time

# physical constants
kpc = 1.0e3 * pc
Mpc = 1.0e6 * pc
c2 = c*c
msun = 1.9884430e30


class Grid1D:
    """
    Class which solves the Hu-Sawicki f(R) equations of motion on a discretised
    1D grid, i.e. assuming spherical symmetry.

    Grid consists of ngrid cells, logspaced from rmin to rmax, and EOMs are
    solved using a Newton-Gauss-Seidel (NGS) relaxation method, essentially a
    1D version of the code found in MG-GADGET, but without the multi-grid
    acceleration.

    Parameters
    ----------
    ngrid : int
        Number of radial grid cells. Default: 175.
    rmin : float
        Inner radius of grid. Default: 50 parsecs. UNITS: m
    rmax : float
        Outer radius of grid. Default: 5 Megaparsecs. UNITS: m

    Attributes
    ----------
    All parameters described above are stored as attributes, in addition to:

    grid_shape : tuple
        Tuple giving shape of grid, which in this 1D case is simply (ngrid)
    rgrid : numpy.ndarray, shape: grid_shape
        Radial coordinates of grid cell centres.
    rin : numpy.ndarray, shape: grid_shape
        Radial coordinates of inner edges of grid cells.
    rout : numpy.ndarray, shape: grid_shape
        Radial coordinates of outer edges of grid cells.
    dx : float
        Spacing between x=ln(r) coordinate of grid cells.
    x : numpy.ndarray, shape: grid_shape
        x=ln(r) coordinates of grid cell centres.
    xin : numpy.ndarray, shape: grid_shape
        x=ln(r) coordinates of inner edges of grid cells.
    xout : numpy.ndarray, shape: grid_shape
        x=ln(r) coordinates of outer edges of grid cells.
    fR : numpy.ndarray, shape: grid_shape
        After iter_solve has been run, scalar field solution fR is given here.
    u : numpy.ndarray, shape: grid_shape
        u = ln(fR/fRa)
    u_guess : numpy.ndarray, shape: grid_shape
        Optional initial guess for u. To be externally set by user before
        running iter_solve. After setting this, user should set GuessFlag to
        True.
    GuessFlag
        Whether a u_guess has been provided.
    time_taken : float
        Time taken by iter_solve to arrive at scalar field solution
    iters_taken : int
        Number of NGS iterations taken to arrive at scalar field solution.
    dvol : numpy.ndarray, shape: grid_shape
        Volumes of grid cells. UNITS: m^3
    fullvol : float
        Volume of whole grid. UNITS: m^3
    h : float
        Dimensionless Hubble constant, e.g. 0.7. Set in set_cosmology method.
    H0 : float
        Dimensional Hubble constant. Set in set_cosmology method. UNITS: s^-1
    omega_m : float
        Matter fraction, e.g. 0.3. Set in set_cosmology method.
    omega_l : float
        Dark energy fraction, e.g. 0.7. Set in set_cosmology method.
    redshift :
        Current redshift at time of solution. Set in set_cosmology method.
    rhocrit : float
        Cosmic critical density at redshift 0. Set in set_cosmology method.
    rhomean : float
        Mean matter density at solution redshift. Set in set_cosmology method.
    Ra : float
        Cosmic background curvature at solution redshift. Set in iter_solve.
    R0 : float
        Cosmic background curvature at redshift 0. Set in iter_solve.
    Fa : float
        Background scalar field at solution redshift. Set in iter_solve.
    F0 : float
        fR0. Given as argument to iter_solve.

    Methods
    -------
    __init__ :
        Initialise an instance of Grid1D
    set_cosmology :
        Set the cosmological parameters.
    laplacian :
        Internal method to calculate laplacian during scalar field solution.
    newton :
        Internal method to perform NGS iteration during scalar field solution.
    iter_solve :
        Solve for the scalar field using NGS method.
    """

    def __init__(self, ngrid=175, rmin=0.05*kpc, rmax=5*Mpc):
        """
        Initialise an instance of Grid1D class, see Grid1D docstring for more
        info.
        """

        self.ngrid = ngrid
        self.rmax = rmax
        self.rmin = rmin
        self.grid_shape = (self.ngrid,)

        # radial direction: log spaced grid from xmin=ln(rmin) to xmax=ln(rmax)
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
        self.__coeff1 = self.rout/(self.dx**2*self.rgrid**3)
        self.__coeff2 = self.rin/(self.dx**2*self.rgrid**3)

        # dLdu_const gives the final constant term in the Newton Raphson
        # expression for dL/du
        d1 = (self.rgrid**3*self.dx**2)
        self.__dLdu_const = (self.rin+self.rout)/d1

        # dvol is cell volume, full vol is volume of whole sphere
        self.dvol = 4.0/3.0*np.pi*(self.rout**3-self.rin**3)
        self.fullvol = 4.0/3.0*np.pi*(self.rmax**3 - self.rmin**3)

        # user change to True if setting a guess
        self.GuessFlag = False

        return

    def set_cosmology(self, h, omega_m, redshift=0):
        """
        Assign a background cosmology to the grid: Hubble constant, redshift,
        omega_m are input, then rho_crit and rho_mean are calculated.

        Parameters
        ----------
        h : float
            Dimensionless Hubble parameter, e.g. 0.7
        omega_m : float
            Cosmic matter fraction, e.g. 0.3
        redshift : float
            Redshift at time of solution. Default: 0.
        """

        self.h = h
        self.omega_m = omega_m
        self.omega_l = 1 - omega_m
        self.redshift = redshift

        # calculate dimensional Hubble constant
        self.H0 = self.h * 100.0 * 1000.0 / Mpc

        # rhocrit is evaluated today, rhomean at given redshift
        self.rhocrit = 3.0 * self.H0**2 / (8.0 * np.pi * G)
        self.rhomean = (1+self.redshift)**3*self.omega_m * self.rhocrit

        return

    def laplacian(self, expu):
        """
        Calculate discretised laplacian of e^u. The coefficients coeff1 etc.
        are initialised in __init__ above. The boundary conditions are
        implemented here: first derivative of u vanishes at inner and outer
        boundaries.

        Parameters
        ----------
        expu : numpy.ndarray, shape grid_shape
            Exponential of u=ln(fR/fa)

        Returns
        -------
        D2expu : numpy.ndarray, shape grid_shape
            Discretised Laplacian of e^u
        """

        # d(e^u)/dx. BCs: vanishes at both boundaries
        deudx = np.zeros(self.ngrid+1,)
        deudx[1:-1] = (expu[1:] - expu[:-1])

        # discretised laplacian
        D2expu = deudx[1:]*self.__coeff1 - deudx[:-1]*self.__coeff2
        return D2expu

    def newton(self, expu, D2expu):
        """
        Calculate the change in u this iteration, i.e. du = - L / (dL/du)

        Parameters
        ----------
        expu : numpy.ndarray, shape grid_shape
            Exponential of u=ln(fR/fa)
        D2expu : numpy.ndarray, shape grid_shape
            Discretised Laplacian of e^u, as calculated in 'laplacian' method

        Returns
        -------
        du : numpy.ndarray, shape grid_shape
            Required change to u this iteration.
        """

        # Newton-Raphson step, as in MG-GADGET paper
        oneoverrteu = 1/np.sqrt(expu)
        L = D2expu + self.__const1*(1.0-oneoverrteu) - self.__const2*self.drho
        dLdu = 0.5*self.__const1*oneoverrteu - expu*self.__dLdu_const
        du = - L / dLdu

        return du

    def iter_solve(self, niter, F0, verbose=False, tol=1e-7):
        """
        Iteratively solve the scalar field equation of motion on grid, using
        a Newton-Gauss-Seidel relaxation technique, as in MG-GADGET. Perform
        iterations until the change in u=ln(fR/fa) is everywhere below the
        threshold specified by the parameter 'tol'.

        Parameters
        ----------
        niter : int
            Max number of NGS iterations, beyond which the computation stops
            with an error code.
        F0 : float
            Cosmic background value of the scalar field. Should be negative,
            e.g. -1e-6 for F6.
        verbose : bool
            Whether to print progress update periodically. Default: False.
        tol : float
            Tolerance level for iterative changes in u. Once du is everywhere
            below this threshold, the computation stops. Default: 1e-7.
        """

        # relevant constants
        msq = self.omega_m*self.H0*self.H0
        self.Ra = 3*msq*((1+self.redshift)**3 + 4*self.omega_l/self.omega_m)
        self.R0 = 3*msq*(1 + 4*self.omega_l/self.omega_m)
        self.F0 = F0
        self.Fa = self.F0*(self.R0/self.Ra)**2
        self.__const1 = self.Ra / (3.0 * c2 * self.Fa)
        self.__const2 = -8.0*np.pi*G / (3.0 * c2 * self.Fa)

        # u = ln(fR/Fa)
        if self.GuessFlag:
            u = self.u_guess
        else:
            u = np.zeros(self.ngrid,)

        # main loop
        t0 = time.time()
        for i in np.arange(niter):

            # calculate change du
            expu = np.exp(u)
            D2expu = self.laplacian(expu)
            du = self.newton(expu, D2expu)

            # check if du is too high or sufficiently low
            if((abs(du) > 1).any()):
                ind = np.where(du > 1.0)
                du[ind] = 1.0
                ind = np.where(du < -1.0)
                du[ind] = -1.0
            elif abs(du).max() < tol:
                break

            # NGS update
            u += du

            if(verbose and i % 1000 == 0):
                print("iteration", i, ", max(|du|)=", abs(du).max())

        t1 = time.time()
        if i == niter-1:
            raise Exception("Solver took too long!")

        # output results
        self.u = u
        self.fR = self.Fa*np.exp(u)
        self.time_taken = t1-t0
        self.iters_taken = i

        if verbose:
            print("Took ", self.time_taken, " seconds")

        return


class Grid2D:
    """
    Class which solves the Hu-Sawicki f(R) equations of motion on a discretised
    2D grid, i.e. assuming azimuthal symmetry.

    Radial grid consists of ngrid cells, logspaced from rmin to rmax, and polar
    grid consists of 101 cells, from theta=0 to theta=pi, such that cell 50
    (starting indexing at 0) is the disc plane. EOMs are solved using a
    Newton-Gauss-Seidel (NGS) relaxation method, essentially a 1D version of
    the code found in MG-GADGET, but without the multi-grid acceleration.

    Parameters
    ----------
    ngrid : int
        Number of radial grid cells. Default: 175.
    rmin : float
        Inner radius of grid. Default: 50 parsecs. UNITS: m
    rmax : float
        Outer radius of grid. Default: 5 Megaparsecs. UNITS: m

    Attributes
    ----------
    All parameters described above are stored as attributes, in addition to:

    nth : int
        Number of polar grid cells. Currently, this is hard-coded as 101
    grid_shape : tuple
        Tuple giving shape of grid, i.e. (ngrid, nth)
    disc_ind : int
        Index of polar cell corresponding to disc plane, i.e. nth-1/2.
    dth : float
        Polar grid cell spacing
    th : numpy.ndarray, shape: (ngrid,)
        1D array of polar coordinates of cell centres.
    thin : numpy.ndarray, shape: (ngrid,)
        1D array of polar coordinates of cell inner edges.
    thout : numpy.ndarray, shape: (ngrid,)
        1D array of polar coordinates of cell outer edges.
    sthgrid : numpy.ndarray, shape: grid_shape
        2D array of sin(theta) values at all cell centres
    sthingrid : numpy.ndarray, shape: grid_shape
        2D array of sin(theta) values at inner edges
    sthoutgrid : numpy.ndarray, shape: grid_shape
        2D array of sin(theta) values at outer edges
    r : numpy.ndarray, shape: (ngrid,)
        1D array of radial coordinates
    rgrid : numpy.ndarray, shape: grid_shape
        2D array of radial coordinates of all grid cell centres.
    rin : numpy.ndarray, shape: (ngrid,)
        1D array of radial coordinates of inner edges of grid cells.
    ringrid : numpy.ndarray, shape: grid_shape
        2D array of radial coordinates of inner edges of all grid cells.
    rout : numpy.ndarray, shape: (ngrid,)
        Radial coordinates of outer edges of grid cells.
    routgrid : numpy.ndarray, shape: grid_shape
        2D array of radial coordinates of outer edges of all grid cells.
    dx : float
        Spacing between x=ln(r) coordinate of grid cells.
    x : numpy.ndarray, shape: (ngrid,)
        x=ln(r) coordinates of grid cell centres.
    xin : numpy.ndarray, shape: (ngrid,)
        x=ln(r) coordinates of inner edges of grid cells.
    xout : numpy.ndarray, shape: (ngrid,)
        x=ln(r) coordinates of outer edges of grid cells.
    fR : numpy.ndarray, shape: grid_shape
        After iter_solve has been run, scalar field solution fR is given here.
    u : numpy.ndarray, shape: grid_shape
        u = ln(fR/fRa)
    u_guess : numpy.ndarray, shape: grid_shape
        Optional initial guess for u. To be externally set by user before
        running iter_solve. After setting this, user should set GuessFlag to
        True.
    GuessFlag
        Whether a u_guess has been provided.
    time_taken : float
        Time taken by iter_solve to arrive at scalar field solution
    iters_taken : int
        Number of NGS iterations taken to arrive at scalar field solution.
    dvol : numpy.ndarray, shape: grid_shape
        Volumes of grid cells. UNITS: m^3
    fullvol : float
        Volume of whole grid. UNITS: m^3
    h : float
        Dimensionless Hubble constant, e.g. 0.7. Set in set_cosmology method.
    H0 : float
        Dimensional Hubble constant. Set in set_cosmology method. UNITS: s^-1
    omega_m : float
        Matter fraction, e.g. 0.3. Set in set_cosmology method.
    omega_l : float
        Dark energy fraction, e.g. 0.7. Set in set_cosmology method.
    redshift :
        Current redshift at time of solution. Set in set_cosmology method.
    rhocrit : float
        Cosmic critical density at redshift 0. Set in set_cosmology method.
    rhomean : float
        Mean matter density at solution redshift. Set in set_cosmology method.
    Ra : float
        Cosmic background curvature at solution redshift. Set in iter_solve.
    R0 : float
        Cosmic background curvature at redshift 0. Set in iter_solve.
    Fa : float
        Background scalar field at solution redshift. Set in iter_solve.
    F0 : float
        fR0. Given as argument to iter_solve.

    Methods
    -------
    __init__ :
        Initialise an instance of Grid1D
    set_cosmology :
        Set the cosmological parameters.
    laplacian :
        Internal method to calculate laplacian during scalar field solution.
    newton :
        Internal method to perform NGS iteration during scalar field solution.
    iter_solve :
        Solve for the scalar field using NGS method.
    """

    def __init__(self, ngrid=175, rmin=0.05*kpc, rmax=5*Mpc):
        """
        Initialise an instance of Grid1D class, see Grid1D docstring for more
        info.
        """

        self.ngrid = ngrid
        self.rmax = rmax
        self.rmin = rmin

        # radial direction: log spaced grid from xmin=ln(rmin) to xmax=ln(rmax)
        # xin and xout are the inner and outer edges of each grid cell
        xmax = np.log(rmax)
        xmin = np.log(rmin)
        self.dx = (xmax-xmin)/ngrid
        self.x = xmin + (np.arange(self.ngrid, dtype=np.float64)+0.5)*self.dx
        self.xout = xmin + (np.arange(self.ngrid, dtype=np.float64)+1)*self.dx
        self.xin = xmin + np.arange(self.ngrid, dtype=np.float64)*self.dx
        self.r = np.exp(self.x)
        self.rout = np.exp(self.xout)
        self.rin = np.exp(self.xin)

        # theta direction: evenly spaced from 0 to pi; number of points is nth
        self.nth = 101
        self.disc_ind = self.nth//2
        self.dth = np.pi/self.nth
        self.th = (np.arange(self.nth, dtype=np.float64)+0.5)*self.dth
        self.thin = np.arange(self.nth, dtype=np.float64)*self.dth
        self.thout = (np.arange(self.nth, dtype=np.float64)+1)*self.dth
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
        self.__coeff1 = self.routgrid/(self.dx**2*self.rgrid**3)
        self.__coeff2 = self.ringrid/(self.dx**2*self.rgrid**3)
        self.__coeff3 = self.sthoutgrid/((self.dth*self.rgrid)**2*self.sthgrid)
        self.__coeff4 = self.sthingrid/((self.dth*self.rgrid)**2*self.sthgrid)

        # dLdu_const gives the final constant term in the Newton Raphson
        # expression for dL/du
        d1 = (self.rgrid**3*self.dx**2)
        d2 = (self.rgrid**2*self.sthgrid*self.dth**2)
        self.__dLdu_const = ((self.ringrid+self.routgrid)/d1 +
                             (self.sthingrid+self.sthoutgrid)/d2)

        # dvol is cell volume, full vol is volume of whole sphere
        self.dvol = 2*np.pi*self.rgrid**3*np.sin(self.thgrid)*self.dx*self.dth
        self.fullvol = 4.0/3.0*np.pi*(self.rmax**3 - self.rmin**3)

        # user change to True if setting a guess
        self.GuessFlag = False

        return

    def set_cosmology(self, h, omega_m, redshift=0):
        """
        Assign a background cosmology to the grid: Hubble constant, redshift,
        omega_m are input, then rho_crit and rho_mean are calculated.

        Parameters
        ----------
        h : float
            Dimensionless Hubble parameter, e.g. 0.7
        omega_m : float
            Cosmic matter fraction, e.g. 0.3
        redshift : float
            Redshift at time of solution. Default: 0.
        """

        self.h = h
        self.omega_m = omega_m
        self.omega_l = 1 - omega_m
        self.redshift = redshift

        # calculate dimensional Hubble constant
        self.H0 = self.h * 100.0 * 1000.0 / Mpc

        # rhocrit is evaluated today, rhomean at given redshift
        self.rhocrit = 3.0 * self.H0**2 / (8.0 * np.pi * G)
        self.rhomean = (1+self.redshift)**3*self.omega_m * self.rhocrit

        return

    def laplacian(self, expu):
        """
        Calculate discretised laplacian of e^u. The coefficients coeff1 etc.
        are initialised in __init__ above. The boundary conditions are
        implemented here: first derivative of u vanishes at inner and outer
        boundaries.

        Parameters
        ----------
        expu : numpy.ndarray, shape grid_shape
            Exponential of u=ln(fR/fa)

        Returns
        -------
        D2expu : numpy.ndarray, shape grid_shape
            Discretised Laplacian of e^u
        """

        # d(e^u)/dx. BCs: vanishes at both boundaries
        deudx = np.zeros((self.ngrid+1, self.nth))
        deudx[1:-1, :] = (expu[1:, :] - expu[:-1, :])

        # d(e^u)/dtheta. BCs: vanishes at both boundaries
        deudth = np.zeros((self.ngrid, self.nth+1))
        deudth[:, 1:-1] = (expu[:, 1:] - expu[:, :-1])

        # discretised laplacian
        D2expu = deudx[1:, :]*self.__coeff1 - deudx[:-1, :]*self.__coeff2
        D2expu += deudth[:, 1:]*self.__coeff3 - deudth[:, :-1]*self.__coeff4
        return D2expu

    def newton(self, expu, D2expu):
        """
        Calculate the change in u this iteration, i.e. du = - L / (dL/du)

        Parameters
        ----------
        expu : numpy.ndarray, shape grid_shape
            Exponential of u=ln(fR/fa)
        D2expu : numpy.ndarray, shape grid_shape
            Discretised Laplacian of e^u, as calculated in 'laplacian' method

        Returns
        -------
        du : numpy.ndarray, shape grid_shape
            Required change to u this iteration.
        """

        # Newton-Raphson step, as in MG-GADGET paper
        oneoverrteu = 1/np.sqrt(expu)
        L = D2expu + self.__const1*(1.0-oneoverrteu) - self.__const2*self.drho
        dLdu = 0.5*self.__const1*oneoverrteu - expu*self.__dLdu_const
        du = - L / dLdu

        return du

    def iter_solve(self, niter, F0, verbose=False, tol=1e-7):
        """
        Iteratively solve the scalar field equation of motion on grid, using
        a Newton-Gauss-Seidel relaxation technique, as in MG-GADGET. Perform
        iterations until the change in u=ln(fR/fa) is everywhere below the
        threshold specified by the parameter 'tol'.

        Parameters
        ----------
        niter : int
            Max number of NGS iterations, beyond which the computation stops
            with an error code.
        F0 : float
            Cosmic background value of the scalar field. Should be negative,
            e.g. -1e-6 for F6.
        verbose : bool
            Whether to print progress update periodically. Default: False.
        tol : float
            Tolerance level for iterative changes in u. Once du is everywhere
            below this threshold, the computation stops. Default: 1e-7.
        """

        # relevant constants
        msq = self.omega_m*self.H0*self.H0
        self.Ra = 3*msq*((1+self.redshift)**3 + 4*self.omega_l/self.omega_m)
        self.R0 = 3*msq*(1 + 4*self.omega_l/self.omega_m)
        self.F0 = F0
        self.Fa = self.F0*(self.R0/self.Ra)**2
        self.__const1 = self.Ra / (3.0 * c2 * self.Fa)
        self.__const2 = -8.0*np.pi*G / (3.0 * c2 * self.Fa)

        # u = ln(fR/Fa)
        if self.GuessFlag:
            u = self.u_guess
        else:
            u = np.zeros((self.ngrid, self.nth))

        # main loop
        t0 = time.time()
        for i in np.arange(niter):

            # calculate change du
            expu = np.exp(u)
            D2expu = self.laplacian(expu)
            du = self.newton(expu, D2expu)

            # check if du is too high or sufficiently low
            if((abs(du) > 1).any()):
                ind = np.where(du > 1.0)
                du[ind] = 1.0
                ind = np.where(du < -1.0)
                du[ind] = -1.0
            elif abs(du).max() < tol:
                break

            # NGS update
            u += du

            if(verbose and i % 1000 == 0):
                print("iteration", i, ", max(|du|)=", abs(du).max())
        t1 = time.time()
        if i == niter-1:
            raise Exception("Solver took too long!")

        # output results
        self.u = u
        self.fR = self.Fa*np.exp(u)
        self.time_taken = t1-t0
        self.iters_taken = i

        if verbose:
            print("Took ", self.time_taken, " seconds")

        return
