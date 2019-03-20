#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created: 1st February 2019
Author: A. P. Naik
Description: Code to attempt global MCMC on full SPARC dataset, using analytic
fifth force prescription, to see whether convergence is feasible.
"""
import emcee
import time
from sparc_data import SPARCGlobal
import numpy as np
from scipy.constants import c as clight
from scipy.constants import parsec as pc
from scipy.constants import G
kpc = 1e+3*pc
delta = 93.6
h = 0.7
H0 = h*100*1000/(1e+6*pc)
omega_m = 0.308
rho_c = 3*H0**2/(8*np.pi*G)
Msun = 1.989e+30
sparc = SPARCGlobal()

# Moster parameters, with errors
M1 = 10**(11.59)*Msun
sig_M1 = M1*np.log(10)*0.236
N = 0.0351
sig_N = 0.0058
beta = 1.376
sig_beta = 0.153
gamma = 0.608
sig_gamma = 0.059


# Dutton parameters, and error on log10c
a = 1.025
b = -0.097
err_logc = 0.11


def SHM(M_halo):

    X = M_halo/M1
    denom = X**(-beta) + X**gamma
    M_star = 2*N*M_halo/denom
    return M_star


def err_SHM(M_halo):

    Ms = SHM(M_halo)
    err_N = sig_N*Ms/N
    err_M1 = sig_M1*Ms/M1
    err_beta = sig_beta*Ms/beta
    err_gamma = sig_gamma*Ms/gamma

    err_M_star = np.sqrt(err_N**2 + err_M1**2 + err_beta**2 + err_gamma**2)
    return err_M_star


def CMR(M_halo):

    logc = a + b*np.log10(M_halo/(1e+12*Msun/h))
    return logc


def lnprior(theta):

    lnP = 0
    for i in range(85):

        lb = np.array([4, 0, -0.52, -9])
        ub = np.array([5.7, 2, -0.1, np.log10(2e-6)])

        galaxy = sparc.galaxies[i]
        theta_gal = np.append(theta[i*3:3*i+3], theta[-1])

        if (theta_gal < ub).all() and (theta_gal > lb).all():

            ML = 10**theta_gal[2]
            V_vir = 10**theta_gal[0]
            c_vir = 10**theta_gal[1]

            M_halo = (V_vir**3)/(np.sqrt(delta/2)*G*H0)
            M_star = ML*1e+9*galaxy.luminosity_tot*Msun

            M_gas = (4/3)*galaxy.HI_mass
            if (M_star+M_gas)/M_halo > 0.2:
                lnP += -1e+20
                continue

            y = np.log10(M_star)
            mu = np.log10(SHM(M_halo))
            sig = err_SHM(M_halo)/(SHM(M_halo)*np.log(10))
            sig += 0.2  # f(R) broadening
            g1 = ((y-mu)/sig)**2

            y = np.log10(c_vir)
            mu = CMR(M_halo)
            sig = err_logc
            sig += 0.1  # f(R) broadening
            g2 = ((y-mu)/sig)**2

            lnP += -0.5*g1*g2
            continue

        else:
            lnP += -np.inf
            continue

    return lnP


def calc_R_scr(theta):

    V_vir = 10**theta[0]
    c_vir = 10**theta[1]
    fR0 = -10**theta[2]

    M_vir = (V_vir**3)/(np.sqrt(delta/2)*G*H0)
    R_vir = (3*M_vir/(4*np.pi*delta*rho_c))**(1/3)

    alpha, beta, gamma = 1, 3, 1
    R_s = R_vir/(c_vir*((2-gamma)/(beta-2))**(1/alpha))

    C1 = 1/(1 + (R_vir/R_s))
    C2 = (3*np.log(1+fR0)*clight**2)/(8*np.pi*G*delta*rho_c*R_s)
    denom = C1 - C2
    R_scr = R_s*(1/denom - 1)

    return R_scr


def halo_v_circ(R, theta):

    V_vir = 10**theta[0]
    c_vir = 10**theta[1]

    M_vir = (V_vir**3)/(np.sqrt(delta/2)*G*H0)
    R_vir = (3*M_vir/(4*np.pi*delta*rho_c))**(1/3)

    alpha, beta, gamma = 1, 3, 1
    R_s = R_vir/(c_vir*((2-gamma)/(beta-2))**(1/alpha))
    denom = 4*np.pi*R_s**3*(np.log(1+c_vir) - (c_vir/(1+c_vir)))

    rho_0 = M_vir/denom

    M_s = 4*np.pi*rho_0*R_s**3
    M_enc = M_s*(np.log((R_s+R)/R_s) - (R/(R_s+R)))

    v_DM = np.sqrt(G*M_enc/R)

    return v_DM


def v_model(theta, galaxy):
    """
    Calculate v_circ for parameter set contained in theta, on SPARC galaxy
    contained in 'galaxy' argument. Fifth force is optionally included by
    setting fR=True.
    """

    ML = 10**theta[2]

    R = galaxy.R*kpc  # metres
    v_g = 1e+3*galaxy.v_gas  # m/s
    v_d = 1e+3*galaxy.v_disc
    v_b = 1e+3*galaxy.v_bul

    # NFW profile
    v_DM = halo_v_circ(R, theta)

    # calculating Newtonian and MG accelerations
    a_N = (v_DM**2 + v_g**2 + ML*v_b**2 + ML*v_d**2)/R

    R_scr = calc_R_scr(theta)
    if R_scr == 0:
        a_scr = 0
    else:
        a_scr = np.interp(R_scr, np.append(0, galaxy.R), np.append(0, a_N))

    a_5 = np.zeros_like(a_N)
    if R_scr < galaxy.R[-1]:
        inds = np.where(galaxy.R > R_scr)[0]
        mass_frac = (a_scr*R_scr**2)/(a_N[inds]*galaxy.R[inds]**2)
        a_5[inds] = (a_N[inds]/3)*(1 - mass_frac)

    # circular velocity
    v_c = 1e-3*np.sqrt((a_N+a_5)*R)  # km/s
    return v_c


def lnlike(theta):

    # loop over 85 galaxies, calculate likelihood for each
    lnL = 0
    for i in range(85):

        galaxy = sparc.galaxies[i]
        theta_gal = np.append(theta[i*3:3*i+3], theta[-1])

        model = v_model(theta=theta_gal, galaxy=galaxy)
        lnL += -0.5*np.sum(((galaxy.v - model)/galaxy.v_err)**2)

    return lnL


def initialise_pos(ntemps, nwalkers, ndim):

    lb_unit = np.array([4, 0, -0.52])
    ub_unit = np.array([5.7, 2, -0.1])

    lb = np.zeros(256)
    ub = np.zeros(256)
    for i in range(85):
        lb[3*i:3*i+3] = lb_unit
        ub[3*i:3*i+3] = ub_unit
    lb[-1] = -9
    ub[-1] = np.log10(2e-6)

    # initial positions, random within prior bounds
    d = ub-lb
    p0 = lb + d*np.random.rand(ntemps, nwalkers, ndim)

    return p0

print("loading state...")
data = np.load("sparc_global_test.npz")
chain = data['chain']
lnprob = data['lnprob']
pos = chain[:, :, -1, :]

ntemps = 4
nwalkers = 512
ndim = 256
sampler = emcee.PTSampler(ntemps, nwalkers, ndim,
                          lnlike, lnprior, threads=4)

print("running sampler...")
t0 = time.time()
nsteps = 1485
for i, result in enumerate(sampler.sample(pos, iterations=nsteps)):
    if (i+1) % 10 == 0:
        print("{0:5.1%}".format(float(i) / nsteps))

t1 = time.time()
print(t1-t0)

chain = np.dstack((chain, sampler.chain))
lnprob = np.dstack((lnprob, sampler.lnprobability))

print("saving state...")
np.savez("sparc_global_test", chain=chain, lnprob=lnprob)

niter = lnprob[0, 0].shape[0]
for i in range(512):
    plt.plot(np.arange(niter), lnprob[0, i])

n = 500
    
W = np.sum(np.std(lnprob[0, :, -500:], axis=-1)**2)/512
a = np.mean(np.mean(lnprob[0, :, -500:], axis=-1))
B = (n/511) * np.sum((np.mean(lnprob[0, :, -500:], axis=-1) - a)**2)
var_theta = (1 - 1/n) * W + 1/n*B

GR = np.sqrt(var_theta/W)
print(GR)
