#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This program sets up the parameter values and solves for the steady states for
the simple infintely-lived agent model.

The parameters and steady state values are stored in the file, ILSfindss,pkl

The baseline values have a 1 at the end of the variable name.
The values after the policy change have a 2 at the end.
"""

import numpy as np
import timeit
import pickle as pkl

# import the modules from LinApp
from LinApp_FindSS import LinApp_FindSS
# from LinApp_SSL import LinApp_SSL

from OLG3funcs import Modeldefs, Modeldyn

# set name for external files written
name = 'OLG3findss'

# set clock for time to calcuate functions
startsolve = timeit.default_timer()

# -----------------------------------------------------------------------------
# BASELINE

# set parameter values
alpha = .35
beta = .99**60
gamma = 2.5
delta = 1 - (1-.02)**60
chi = 20.
theta = .33
rho_z = .9**60
sigma_z = .005
# set old and new tax rates
tau = .05
tau2 = .055
pi2 = .95
pi3 = .8
f1 = 1.
f2 = 2.

# set LinApp parameters
zbar = 0.
Zbar = np.array([zbar])
NN = np.array([rho_z])
nx = 2
ny = 2
nz = 1
logX = 0
Sylv = 0

# make parameter list to pass to functions for baseline
params1 = np.array([alpha, beta, gamma, delta, chi, theta, tau, rho_z, \
    sigma_z, pi2, pi3, f1, f2, nx, ny, nz])

# save in list
LINparams = (zbar, Zbar, NN, nx, ny, nz, logX, Sylv)

# take a guess for steady state values of k and l
guessXY = np.array([.01, .02, .1, .1])

# find the steady state values using LinApp_FindSS
XYbar1 = LinApp_FindSS(Modeldyn, params1, guessXY, Zbar, nx, ny)
(k2bar1, k3bar1, l1bar1, l2bar1) = XYbar1
kbar1 = np.array([k2bar1, k3bar1])
lbar1 = np.array([l1bar1, l2bar1])

# set up SS theta vector
theta1 = np.array([k2bar1, k3bar1, k2bar1, k3bar1, k2bar1, k3bar1, \
                   l1bar1, l2bar1, l1bar1, l2bar1, 0., 0.])

# check SS solution
check = Modeldyn(theta1, params1)
print ('check SS: ', check)
if np.max(np.abs(check)) > 1.E-6:
    print ('Have NOT found steady state')
    
# find the steady state values for the definitions
Kbar1, Lbar1, GDPbar1, wbar1, rbar1, T4bar1, Bbar1, c1bar1, c2bar1, \
    c3bar1, Cbar1, Ibar1, u1bar1, u2bar1, u3bar1 = Modeldefs(kbar1, \
    kbar1, lbar1, 0., params1)
    
bar1 = np.array([k2bar1, k3bar1, l1bar1, l2bar1, Kbar1, \
    Lbar1, GDPbar1, wbar1, rbar1, T4bar1, Bbar1, c1bar1, c2bar1, c3bar1, \
    Cbar1, Ibar1, u1bar1, u2bar1, u3bar1])

# display all steady state values for baseline
print ('baselines steady state values')
print ('kbar:   ', kbar1)
print ('lbar:   ', lbar1)
print ('Ybar:   ', GDPbar1)
print ('wbar:   ', wbar1)
print ('rbar:   ', rbar1)
print ('Cbar:   ', Cbar1)
print ('Ibar:   ', Ibar1)
print ('Lbar:   ', Lbar1)
print ('T4bar:  ', T4bar1)
print (' ')

# -----------------------------------------------------------------------------
# CHANGE POLICY

# make parameter list to pass to functions for new tax
params2 = np.array([alpha, beta, gamma, delta, chi, theta, tau2, rho_z, \
    sigma_z, pi2, pi3, f1, f2, nx, ny, nz])

# use the old steady state values of k and ell for our guess
guessXY = np.array([.01, .02, .1, .1])

# find the steady state values using LinApp_FindSS
XYbar2 = LinApp_FindSS(Modeldyn, params2, guessXY, Zbar, nx, ny)
(k2bar2, k3bar2, l1bar2, l2bar2) = XYbar2
kbar2 = np.array([k2bar2, k3bar2])
lbar2 = np.array([l1bar2, l2bar2])

# set up SS theta vector
theta1 = np.array([k2bar2, k3bar2, k2bar2, k3bar2, k2bar2, k3bar2, \
                   l1bar2, l2bar2, l1bar2, l2bar2, 0., 0.])

# check SS solution
check = Modeldyn(theta1, params2)
print ('check SS: ', check)
if np.max(np.abs(check)) > 1.E-6:
    print ('Have NOT found steady state')
    
# find the steady state values for the definitions
Kbar2, Lbar2, GDPbar2, wbar2, rbar2, T4bar2, Bbar2, c1bar2, c2bar2, c3bar2, \
    Cbar2, Ibar2, u1bar2, u2bar2, u3bar2, = Modeldefs(kbar2, \
    kbar2, lbar2, 0., params2)
    
bar2 = np.array([k2bar2, k3bar2, l1bar2, l2bar2, Kbar2, \
    Lbar2, GDPbar2, wbar2, rbar2, T4bar2, Bbar2, c1bar2, c2bar2, c3bar2, \
    Cbar2, Ibar2, u1bar2, u2bar2, u3bar2])

# display all steady state values for baseline
print ('baselines steady state values')
print ('kbar:   ', kbar2)
print ('lbar:   ', lbar2)
print ('Ybar:   ', GDPbar2)
print ('wbar:   ', wbar2)
print ('rbar:   ', rbar2)
print ('Cbar:   ', Cbar2)
print ('Ibar:   ', Ibar2)
print ('Lbar:   ', Lbar2)
print ('T4bar:  ', T4bar2)
print (' ')

# -----------------------------------------------------------------------------
# SAVE RESULTS

output = open(name + '.pkl', 'wb')

# write timing
pkl.dump((bar1, bar2, params1, params2, LINparams), output)

output.close()