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

from BMfuncs import Modeldefs, Modeldyn

# set name for external files written
name = 'BMfindss'

# set clock for time to calcuate functions
startsolve = timeit.default_timer()

# -----------------------------------------------------------------------------
# BASELINE

# set parameter values
alpha = .35
beta = .99
rho_z = .9
sigma_z = .005
# set old and new tax rates
tau = .05
tau2 = .055

# make parameter list to pass to functions for baseline
params1 = np.array([alpha, beta, tau, rho_z, sigma_z])

# set LinApp parameters
zbar = 0.
Zbar = np.array([zbar])
NN = np.array([rho_z])
nx = 1
ny = 0
nz = 1
logX = 0
Sylv = 0
# save in list
LINparams = (zbar, Zbar, NN, nx, ny, nz, logX, Sylv)

# find the steady state values using closed form solution
XYbar1 = (alpha*beta*(1-tau))**(1/(1-alpha))
kbar1 = XYbar1

# set up SS theta vector
theta1 = np.array([kbar1, kbar1, kbar1, 0., 0.])

# check SS solution
check = Modeldyn(theta1, params1)
print ('check SS: ', check)
if np.max(np.abs(check)) > 1.E-6:
    print ('Have NOT found steady state')
    
# find the steady state values for the definitions
Ybar1, wbar1, rbar1, Tbar1, cbar1, ibar1, ubar1 = \
    Modeldefs(kbar1, kbar1, 0., params1)
    
bar1 = np.array([kbar1, Ybar1, wbar1, rbar1, Tbar1, cbar1, ibar1, \
                 ubar1])

# display all steady state values for baseline
print ('baselines steady state values')
print ('kbar:   ', kbar1)
print ('Ybar:   ', Ybar1)
print ('wbar:   ', wbar1)
print ('rbar:   ', rbar1)
print ('Tbar:   ', Tbar1)
print ('cbar:   ', cbar1)
print ('ibar:   ', ibar1)
print ('ubar:   ', ubar1)
print (' ')

# -----------------------------------------------------------------------------
# CHANGE POLICY

# make parameter list to pass to functions for new tax
params2 = np.array([alpha, beta, tau2, rho_z, sigma_z])

# find the steady state values using closed form solution
XYbar2 = (alpha*beta*(1-tau2))**(1/(1-alpha))
kbar2 = XYbar2

# set up SS theta vector
theta2 = np.array([kbar2, kbar2, kbar2, 0., 0.])

# check SS solution
check = Modeldyn(theta2, params2)
print ('check SS: ', check)
if np.max(np.abs(check)) > 1.E-6:
    print ('Have NOT found steady state')
    
# find the steady state values for the definitions
Ybar2, wbar2, rbar2, Tbar2, cbar2, ibar2, ubar2 = \
    Modeldefs(kbar2, kbar2, 0., params2)
    
bar2 = np.array([kbar2, Ybar2, wbar2, rbar2, Tbar2, cbar2, ibar2, \
                  ubar2])

# display all steady state values
print ('steady state values after policy change')
print ('kbar:   ', kbar2)
print ('Ybar:   ', Ybar2)
print ('wbar:   ', wbar2)
print ('rbar:   ', rbar2)
print ('Tbar:   ', Tbar2)
print ('cbar:   ', cbar2)
print ('ibar:   ', ibar2)
print ('ubar:   ', ubar2)

# -----------------------------------------------------------------------------
# SAVE RESULTS

output = open(name + '.pkl', 'wb')

# write timing
pkl.dump((bar1, bar2, params1, params2, LINparams), output)

output.close()