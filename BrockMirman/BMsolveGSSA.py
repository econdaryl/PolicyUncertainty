#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This program reads in paramter values and steady states from the file, 
ILAfindss.pkl.

It then calculates the linear coefficients for the policy and jump function
approximations using the LinApp toolkit.

The coefficients and time to solve are written to the file, ILAsolveLIN.pkl.

The baseline values have a 1 at the end of the variable name.
The values after the policy change have a 2 at the end. 
"""

import timeit
import pickle as pkl

# import the modules from LinApp
from gssa import GSSA

# -----------------------------------------------------------------------------
# READ IN VALUES FROM STEADY STATE CALCULATIONS

# load steady state values and parameters
infile = open('BMfindss.pkl', 'rb')
(bar1, bar2, params1, params2, LINparams) = pkl.load(infile)
infile.close()

# unpack
[kbar1, Ybar1, wbar1, rbar1, Tbar1, cbar1, ibar1, ubar1] = bar1
[kbar2, Ybar2, wbar2, rbar2, Tbar2, cbar2, ibar2, ubar2] = bar2
[alpha, beta, tau, rho_z, sigma_z] = params1
tau2 = params2[2]
(zbar, Zbar, NN, nx, ny, nz, logX, Sylv) = LINparams

# set clock for time to calcuate functions
startsolve = timeit.default_timer()

# set name for external files written
name = 'BMsolveGSSA'

# -----------------------------------------------------------------------------
# BASELINE

# set up steady state input vector for baseline
coeffs1 = GSSA(params1, kbar1)

# -----------------------------------------------------------------------------
# CHANGE POLICY

# set up coefficient list
coeffs2 = GSSA(params2, kbar2)

# calculate time to solve for functions
stopsolve = timeit.default_timer()
timesolve =  stopsolve - startsolve
print('time to solve: ', timesolve)


# -----------------------------------------------------------------------------
# SAVE RESULTS

output = open(name + '.pkl', 'wb')

# write timing
pkl.dump((coeffs1, coeffs2, timesolve), output)

output.close()