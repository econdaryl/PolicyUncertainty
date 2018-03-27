# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 14:38:20 2017

@author: Daryl Larsen

This program reads in parameter values and steady states from the file, 
ILAfindss.pkl.

It then calculates the linear coefficients for the policy and jump function
approximations using the GSSA toolkit.

The coefficients and time to solve are written to the file, ILAsolveGSSA.pkl.

The baseline values have a 1 at the end of the variable name.
The values after the policy change have a 2 at the end. 
"""

import numpy as np
import timeit
import pickle as pkl

# import the modules from GSSA
from gssa import GSSA

pord = 4
# -----------------------------------------------------------------------------
# READ IN VALUES FROM STEADY STATE CALCULATIONS

# load steady state values and parameters
infile = open('ILAfindss.pkl', 'rb')
(bar1, bar2, params1, params2, GSSAparams) = pkl.load(infile)
infile.close()

infile = open('ILAsolveGSSA_3.pkl', 'rb')
(coeffs3a, coeffs3b, timesolve) = pkl.load(infile)
infile.close()

A = np.array([[0., 0.], \
              [0., 0.]])
coeffs3a = np.insert(coeffs3a, 2*pord-1, A)
coeffs3b = np.insert(coeffs3b, 2*pord-1, A)

try:
    infile = open('ILAsolveGSSA_4.pkl', 'rb')
    (coeffs4a, coeffs4b, timesolve) = pkl.load(infile)
    infile.close()
    old_pord = False
except FileNotFoundError:
    old_pord = True
    pass

# unpack
[kbar1, ellbar1, Ybar1, wbar1, rbar1, Tbar1, cbar1, ibar1, ubar1] = bar1
[kbar2, ellbar2, Ybar2, wbar2, rbar2, Tbar2, cbar2, ibar2, ubar2] = bar2
[alpha, beta, gamma, delta, chi, theta, tau, rho_z, sigma_z] = params1
tau2 = params2[6]
(zbar, Zbar, NN, nx, ny, nz, logX, Sylv) = GSSAparams

# set clock for time to calcuate functions
startsolve = timeit.default_timer()

# set name for external files written
name = 'ILAsolveGSSA_4'

# -----------------------------------------------------------------------------
# BASELINE
T = 10000
old = True
GSSAparams = (T, nx, ny, nz, pord, old)
# find the policy and jump function coefficients
if old_pord == True:
    coeffs1 = GSSA(params1, kbar1, ellbar1, GSSAparams, coeffs3a)
elif old_pord == False:
    coeffs1 = GSSA(params1, kbar1, ellbar1, GSSAparams, coeffs4a)
# -----------------------------------------------------------------------------
# CHANGE POLICY

# find the policy and jump function coefficients
if old_pord == True:
    coeffs2 = GSSA(params2, kbar2, ellbar2, GSSAparams, coeffs3b)
elif old_pord == False:
    coeffs2 = GSSA(params2, kbar2, ellbar2, GSSAparams, coeffs4b)
print ('baseline coeffs')
print (coeffs1)
print  (' ')
print ('policy change coeffs')
print(coeffs2)
print  (' ')

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