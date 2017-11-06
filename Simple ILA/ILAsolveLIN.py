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

import numpy as np
import timeit
import pickle as pkl

# import the modules from LinApp
from LinApp_Deriv import LinApp_Deriv
from LinApp_Solve import LinApp_Solve
# from LinApp_SSL import LinApp_SSL

from Simple_ILA_Model_Funcs import Modeldyn

# -----------------------------------------------------------------------------
# READ IN VALUES FROM STEADY STATE CALCULATIONS

# load steady state values and parameters
infile = open('ILAfindss.pkl', 'rb')
(bar1, bar2, params1, params2, LINparams) = pkl.load(infile)
infile.close()

# unpack
[kbar1, ellbar1, Ybar1, wbar1, rbar1, Tbar1, cbar1, ibar1, ubar1] = bar1
[kbar2, ellbar2, Ybar2, wbar2, rbar2, Tbar2, cbar2, ibar2, ubar2] = bar2
[alpha, beta, gamma, delta, chi, theta, tau, rho_z, sigma_z] = params1
tau2 = params2[6]
(zbar, Zbar, NN, nx, ny, nz, logX, Sylv) = LINparams

# set clock for time to calcuate functions
startsolve = timeit.default_timer()

# set name for external files written
name = 'ILAsolveLIN'

# -----------------------------------------------------------------------------
# BASELINE

# set up steady state input vector for baseline
theta1 = np.array([kbar1, kbar1, kbar1, ellbar1, ellbar1, 0., 0.])

# find the derivatives matrices
[AA1, BB1, CC1, DD1, FF1, GG1, HH1, JJ1, KK1, LL1, MM1, WW1, TT1] = \
    LinApp_Deriv(Modeldyn, params1, theta1, nx, ny, nz, logX)

# find the policy and jump function coefficients
PP1, QQ1, UU1, RR1, SS1, VV1= \
    LinApp_Solve(AA1,BB1,CC1,DD1,FF1,GG1,HH1,JJ1,KK1,LL1,MM1,WW1,TT1,NN,Zbar, \
                 Sylv)
print ('baseline coeffs')
print ('P: ', PP1)
print ('Q: ', QQ1)
print ('R: ', RR1)
print ('S: ', SS1)
print  (' ')

# set up coefficient list
coeffs1 = (PP1, QQ1, UU1, RR1, SS1, VV1)


# -----------------------------------------------------------------------------
# CHANGE POLICY

# set up steady state input vector for baseline
theta2 = np.array([kbar2, kbar2, kbar2, ellbar2, ellbar2, 0., 0.])

# find the new derivatives matrices
[AA2, BB2, CC2, DD2, FF2, GG2, HH2, JJ2, KK2, LL2, MM2, WW2, TT2] = \
    LinApp_Deriv(Modeldyn, params2, theta2, nx, ny, nz, logX)
    
# find the policy and jump function coefficients
PP2, QQ2, UU2, RR2, SS2, VV2 = \
    LinApp_Solve(AA2,BB2,CC2,DD2,FF2,GG2,HH2,JJ2,KK2,LL2,MM2,WW2,TT2,NN,Zbar, \
                 Sylv)
print ('policy change coeffs')
print ('P: ', PP2)
print ('Q: ', QQ2)
print ('R: ', RR2)
print ('S: ', SS2)
print  (' ')

# calculate time to solve for functions
stopsolve = timeit.default_timer()
timesolve =  stopsolve - startsolve
print('time to solve: ', timesolve)

# set up coefficient list
coeffs2 = (PP2, QQ2, UU2, RR2, SS2, VV2)


# -----------------------------------------------------------------------------
# SAVE RESULTS

output = open(name + '.pkl', 'wb')

# write timing
pkl.dump((coeffs1, coeffs2, timesolve), output)

output.close()