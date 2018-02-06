#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Run a series of monte carlo simulations and analyzes them using the
GSSA method
'''

import numpy as np
import pickle as pkl
import timeit

from ILArunmc import runmc
from ILAmcanalysis import mcanalysis
from ILAsimGSSA import generateGSSA

name = 'ILAonesimGSSA'

# -----------------------------------------------------------------------------
# LOAD VALUES FROM SS and Zhist pickle file
    
# load steady state values and parameters
infile = open('ILAfindss.pkl', 'rb')
(bar1, bar2, params1, params2, LINparams) = pkl.load(infile)
infile.close()

infile = open('ILAsolveGSSA.pkl', 'rb')
(coeffs1, coeffs2, timesolve) = pkl.load(infile)
infile.close()


# unpack
[kbar1, Ybar1, wbar1, rbar1, Tbar1, cbar1, ibar1, ubar1] = bar1
[kbar2, Ybar2, wbar2, rbar2, Tbar2, cbar2, ibar2, ubar2] = bar2
[alpha, beta, tau, rho_z, sigma_z] = params1
tau2 = params2[2]
(zbar, Zbar, NN, nx, ny, nz, logX, Sylv) = LINparams
    
# create args lists
pord = 3
args1 = (pord, nx, ny, nz, coeffs1)
args2 = (pord, nx, ny, nz, coeffs2)

# load zhist
infile = open('ILAzhist.pkl', 'rb')
(nobs, zhist) = pkl.load(infile)
infile.close()

# RUN SINGLE SIMULATION
from ILAonepolsim import polsim
initial = (kbar1, zhist[0])
ts = 20
simargs = (initial, zhist, nobs, ts, generateGSSA, args1, args2, params1, \
           params2) 
khist, Yhist, whist, rhist, Thist, chist, ihist, uhist = polsim(simargs)

# SAVE AND PLOT SIMULATIOn

# write histories
output = open(name + '.pkl', 'wb')
alldata = (khist, Yhist, whist, rhist, Thist, chist, ihist, uhist)
pkl.dump(alldata, output)
output.close()

from ILAonesimplots import ILAonesimplots
ILAonesimplots(alldata, name)
