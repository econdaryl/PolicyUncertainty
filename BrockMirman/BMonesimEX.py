#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Run a series of monte carlo simulations and analyzes them using the
linearization method
'''

import numpy as np
import pickle as pkl
import timeit

from BMrunmc import runmc
from BMmcanalysis import mcanalysis

name = 'BMonesimEX'

def generateEX(k, z, args):
    
    '''
    This function generates values of k next period and ell this period given
    values for k and z this period.
    
    Inputs
    k - k this period
    z - z this period
    args - lists of linear coeffiecients and the steady state values.
    
    Outputs
    kp - k next period
    '''
    
    # unpack args
    (params, XYbar) = args
    [alpha, beta, tau, rho_z, sigma_z] = params

    kp = alpha*beta*(1-tau)*np.exp(z)*k**alpha
    
    return kp


# -----------------------------------------------------------------------------
# LOAD VALUES FROM SS and Zhist pickle file
    
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
    
# create args lists
XYbar1 = kbar1
XYbar2 = kbar2
args1 = (params1, XYbar1)
args2 = (params2, XYbar2)

# load zhist
infile = open('BMzhist.pkl', 'rb')
(nobs, zhist) = pkl.load(infile)
infile.close()

# RUN SINGLE SIMULATION
from BMonepolsim import polsim
initial = (kbar1, zhist[0])
ts = 20
simargs = (initial, zhist, nobs, ts, generateEX, args1, args2, params1, \
           params2) 
khist, Yhist, whist, rhist, Thist, chist, ihist, uhist = polsim(simargs)

# SAVE AND PLOT SIMULATIOn

# write histories
output = open(name + '.pkl', 'wb')
alldata = (khist, Yhist, whist, rhist, Thist, chist, ihist, uhist, zhist)
pkl.dump(alldata, output)
output.close()

from BMonesimplots import BMonesimplots
BMonesimplots(alldata, name)
