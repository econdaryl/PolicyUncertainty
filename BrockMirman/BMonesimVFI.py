#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Run a series of monte carlo simulations and analyzes them using the
VFI method
'''

import numpy as np
import pickle as pkl
import timeit

from BMrunmc import runmc
from BMmcanalysis import mcanalysis

name = 'BMonesimVFI'

def generateVFI(k, z, args):
    
    '''
    This function generates values of k next period and ell this period given
    values for k and z this period.
    
    Inputs
    k - k this period
    z - z this period
    args - lists of linear coeffiecients and the steady state values.
    
    Outputs
    kp - k next period
    ell - ell this period
    '''
    
     # unpack args
    (coeffs, XYbar) = args
    (Vf1, Pf1, coeffsPF) = coeffs
    
    # inputs must be 1D numpy arrays and deviation from SS values
    Xvec = np.array([[1.0], [k], [k**2], [k**3], [z], [z**2], [z**3], \
                     [k*z], [k**2*z], [k*z**2]])

    kp = np.vdot(Xvec, coeffsPF)
    
    return kp

# -----------------------------------------------------------------------------
# LOAD VALUES FROM SS and Zhist pickle file
    
# load steady state values and parameters
infile = open('BMfindss.pkl', 'rb')
(bar1, bar2, params1, params2, VFIparams) = pkl.load(infile)
infile.close()

infile = open('BMsolveVFI.pkl', 'rb')
(coeffs1, coeffs2, timesolve) = pkl.load(infile)
infile.close()

# unpack
[kbar1, Ybar1, wbar1, rbar1, Tbar1, cbar1, ibar1, ubar1] = bar1
[kbar2, Ybar2, wbar2, rbar2, Tbar2, cbar2, ibar2, ubar2] = bar2
[alpha, beta, tau, rho_z, sigma_z] = params1
tau2 = params2[2]
(zbar, Zbar, NN, nx, ny, nz, logX, Sylv) = VFIparams
    
# create args lists
XYbar1 = kbar1
XYbar2 = kbar2
args1 = (coeffs1, XYbar1)
args2 = (coeffs2, XYbar2)

# load zhist
infile = open('BMzhist.pkl', 'rb')
(nobs, zhist) = pkl.load(infile)
infile.close()

# RUN SINGLE SIMULATION
from BMonepolsim import polsim
initial = (kbar1, zhist[0])
ts = 20
simargs = (initial, zhist, nobs, ts, generateVFI, args1, args2, params1, \
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
