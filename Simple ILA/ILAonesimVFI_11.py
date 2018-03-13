#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Run a series of monte carlo simulations and analyzes them using the
VFI method (for only one time) 
'''

import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
import timeit

from ILArunmc import runmc
from ILAmcanalysis import mcanalysis

name = 'ILAonesimVFI_11_AL'

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
    (Vf, Pf, Jf, coeffsPF, coeffsJF) = coeffs
    
    # inputs must be 1D numpy arrays and deviation from SS values
    Xvec = np.array([[1.0], [k], [k**2], [k**3], [z], [z**2], [z**3], \
                     [k*z], [k**2*z], [k*z**2]])

    kp = np.vdot(Xvec, coeffsPF)
    ell= np.vdot(Xvec, coeffsJF)
    
    return kp, ell

# -----------------------------------------------------------------------------
# LOAD VALUES FROM SS and Zhist pickle file
    
# load steady state values and parameters
infile = open('ILAfindss.pkl', 'rb')
(bar1, bar2, params1, params2, VFIparams) = pkl.load(infile)
infile.close()

# load VFI coeffs
infile = open('ILAsolveVFI_11.pkl', 'rb')
(coeffs1, coeffs2, timesolve) = pkl.load(infile)
infile.close()

# unpack
[kbar1, ellbar1, Ybar1, wbar1, rbar1, Tbar1, cbar1, ibar1, ubar1] = bar1
[kbar2, ellbar2, Ybar2, wbar2, rbar2, Tbar2, cbar2, ibar2, ubar2] = bar2
[alpha, beta, gamma, delta, chi, theta, tau, rho_z, sigma_z] = params1
tau2 = params2[6]
(zbar, Zbar, NN, nx, ny, nz, logX, Sylv) = VFIparams
    
# create args lists
XYbar1 = (kbar1, ellbar1)
XYbar2 = (kbar2, ellbar2)

args1 = (coeffs1, XYbar1)
args2 = (coeffs2, XYbar2)

# load zhist
infile = open('ILAzhist.pkl', 'rb')
(nobs, zhist) = pkl.load(infile)
infile.close()

#------------------------------------------------------------------------------
# RUN SINGLE SIMULATION
from ILAonepolsim import polsim
initial = (kbar1, zhist[0])
ts = 20
simargs = (initial, zhist, nobs, ts, generateVFI, args1, args2, params1, \
           params2) 
khist, ellhist, Yhist, whist, rhist, Thist, chist, ihist, uhist = polsim(simargs)

# SAVE AND PLOT SIMULATION

# write histories
output = open(name + '.pkl', 'wb')
alldata = (khist, ellhist, Yhist, whist, rhist, Thist, chist, ihist, uhist)
pkl.dump(alldata, output)
output.close()


from ILAonesimplots import ILAonesimplots
ILAonesimplots(alldata, name)


