#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Run a series of monte carlo simulations and analyzes them using the
Linearization method
'''

import numpy as np
import pickle as pkl
import timeit

from ILArunmc import runmc
from ILAmcanalysis import mcanalysis

def generateLIN(k, z, args):
    from LinApp_Sim import LinApp_Sim
    
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
    (PP, QQ, UU, RR, SS, VV) = coeffs
    (kbar, ellbar) = XYbar
    
    # inputs must be 1D numpy arrays and deviation from SS values
    ktil = np.array([k - kbar])
    ztil = np.array([z])
    kptil, elltil = LinApp_Sim(ktil, ztil, PP, QQ, UU, RR, SS, VV)
    # k and ell are deviations from SS values, so add these back.
    # they are also 1D numpy arrays, so pull out the values rather than 
    # use the arrays.
    kp = kptil + kbar
    ell = elltil + ellbar
    
    return kp, ell

name = 'ILAonesimLIN'

# -----------------------------------------------------------------------------
# LOAD VALUES FROM SS and Zhist pickle file
    
# load steady state values and parameters
infile = open('ILAfindss.pkl', 'rb')
(bar1, bar2, params1, params2, LINparams) = pkl.load(infile)
infile.close()

infile = open('ILAsolveLIN.pkl', 'rb')
(coeffs1, coeffs2, timesolve) = pkl.load(infile)
infile.close()


# unpack
[kbar1, ellbar1, Ybar1, wbar1, rbar1, Tbar1, cbar1, ibar1, ubar1] = bar1
[kbar2, ellbar2, Ybar2, wbar2, rbar2, Tbar2, cbar2, ibar2, ubar2] = bar2
[alpha, beta, gamma, delta, chi, theta, tau2, rho_z, sigma_z] = params1
tau2 = params2[6]
(zbar, Zbar, NN, nx, ny, nz, logX, Sylv) = LINparams
    
# create args lists
XYbar1 = (kbar1, ellbar1)
XYbar2 = (kbar2, ellbar2)
args1 = (coeffs1, XYbar1)
args2 = (coeffs2, XYbar2)

# load zhist
infile = open('ILAzhist.pkl', 'rb')
(nobs, zhist) = pkl.load(infile)
infile.close()

# RUN SINGLE SIMULATION
from ILAonepolsim import polsim
initial = (kbar1, zhist[0])
ts = 20
simargs = (initial, zhist, nobs, ts, generateLIN, args1, args2, params1, \
           params2) 
khist, ellhist, Yhist, whist, rhist, Thist, chist, ihist, uhist = polsim(simargs)

# SAVE AND PLOT SIMULATIOn

# write histories
output = open(name + '.pkl', 'wb')
alldata = (khist, ellhist, Yhist, whist, rhist, Thist, chist, ihist, uhist)
pkl.dump(alldata, output)
output.close()

from ILAonesimplots import ILAonesimplots
ILAonesimplots(alldata, name)
