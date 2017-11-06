#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This function reads in paramter values and steady states from the file, 
ILAfindss.pkl, and 

"""
import numpy as np
import pickle as pkl

from Simple_ILA_Model_Funcs import Modeldefs



# define a function that runs simulation with shift in tax rates
# -----------------------------------------------------------------------------
def polsim(simargs):
    
    # unpack
    (initial, nobs, ts, funcname, args1, args2, params1, params2) = simargs
    '''
    Generates a history of k & ell with a switch in regime in period ts.
    
    Inputs
    -----------    
    initial: list of values for k & z (k0, z0) in the first period.
    nobs: number of periods to simulate.
    ts: period in which the shift occurs.
    args1: is a list of arguments needed by the solution method in baseline.
        For linearization these are:
        coeffs1: list of (PP, QQ, UU, RR, SS, VV) under the baseline regime.
        XYbar1: numpy array of X & Y bSS values under the baseline regime.
        params1: list of parameters under the baseline regime.
    args2: is a list of arguments needed by the solution method after change    
        coeffs2: list of (PP2, QQ2, UU2, RR2, SS2, VV2) under the new regime.
        XYbar2: numpy array of X & Y bSS values under the new regime.
        params2: list of parameters under the new regime.
    
    Returns
    --------
    khist: 2D-array, dtype=float
        nobs-by-1 matrix containing the values of k
    
    ellhist: 2D-array, dtype=float
        nobs-by-1 matrix vector containing the values of ell 
        
    zhist: 2D-array, dtype=float
        nobs-by-1 matrix vector containing the values of z 
    '''
        
    
    # -----------------------------------------------------------------------------
    # READ IN VALUES FROM STEADY STATE CALCULATIONS
    
    # load steady state values and parameters
    infile = open('ILAfindss.pkl', 'rb')
    (bar1, bar2, temp1, temp2, LINparams) = pkl.load(infile)
    infile.close()
    
    # unpack
    [kbar1, ellbar1, Ybar1, wbar1, rbar1, Tbar1, cbar1, ibar1, ubar1] = bar1
    [kbar2, ellbar2, Ybar2, wbar2, rbar2, Tbar2, cbar2, ibar2, ubar2] = bar2
    [alpha, beta, gamma, delta, chi, theta, tau, rho_z, sigma_z] = params1
    (zbar, Zbar, NN, nx, ny, nz, logX, Sylv) = LINparams

    
    # preallocate histories
    khist = np.zeros(nobs+1)
    ellhist = np.zeros(nobs)
    zhist = np.zeros(nobs)
    Yhist = np.zeros(nobs)
    whist = np.zeros(nobs)
    rhist = np.zeros(nobs)
    Thist = np.zeros(nobs)
    chist = np.zeros(nobs)
    ihist = np.zeros(nobs)
    uhist = np.zeros(nobs)
    
    # preallocate forecast histories
    kfhist = np.zeros(nobs+2)
    ellfhist = np.zeros(nobs+1)
    zfhist = np.zeros(nobs+1)
    Yfhist = np.zeros(nobs+1)
    wfhist = np.zeros(nobs+1)
    rfhist = np.zeros(nobs+1)
    Tfhist = np.zeros(nobs+1)
    cfhist = np.zeros(nobs+1)
    ifhist = np.zeros(nobs+1)
    ufhist = np.zeros(nobs+1)
    
    # upack simulation parameters
    rho_z = params1[7] 
    sigma_z = params1[8]
    
    # set starting values
    khist[0] = kbar1
    zhist[0] = zbar
    
    
    # generate history of random shocks
    for t in range(1, nobs):
        zhist[t] = rho_z*zhist[t] + sigma_z*np.random.normal(0., 1.)
        
    # generate histories for k and ell for the first ts-1 periods
    for t in range(0, ts-1):
        khist[t+1], ellhist[t] = funcname(khist[t], zhist[t], args1)
        Yhist[t], whist[t], rhist[t], Thist[t], chist[t], ihist[t], \
            uhist[t] = Modeldefs(khist[t+1], khist[t], ellhist[t], zhist[t], \
            params1)
        
        # get 1-period ahead forecasts
        zfhist[t+1] = rho_z*zhist[t]
        kfhist[t+2], ellfhist[t+1] = funcname(khist[t+1], zfhist[t+1], args1)
        Yfhist[t+1], wfhist[t+1], rfhist[t+1], Tfhist[t+1], cfhist[t+1], \
            ifhist[t], ufhist[t] = Modeldefs(kfhist[t+2], khist[t+1], \
            ellfhist[t+1], zfhist[t+1], params1)
        
    
    for t in range(ts-1, nobs):
        khist[t+1], ellhist[t] = funcname(khist[t], zhist[t], args2)
        Yhist[t], whist[t], rhist[t], Thist[t], chist[t], ihist[t], \
            uhist[t] = Modeldefs(khist[t+1], khist[t], ellhist[t], zhist[t], \
            params2)
        
        # get 1-period ahead forecasts
        zfhist[t+1] = rho_z*zhist[t]
        kfhist[t+2], ellfhist[t+1] = funcname(khist[t+1], zfhist[t+1], args2)
        Yfhist[t+1], wfhist[t+1], rfhist[t+1], Tfhist[t+1], cfhist[t+1], \
            ifhist[t], ufhist[t] = Modeldefs(kfhist[t+2], khist[t+1], \
            ellfhist[t+1], zfhist[t+1], params2)
    
    return khist, ellhist, zhist, Yhist, whist, rhist, Thist, chist, ihist, \
        uhist, kfhist, ellfhist, zfhist, Yfhist, wfhist, rfhist, Tfhist, \
        cfhist, ifhist, ufhist
        
