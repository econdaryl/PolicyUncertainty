#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pickle as pkl
from scipy.stats import norm

from BMfuncs import Modeldefs, Modeldyn

# define a function that runs simulation with shift in tax rates
# -----------------------------------------------------------------------------
def polsim(simargs):
    
    # unpack
    (initial, zhist, nobs, ts, funcname, args1, args2, params1, params2) = \
        simargs
    '''
    Generates a history of k & ell with a switch in regime in period ts
    
    This function reads values from the following pkl files:
        ILAfindss.pkl - steady state values and parameters
    
    Inputs
    -----------    
    initial: list of values for k & z (k0, z0) in the first period.
    zhist,: history of z shocks.
    nobs: number of periods to simulate.
    ts: period in which the shift occurs.
    args1: is a list of arguments needed by the solution method in baseline.
        For example, with linearization these are:
        coeffs1: list of (PP, QQ, UU, RR, SS, VV) under the baseline regime.
        XYbar1: numpy array of X & Y SS values under the baseline regime.
    args2: is a list of arguments needed by the solution method after change    
    params1: list of parameters under the baseline regime.
    params2: list of parameters under the new regime.
    z
    
    Returns
    --------
    For the following variables x in (k, z, Y, w, r, T, c, i, u):
        xhist: history of simultated values

    '''
        
    
    # -------------------------------------------------------------------------
    # READ IN VALUES FROM STEADY STATE CALCULATIONS
    
    # load steady state values and parameters
    infile = open('BMfindss.pkl', 'rb')
    (bar1, bar2, temp1, temp2, LINparams) = pkl.load(infile)
    infile.close()
    
    # unpack
    [kbar1, Ybar1, wbar1, rbar1, Tbar1, cbar1, ibar1, ubar1] = bar1
    [kbar2, Ybar2, wbar2, rbar2, Tbar2, cbar2, ibar2, ubar2] = bar2
    [alpha, beta, tau, rho_z, sigma_z] = params1
    (zbar, Zbar, NN, nx, ny, nz, logX, Sylv) = LINparams
    
    # preallocate histories
    khist = np.zeros(nobs+1)
    Yhist = np.zeros(nobs)
    whist = np.zeros(nobs)
    rhist = np.zeros(nobs)
    Thist = np.zeros(nobs)
    chist = np.zeros(nobs)
    ihist = np.zeros(nobs)
    uhist = np.zeros(nobs)
    
    # set starting values
    (khist[0], zhist[0]) = initial
        
    # generate histories for k and ell for the first ts-1 periods
    for t in range(0, ts-1):
        khist[t+1] = funcname(khist[t], zhist[t], args1)
        Yhist[t], whist[t], rhist[t], Thist[t], chist[t], ihist[t], \
            uhist[t] = Modeldefs(khist[t+1], khist[t], zhist[t], \
            params1)

    # generate histories for k and ell for the remaning periods        
    for t in range(ts-1, nobs):
        khist[t+1] = funcname(khist[t], zhist[t], args2)
        Yhist[t], whist[t], rhist[t], Thist[t], chist[t], ihist[t], \
            uhist[t] = Modeldefs(khist[t+1], khist[t], zhist[t], \
            params2)
        
    return khist, Yhist, whist, rhist, Thist, chist, ihist, uhist