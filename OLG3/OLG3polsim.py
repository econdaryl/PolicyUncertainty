#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pickle as pkl
from scipy.stats import norm

from OLG3funcs import Modeldefs, Modeldyn

# define a function that runs simulation with shift in tax rates
# -----------------------------------------------------------------------------
def polsim(simargs):
    
    # unpack
    (initial, nobs, ts, funcname, args1, args2, params1, params2) = simargs
    '''
    Generates a history of k & ell with a switch in regime in period ts
    
    This function reads values from the following pkl files:
        OLGfindss.pkl - steady state values and parameters
    
    Inputs
    -----------    
    initial: list of values for k & z (k0, z0) in the first period.
    nobs: number of periods to simulate.
    ts: period in which the shift occurs.
    args1: is a list of arguments needed by the solution method in baseline.
        For example, with linearization these are:
        coeffs1: list of (PP, QQ, UU, RR, SS, VV) under the baseline regime.
        XYbar1: numpy array of X & Y SS values under the baseline regime.
    args2: is a list of arguments needed by the solution method after change    
    params1: list of parameters under the baseline regime.
    params2: list of parameters under the new regime.
    
    Returns
    --------
    For the following variables x in (k, ell, z, Y, w, r, T, c, i, u):
        xhist: history of simultated values
        xfhist: history of one-period-ahed forecasts
        MsqEerr: root mean squared Euler errors
    '''
        
    
    # -------------------------------------------------------------------------
    # READ IN VALUES FROM STEADY STATE CALCULATIONS
    
    # load steady state values and parameters
    infile = open('OLG3findss.pkl', 'rb')
    (bar1, bar2, temp1, temp2, LINparams) = pkl.load(infile)
    infile.close()
    
# unpack
    [k2bar1, k3bar1, l1bar1, l2bar1, Kbar1, \
        Lbar1, GDPbar1, wbar1, rbar1, T3bar1, Bbar1, c1bar1, c2bar1, c3bar1, \
        Cbar1, Ibar1, u1bar1, u2bar1, u3bar1] = bar1
    [k2bar2, k3bar2, l1bar2, l2bar2, Kbar2, \
        Lbar2, GDPbar2, wbar2, rbar2, T3bar2, Bbar2, c1bar2, c2bar2, c3bar2, \
        Cbar2, Ibar2, u1bar2, u2bar2, u3bar2] = bar2
    (zbar, Zbar, NN, nx, ny, nz, logX, Sylv) = LINparams
    
    # set parameter values for calculating Euler errors
    npts = 10 # number of point for rectangular quadrature
    # generate discret support for epsilon to be used in Euler error
    # Eps are the central values
    # Phi are the associated probabilities
    Eps = np.zeros(npts);
    Cum = np.linspace(0.0, 1.0, num=npts+1)+.5/npts
    Cum = Cum[0:npts]
    Phi = np.ones(npts)/npts
    Eps = norm.ppf(Cum)

    
    # preallocate histories
    k2hist = np.zeros(nobs+1)
    k3hist = np.zeros(nobs+1)
    l1hist = np.zeros(nobs)
    l2hist = np.zeros(nobs)
    zhist = np.zeros(nobs)
    Khist = np.zeros(nobs)
    Lhist = np.zeros(nobs)
    GDPhist = np.zeros(nobs)
    whist = np.zeros(nobs)
    rhist = np.zeros(nobs)
    T3hist = np.zeros(nobs)
    Bhist = np.zeros(nobs)
    c1hist = np.zeros(nobs)
    c2hist = np.zeros(nobs)
    c3hist = np.zeros(nobs)
    Chist = np.zeros(nobs)
    Ihist = np.zeros(nobs)
    u1hist = np.zeros(nobs)
    u2hist = np.zeros(nobs)
    u3hist = np.zeros(nobs)
    RMsqEerrhist = np.zeros((nobs, nx+ny))
    
    # preallocate forecast histories
    k2fhist = np.zeros(nobs+2)
    k3fhist = np.zeros(nobs+2)
    l1fhist = np.zeros(nobs+1)
    l2fhist = np.zeros(nobs+1)
    zfhist = np.zeros(nobs+1)
    Kfhist = np.zeros(nobs+1)
    Lfhist = np.zeros(nobs+1)
    GDPfhist = np.zeros(nobs+1)
    wfhist = np.zeros(nobs+1)
    rfhist = np.zeros(nobs+1)
    T3fhist = np.zeros(nobs+1)
    Bfhist = np.zeros(nobs+1)
    c1fhist = np.zeros(nobs+1)
    c2fhist = np.zeros(nobs+1)
    c3fhist = np.zeros(nobs+1)
    Cfhist = np.zeros(nobs+1)
    Ifhist = np.zeros(nobs+1)
    u1fhist = np.zeros(nobs+1)
    u2fhist = np.zeros(nobs+1)
    u3fhist = np.zeros(nobs+1)
    
    # upack simulation parameters
    rho_z = params1[7] 
    sigma_z = params1[8]
    
    # set starting values
    (k2hist[0], k3hist[0], zhist[0]) = initial
    
    
    
    # generate history of random shocks
    for t in range(1, nobs):
        zhist[t] = rho_z*zhist[t-1] + sigma_z*np.random.normal(0., 1.)
        
    # generate histories for the first ts-1 periods
    for t in range(0, ts-1):
        k = np.array([k2hist[t], k3hist[t]])
        kp, l = funcname(k, zhist[t], args1)
        [k2hist[t+1], k3hist[t+1]] = kp
        [l1hist[t], l2hist[t]] = l
        Khist[t], Lhist[t], GDPhist[t], whist[t], rhist[t], T3hist[t], \
            Bhist[t], c1hist[t], c2hist[t], c3hist[t], Chist[t], \
            Ihist[t], u1hist[t], u2hist[t], u3hist[t] \
            = Modeldefs(kp, k, l, zhist[t], params1)
        
        # get 1-period ahead forecasts
        if t < ts-2:  # use baseline model for predictions
            zfhist[t+1] = rho_z*zhist[t]
            kfp, lf = funcname(kp, zfhist[t+1], args1)
            [k2fhist[t+2], k3fhist[t+2]] = kp
            [l1fhist[t+1], l2fhist[t+1]] = l
            Kfhist[t+1], Lfhist[t+1], GDPfhist[t+1], wfhist[t+1], rfhist[t+1],\
                T3fhist[t+1], Bfhist[t+1], c1fhist[t+1], c2fhist[t+1], \
                c3fhist[t+1], Cfhist[t+1], Ifhist[t+1], \
                u1fhist[t+1], u2fhist[t+1], u3fhist[t+1] \
                = Modeldefs(kfp, kp, lf, zfhist[t+1], params1)
            
            # begin loop over possible values of shock next period for Euler errors
            MsqEerr = np.zeros(nx + ny)
            for i in range(0, npts):
                # find value of next period z
                zp = rho_z*zhist[t] + sigma_z*Eps[i]
                # find the value of k in two periods
                kpp, lp = funcname(kp, zp, args1)
                # find the Euler errors
                [k2pp, k3pp] = kpp
                [k2p, k3p] = kp
                [k2, k3] = kp
                [l1p, l2p] = lp
                [l1, l2] = l
                invec = np.array([k2pp, k3pp, k2p, k3p, k2, k3, \
                    l1p, l2p, l1, l2, zp, zhist[t]])
                Eerr = Phi[i]*Modeldyn(invec, params1)
                MsqEerr = 1/(1+i) * Eerr**2 + i/(1+i) * MsqEerr
            RMsqEerrhist[t,:] = MsqEerr**.5    
                
        else:  # use change model for predictions
            zfhist[t+1] = rho_z*zhist[t]
            kfp, lf = funcname(kp, zfhist[t+1], args2)
            Kfhist[t+1], Lfhist[t+1], GDPfhist[t+1], wfhist[t+1], rfhist[t+1],\
                T3fhist[t+1], Bfhist[t+1], c1fhist[t+1], c2fhist[t+1], \
                c3fhist[t+1], Cfhist[t+1], Ifhist[t+1], \
                u1fhist[t+1], u2fhist[t+1], u3fhist[t+1] \
                = Modeldefs(kfp, kp, lf, zfhist[t+1], params2)

            # begin loop over possible values of shock next period for Euler errors
            MsqEerr = np.zeros(nx + ny)
            for i in range(0, npts):
                # find value of next period z
                zp = rho_z*zhist[t] + sigma_z*Eps[i]
                # find the value of k in two periods
                kpp, lp = funcname(kp, zp, args2)
                # find the Euler errors
                [k2pp, k3pp] = kpp
                [k2p, k3p] = kp
                [k2, k3] = kp
                [l1p, l2p] = lp
                [l1, l2] = l
                invec = np.array([k2pp, k3pp, k2p, k3p, k2, k3, \
                    l1p, l2p, l1, l2, zp, zhist[t]])
                Eerr = Phi[i]*Modeldyn(invec, params2)
                MsqEerr = 1/(1+i) * Eerr**2 + i/(1+i) * MsqEerr
            RMsqEerrhist[t,:] = MsqEerr**.5         

            
    # generate histories for the remaining periods
    for t in range(ts-1, nobs):
        k = np.array([k2hist[t], k3hist[t]])
        kp, l = funcname(k, zhist[t], args2)
        [k2hist[t+1], k3hist[t+1]] = kp
        [l1hist[t], l2hist[t]] = l
        Khist[t], Lhist[t], GDPhist[t], whist[t], rhist[t], T3hist[t], \
            Bhist[t], c1hist[t], c2hist[t], c3hist[t], Chist[t], \
            Ihist[t], u1hist[t], u2hist[t], u3hist[t] \
            = Modeldefs(kp, k, l, zhist[t], params2)
        
        # get 1-period ahead forecasts
        zfhist[t+1] = rho_z*zhist[t]
        kfp, lf = funcname(kp, zfhist[t+1], args2)
        [k2fhist[t+2], k3fhist[t+2]] = kp
        [l1fhist[t+1], l2fhist[t+1]] = l
        Kfhist[t+1], Lfhist[t+1], GDPfhist[t+1], wfhist[t+1], rfhist[t+1],\
            T3fhist[t+1], Bfhist[t+1], c1fhist[t+1], c2fhist[t+1], \
            c3fhist[t+1], Cfhist[t+1], Ifhist[t+1], \
            u1fhist[t+1], u2fhist[t+1], u3fhist[t+1] \
            = Modeldefs(kfp, kp, lf, zfhist[t+1], params2)
            
        # begin loop over possible values of shock next period for Euler errors
        MsqEerr = np.zeros(nx + ny)
        for i in range(0, npts):
            # find value of next period z
            zp = rho_z*zhist[t] + sigma_z*Eps[i]
            # find the value of k in two periods
            kpp, lp = funcname(kp, zp, args2)
            # find the Euler errors
            [k2pp, k3pp] = kpp
            [k2p, k3p] = kp
            [k2, k3] = kp
            [l1p, l2p] = lp
            [l1, l2] = l
            invec = np.array([k2pp, k3pp, k2p, k3p, k2, k3, \
                l1p, l2p, l1, l2, zp, zhist[t]])
            Eerr = Phi[i]*Modeldyn(invec, params2)
            MsqEerr = 1/(1+i) * Eerr**2 + i/(1+i) * MsqEerr
        RMsqEerrhist[t,:] = MsqEerr**.5  
    
    return k2hist, k3hist, l1hist, l2hist, zhist, GDPhist, \
        Khist, Lhist, whist, rhist, T3hist, Bhist, c1hist, c2hist, c3hist, \
        Chist, Ihist, u1hist, u2hist, u3hist, k2fhist, \
        k3fhist, l1fhist, l2fhist, zfhist, GDPfhist, \
        Kfhist, Lfhist, wfhist, rfhist, T3fhist, Bfhist, c1fhist, c2fhist, \
        c3fhist, Cfhist, Ifhist, u1fhist, u2fhist, u3fhist, RMsqEerrhist