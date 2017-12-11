#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Run a series of monte carlo simulations and analyzes them using the
linearization method
'''

import numpy as np
import pickle as pkl
import timeit

from ILArunmc import runmc
from ILAmcanalysis import mcanalysis

name = 'baseline'

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


# -----------------------------------------------------------------------------
# LOAD VALUES FROM SS AND LINEARIZATION
    
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
    
# load Linearization coeffs
infile = open('ILAsolveLIN.pkl', 'rb')
(coeffs1, coeffs2, timesolve) = pkl.load(infile)
infile.close()

# create args lists
XYbar1 = (kbar1, ellbar1)
XYbar2 = (kbar2, ellbar2)
args1 = (coeffs1, XYbar1)
args2 = (coeffs2, XYbar2)

# -----------------------------------------------------------------------------
# RUN MONTE CARLOS

# start clock for all MCs
startsim = timeit.default_timer()

# specify the number of observations per simulation
nobs = 120
# specify the period policy shifts
ts = 20
# specify the number of simulations
nsim = 1000
# specify the increment between MC reports
repincr = 100

# specify initial values
k0 = kbar1
z0 = 0.
initial = (k0, z0)

# arguments and parameters for time zero prediction
# parameters for tau1 portion
params3 = np.array([alpha, beta, gamma, delta, chi, theta, tau, rho_z, 0.])
# paramters for tau2 portion
params4 = np.array([alpha, beta, gamma, delta, chi, theta, tau2, rho_z, 0.])
# get list of arguments for predictions simulation
predargs = (initial, nobs, ts, generateLIN, args1, args2, params3, params4)

# get list of arguments for monte carlos simulations 
simargs = (initial, nobs, ts, generateLIN, args1, args2, params1, params2)

# run the Monte Carlos
mcdata, histdata, preddata, act = runmc(simargs, nsim, nobs, repincr)

# calculate time to simulate all MCs
stopsim = timeit.default_timer()
timesim = stopsim - startsim
print('time to simulate', nsim, 'monte carlos: ', timesim)

# -----------------------------------------------------------------------------
# DO ANALYSIS

# load data for plots
bardata = (kbar1, ellbar1, zbar, Ybar1, wbar1, rbar1, Tbar1, cbar1, ibar1, 
           ubar1)
  
avgdata, uppdata, lowdata = \
    mcanalysis(mcdata, preddata, bardata, histdata, name, nsim)
    
# unpack
(kavg, ellavg, zavg, Yavg, wavg, ravg, Tavg, cavg, iavg, uavg, foremeanavg, \
     zformeanavg, MsqEerravg) = avgdata
(kupp, ellupp, zupp, Yupp, wupp, rupp, Tupp, cupp, iupp, uupp, foremeanupp, \
     zformeanupp) = uppdata
(klow, elllow, zlow, Ylow, wlow, rlow, Tlow, clow, ilow, ulow, foremeanlow, \
     zformeanlow) = lowdata
    
foreperc = np.delete(foremeanavg, 2, 0)/np.abs(bar1)
print('1-period-ahead average forecast errors')
print(foreperc)

zforperc = np.delete(zformeanavg, 2, 0)/np.abs(bar1)
print('period-0 average forecast errors')
print(zforperc)

print('root mean squared Euler errors')
print(MsqEerravg)

# -----------------------------------------------------------------------------
# SAVE RESULTS

output = open(name + '.pkl', 'wb')

# write timing
pkl.dump(timesim, output)

# write monte carlo results
alldata = (preddata, avgdata, uppdata, lowdata, foreperc, zforperc, \
           MsqEerravg, act)
pkl.dump(alldata, output)

output.close()
