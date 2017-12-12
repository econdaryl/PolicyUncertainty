#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Run a series of monte carlo simulations and analyzes them using the
linearization method
'''

import numpy as np
import pickle as pkl
import timeit

from OLGrunmc import runmc
from OLGmcanalysis import mcanalysis

name = 'OLGsimLIN'

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
    kbar = XYbar[0:3]
    lbar = XYbar[3:6]
    
    # inputs must be 1D numpy arrays and deviation from SS values
    ktil = k - kbar
    ztil = np.array([z])
    kptil, ltil  = LinApp_Sim(ktil, ztil, PP, QQ, UU, RR, SS, VV)
    # k and ell are deviations from SS values, so add these back.
    # they are also 1D numpy arrays, so pull out the values rather than 
    # use the arrays.
    kp = kptil + kbar
    l = ltil + lbar
    
    return kp, l


# -----------------------------------------------------------------------------
# LOAD VALUES FROM SS AND LINEARIZATION
    
# load steady state values and parameters
infile = open('OLGfindss.pkl', 'rb')
(bar1, bar2, params1, params2, LINparams) = pkl.load(infile)
infile.close()

# unpack
[k2bar1, k3bar1, k4bar1, l1bar1, l2bar1, l3bar1, Kbar1, \
    Lbar1, GDPbar1, wbar1, rbar1, T4bar1, Bbar1, c1bar1, c2bar1, c3bar1, \
    c4bar1, Cbar1, Ibar1, u1bar1, u2bar1, u3bar1, u4bar1] = bar1
[k2bar2, k3bar2, k4bar2, l1bar2, l2bar2, l3bar2, Kbar2, \
    Lbar2, GDPbar2, wbar2, rbar2, T4bar2, Bbar2, c1bar2, c2bar2, c3bar2, \
    c4bar2, Cbar2, Ibar2, u1bar2, u2bar2, u3bar2, u4bar2] = bar2
[alpha, beta, gamma, delta, chi, theta, tau, rho_z, \
    sigma_z, pi2, pi3, pi4, f1, f2, f3, nx, ny, nz] = params1
tau2 = params2[6]
(zbar, Zbar, NN, nx, ny, nz, logX, Sylv) = LINparams
    
# load Linearization coeffs
infile = open('OLGsolveLIN.pkl', 'rb')
(coeffs1, coeffs2, timesolve) = pkl.load(infile)
infile.close()

# create args lists
XYbar1 = (k2bar1, k3bar1, k4bar1, l1bar1, l2bar1, l3bar1)
XYbar2 = (k2bar2, k3bar2, k4bar2, l1bar2, l2bar2, l3bar2)
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
nsim = 100
# specify the increment between MC reports
repincr = 100

# specify initial values
k20 = XYbar1[0]
k30 = XYbar1[1]
k40 = XYbar1[2]
z0 = 0.
initial = (k20, k30, k40, z0)

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
bardata = (k2bar1, k3bar1, k4bar1, l1bar1, l2bar1, l3bar1, Kbar1, \
     Lbar1, GDPbar1, wbar1, rbar1, T4bar1, Bbar1, c1bar1, c2bar1, c3bar1, \
     c4bar1, Cbar1, Ibar1, u1bar1, u2bar1, u3bar1, u4bar1)
  
avgdata, uppdata, lowdata = \
    mcanalysis(mcdata, preddata, bar1, histdata, name, nsim)
    
# unpack
(k2avg, k3avg, k4avg, l1avg, l2avg, l3avg, zavg, Kavg, Lavg, \
    GDPavg, wavg, ravg, T4avg, Bavg, c1avg, c2avg, c3avg, c4avg, \
    Cavg, Iavg, u1avg, u2avg, u3avg, u4avg, foremeanavg, \
    zformeanavg, RMsqEerravg) = avgdata
(k2upp, k3upp, k4upp, l1upp, l2upp, l3upp, zupp, Kupp, Lupp, \
    GDPupp, wupp, rupp, T4upp, Bupp, c1upp, c2upp, c3upp, c4upp, \
    Cupp, Iupp, u1upp, u2upp, u3upp, u4upp, \
    foremeanupp, zformeanupp) = uppdata
(k2low, k3low, k4low, l1low, l2low, l3low, zlow, Klow, Llow, \
    GDPlow, wlow, rlow, T4low, Blow, c1low, c2low, c3low, c4low, \
    Clow, Ilow, u1low, u2low, u3low, u4low, \
    foremeanlow, zformeanlow) = lowdata
    
foreperc = np.delete(foremeanavg, 2, 0)/np.abs(bar1)
print('1-period-ahead average forecast errors')
print(foreperc)

zforperc = np.delete(zformeanavg, 2, 0)/np.abs(bar1)
print('period-0 average forecast errors')
print(zforperc)

print('root mean squared Euler errors')
print(RMsqEerravg)

# -----------------------------------------------------------------------------
# SAVE RESULTS

output = open(name + '.pkl', 'wb')

# write timing
pkl.dump(timesim, output)

# write monte carlo results
alldata = (preddata, avgdata, uppdata, lowdata, foreperc, zforperc, \
           RMsqEerravg, act)
pkl.dump(alldata, output)

output.close()
