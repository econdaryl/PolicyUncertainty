#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Run a series of monte carlo simulations and analyzes them using the
linearization method
'''

import numpy as np
import pickle as pkl
import timeit

from OLG3runmc import runmc
from OLG3mcanalysis import mcanalysis

name = 'OLG3simGSSA_KL'

def generateGSSA(Xm, Zn, args):
    from gssa import poly1
    (pord, nx, ny, nz, coeffs) = args
    XYparams = (pord, nx, ny, nz)
    An = np.exp(Zn)
    XZin = np.append(Xm, An)
    XYbasis = np.append(1., XZin)
    for i in range(1, pord+1):
        XYbasis = poly1(XZin, XYparams)
    XYout = np.dot(XYbasis, coeffs)
    Xn = XYout[0:nx]
    Y = XYout[nx:nx+ny]
    Y[Y > 0.9999] = 0.9999
    Y[Y < 0.0001] = 0.0001
    return Xn, Y

# -----------------------------------------------------------------------------
# LOAD VALUES FROM SS AND LINEARIZATION
    
# load steady state values and parameters
infile = open('OLG3findss.pkl', 'rb')
(bar1, bar2, params1, params2, LINparams) = pkl.load(infile)
infile.close()

# unpack
[k2bar1, k3bar1, l1bar1, l2bar1, Kbar1, \
    Lbar1, GDPbar1, wbar1, rbar1, T3bar1, Bbar1, c1bar1, c2bar1, c3bar1, \
    Cbar1, Ibar1, u1bar1, u2bar1, u3bar1] = bar1
[k2bar2, k3bar2, l1bar2, l2bar2, Kbar2, \
    Lbar2, GDPbar2, wbar2, rbar2, T3bar2, Bbar2, c1bar2, c2bar2, c3bar2, \
    Cbar2, Ibar2, u1bar2, u2bar2, u3bar2] = bar2
[alpha, beta, gamma, delta, chi, theta, tau, rho_z, \
    sigma_z, pi2, pi3, f1, f2, nx, ny, nz] = params1
tau2 = params2[6]
(zbar, Zbar, NN, nx, ny, nz, logX, Sylv) = LINparams
    
# load GSSA coeffs
infile = open('OLG3solveGSSA_3.pkl', 'rb')
(coeffs1, coeffs2, timesolve) = pkl.load(infile)
infile.close()

# create args lists
XYbar1 = (k2bar1, k3bar1, l1bar1, l2bar1)
XYbar2 = (k2bar2, k3bar2, l1bar2, l2bar2)
pord = 3
args1 = (pord, nx, ny, nz, coeffs1)
args2 = (pord, nx, ny, nz, coeffs2)

# -----------------------------------------------------------------------------
# RUN MONTE CARLOS

# start clock for all MCs
startsim = timeit.default_timer()

# specify the number of observations per simulation
nobs = 20
# specify the period policy shifts
ts = 2
# specify the number of simulations
nsim = 100000
# specify the increment between MC reports
repincr = 100

# specify initial values
k20 = XYbar1[0]
k30 = XYbar1[1]
z0 = 0.
initial = (k20, k30, z0)

# get list of arguments for monte carlos simulations 
simargs = (initial, nobs, ts, generateGSSA, args1, args2, params1, params2)

# run the Monte Carlos
mcdata, histdata, preddata, act = runmc(simargs, nsim, nobs, repincr)

# calculate time to simulate all MCs
stopsim = timeit.default_timer()
timesim = stopsim - startsim
print('time to simulate', nsim, 'monte carlos: ', timesim)

# -----------------------------------------------------------------------------
# DO ANALYSIS

# load data for plots
bardata = (k2bar1, k3bar1, l1bar1, l2bar1, Kbar1, \
     Lbar1, GDPbar1, wbar1, rbar1, T3bar1, Bbar1, c1bar1, c2bar1, c3bar1, \
     Cbar1, Ibar1, u1bar1, u2bar1, u3bar1)
  
avgdata, uppdata, lowdata = \
    mcanalysis(mcdata, preddata, bar1, histdata, name, nsim)
    
# unpack
(k2avg, k3avg, l1avg, l2avg, zavg, Kavg, Lavg, \
    GDPavg, wavg, ravg, T3avg, Bavg, c1avg, c2avg, c3avg, \
    Cavg, Iavg, u1avg, u2avg, u3avg, foremeanavg, \
    forevaravg, zformeanavg, zforvaravg, RMsqEerravg) = avgdata
(k2upp, k3upp, l1upp, l2upp, zupp, Kupp, Lupp, \
    GDPupp, wupp, rupp, T3upp, Bupp, c1upp, c2upp, c3upp, \
    Cupp, Iupp, u1upp, u2upp, u3upp, \
    foremeanupp, forevarupp, zformeanupp, zforvarup) = uppdata
(k2low, k3low, l1low, l2low, zlow, Klow, Llow, \
    GDPlow, wlow, rlow, T3low, Blow, c1low, c2low, c3low, \
    Clow, Ilow, u1low, u2low, u3low, \
    foremeanlow, forevarlow, zformeanlow, zformvarlow) = lowdata
    
foreperc = np.delete(foremeanavg, 4, 0)/np.abs(bar1)
print('1-period-ahead average forecast errors')
print(foreperc)

forevarc = (np.delete(forevaravg, 4, 0))**.5/np.abs(bar1)
print('1-period-ahead RMSE forecast errors')
print(forevarc)

zforperc = np.delete(zformeanavg, 4, 0)/np.abs(bar1)
print('period-0 average forecast errors')
print(zforperc)

zforvarc = (np.delete(zformeanavg, 4, 0))**.5/np.abs(bar1)
print('period-0 RMSE forecast errors')
print(zforvarc)

# -----------------------------------------------------------------------------
# SAVE RESULTS

output = open(name + '.pkl', 'wb')

# write timing
pkl.dump(timesim, output)

# write monte carlo results
alldata = (preddata, avgdata, uppdata, lowdata, foreperc, forevarc, zforperc, \
           zforvarc, RMsqEerravg, act)
pkl.dump(alldata, output)

output.close()
