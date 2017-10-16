#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 16:08:19 2017

@author: klp4
"""

import numpy as np

# import the modules from LinApp
from LinApp_FindSS import LinApp_FindSS

from Simple_ILA_Model_Funcs import Modeldefs, Modeldyn

# set name for external files written
name = 'ILAGSSA'

# set parameter values
alpha = .35
beta = .99
gamma = 2.5
delta = .08
chi = 10.
theta = 2.
tau = .05
rho_z = .9
sigma_z = .01

# make parameter list to pass to functions
params = np.array([alpha, beta, gamma, delta, chi, theta, tau, rho_z, sigma_z])

# set LinApp parameters
Zbar = np.array([0.])
nx = 1
ny = 1
nz = 1
logX = 0
Sylv = 0

# take a guess for steady state values of k and ell
guessXY = np.array([.1, .25])

# find the steady state values using LinApp_FindSS
XYbar = LinApp_FindSS(Modeldyn, params, guessXY, Zbar, nx, ny)
(kbar, ellbar) = XYbar

# set up steady state input vector
theta0 = np.array([kbar, kbar, kbar, ellbar, ellbar, 0., 0.])

# check SS solution
check = Modeldyn(theta0, params)
print ('check SS: ', check)
if np.max(np.abs(check)) > 1.E-6:
    print ('Have NOT found steady state')
    
# find the steady state values for the definitions
Ybar, wbar, rbar, Tbar, cbar, ibar, ubar = \
    Modeldefs(kbar, kbar, ellbar, 0., params)

# display all steady state values
print ('kbar:   ', kbar)
print ('ellbar: ', ellbar)
print ('Ybar:   ', Ybar)
print ('wbar:   ', wbar)
print ('rbar:   ', rbar)
print ('Tbar:   ', Tbar)
print ('cbar:   ', cbar)
print ('ibar:   ', ibar)
print ('ubar:   ', ubar)

##### Insert GSSA Solution Code ######################################################

# generate a history of Z's
nobs = 250
Zhist = np.zeros((nobs,1))
for t in range(1, nobs):
    Zhist[t,0] = rho_z*Zhist[t,0] + sigma_z*np.random.normal(0., 1.)
    
# put SS values and starting values into numpy vectors
XYbar = np.array([kbar, ellbar])
X0 = np.array([kbar])
Y0 = np.array([ellbar])


## CHANGE POLICY

# set new tax rate
tau2 = .055

# make parameter list to pass to functions
params2 = np.array([alpha, beta, gamma, delta, chi, theta, tau2, rho_z, 
                    sigma_z])

# find new steady state
# use the old steady state values of k and ell for our guess
guessXY = XYbar

# find the steady state values using LinApp_FindSS
XYbar2 = LinApp_FindSS(Modeldyn, params2, guessXY, Zbar, nx, ny)
(kbar2, ellbar2) = XYbar2

# set up steady state input vector
theta02 = np.array([kbar2, kbar2, kbar2, ellbar2, ellbar2, 0., 0.])

# check SS solution
check = Modeldyn(theta02, params2)
print ('check SS: ', check)
if np.max(np.abs(check)) > 1.E-6:
    print ('Have NOT found steady state')
    
# find the steady state values for the definitions
Ybar2, wbar2, rbar2, Tbar2, cbar2, ibar2, ubar2 = \
    Modeldefs(kbar2, kbar2, ellbar2, 0., params2)

# display all steady state values
print ('kbar:   ', kbar2)
print ('ellbar: ', ellbar2)
print ('Ybar:   ', Ybar2)
print ('wbar:   ', wbar2)
print ('rbar:   ', rbar2)
print ('Tbar:   ', Tbar2)
print ('cbar:   ', cbar2)
print ('ibar:   ', ibar2)
print ('ubar:   ', ubar2)

##### Insert GSSA Solution Code ######################################################

def PolSim(initial, nobs, ts, coeffs1, state1, params1, coeffs2, state2, \
           params2):
    '''
    Generates a history of k & ell with a switch in regime in period ts.
    
    Inputs
    -----------    
    initial: list of values for k & z (k0, z0) in the first period.
    nobs: number of periods to simulate.
    ts: period in which the shift occurs.
    coeffs1: list of (PP, QQ, UU, RR, SS, VV) under the baseline regime.
    state1: numpy array of XYbar under the baseline regime.
    params1: list of parameters under the baseline regime.
    coeffs2: list of (PP2, QQ2, UU2, RR2, SS2, VV2) under the new regime.
    state2: numpy array of XYbar2 under the new regime.
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
    
    # upack simulation parameters
    rho_z = params1[7] 
    sigma_z = params1[8]
    
    # set starting values
    khist[0] = k0
    zhist[0] = z0
    
    # unpack state1 and state2
    (kbar, ellbar) = XYbar
    (kbar2, ellbar2) = XYbar2
    
    # generate history of random shocks
    for t in range(1, nobs):
        zhist[t] = rho_z*zhist[t] + sigma_z*np.random.normal(0., 1.)
        
    # generate histories for k and ell for the first ts-1 periods
    for t in range(0, ts-1):
        # inputs must be 1D numpy arrays and deviation from SS values
        # USE GSSA polynomials ######################################################
        Yhist[t], whist[t], rhist[t], Thist[t], chist[t], ihist[t], \
            uhist[t] = \
            Modeldefs(khist[t+1], khist[t], ellhist[t], zhist[t], params)
    
    for t in range(ts-1, nobs):
        # USE GSSA polynomials ######################################################
        Yhist[t], whist[t], rhist[t], Thist[t], chist[t], ihist[t], \
            uhist[t] = \
            Modeldefs(khist[t+1], khist[t], ellhist[t], zhist[t], params2)
    
    return khist, ellhist, zhist, Yhist, whist, rhist, Thist, chist, ihist, \
        uhist



# specify the number of observations per simulation
nobs = 120

# specify the period policy shifts
ts = 20

# specify initial values
k0 = kbar
z0 = 0.
initial = (k0, z0)

# set up coefficient lists
######################## GSSA Coefs ###############################################

# get a time zero prediction
params3 = np.array([alpha, beta, gamma, delta, chi, theta, tau, rho_z, 0.])

kpred, ellpred, zpred, Ypred, wpred, rpred, Tpred, cpred, ipred, upred = \
    PolSim(initial, nobs, ts, coeffs1, XYbar, params3, coeffs2, XYbar2, \
           params3)

# begin Monte Carlos
# specify the number of simulations
nsim = 10000

# run first simulation and store in Monte Carlo matrices
kmc, ellmc, zmc, Ymc, wmc, rmc, Tmc, cmc, imc, umc \
    = PolSim(initial, nobs, ts, coeffs1, XYbar, params, coeffs2, XYbar2, \
             params2)

for i in range(1, nsim):
    # run remaining simulations
    khist, ellhist, zhist, Yhist, whist, rhist, Thist, chist, ihist, uhist = \
        PolSim(initial, nobs, ts, coeffs1, XYbar, params, coeffs2, XYbar2, \
               params2)
    # stack results in Monte Carlo matrices
    kmc = np.vstack((kmc, khist))
    ellmc = np.vstack((ellmc, ellhist))
    zmc = np.vstack((zmc, zhist))
    Ymc = np.vstack((Ymc, Yhist))
    wmc = np.vstack((wmc, whist))
    rmc = np.vstack((rmc, rhist))
    Tmc = np.vstack((Tmc, Thist))
    cmc = np.vstack((cmc, chist))
    imc = np.vstack((imc, ihist))
    umc = np.vstack((umc, uhist))
    
# now sort the Monte Carlo matrices over the rows
kmc = np.sort(kmc, axis = 0)
ellmc = np.sort(ellmc, axis = 0)
zmc = np.sort(zmc, axis = 0)
Ymc = np.sort(Ymc, axis = 0)
wmc = np.sort(wmc, axis = 0)
rmc = np.sort(rmc, axis = 0)
Tmc = np.sort(Tmc, axis = 0)
cmc = np.sort(cmc, axis = 0)
imc = np.sort(imc, axis = 0)
umc = np.sort(umc, axis = 0)

# find the average values for each variable in each time period across 
# Monte Carlos
kavg = np.mean(kmc, axis = 0)
ellavg = np.mean(ellmc, axis = 0)
zavg = np.mean(zmc, axis = 0)
Yavg = np.mean(Ymc, axis = 0)
wavg = np.mean(wmc, axis = 0)
ravg = np.mean(rmc, axis = 0)
Tavg = np.mean(Tmc, axis = 0)
cavg = np.mean(cmc, axis = 0)
iavg = np.mean(imc, axis = 0)
uavg = np.mean(umc, axis = 0)

# find the rows for desired confidence bands
conf = .1
low = int(np.floor((conf/2)*nsim))
high = nsim - low

# find the upper and lower confidence bands for each variable
kupp = kmc[high,:]
ellupp = ellmc[high,:]
zupp = zmc[high,:]
Yupp = Ymc[high,:]
wupp = wmc[high,:]
rupp = rmc[high,:]
Tupp = Tmc[high,:]
cupp = cmc[high,:]
iupp = imc[high,:]
uupp = umc[high,:]

klow = kmc[low,:]
elllow = ellmc[low,:]
zlow = zmc[low,:]
Ylow = Ymc[low,:]
wlow = wmc[low,:]
rlow = rmc[low,:]
Tlow = Tmc[low,:]
clow = cmc[low,:]
ilow = imc[low,:]
ulow = umc[low,:]

# create a list of time series to plot
data = (kpred/kbar, kupp/kbar, klow/kbar, khist/kbar, \
        ellpred/ellbar, ellupp/ellbar, elllow/ellbar, ellhist/ellbar, \
        zpred, zupp, zlow, zhist, \
        Ypred/Ybar, Yupp/Ybar, Ylow/Ybar, Yhist/Ybar, \
        wpred/wbar, wupp/wbar, wlow/wbar, whist/wbar, \
        rpred/rbar, rupp/rbar, rlow/rbar, rhist/rbar, \
        Tpred/Tbar, Tupp/Tbar, Tlow/Tbar, Thist/Tbar, \
        cpred/cbar, cupp/cbar, clow/cbar, chist/cbar, \
        ipred/ibar, iupp/ibar, ilow/ibar, ihist/ibar, \
        upred/ubar, uupp/ubar, ulow/ubar, uhist/ubar)

# plot using Simple ILA Model Plot.py
from ILAplots import ILAplots
ILAplots(data, name)

## save results in pickle file
#import pickle as pkl
#
#output = open(name + '.pkl', 'wb')
#
#polsimpars = (initial, nobs, ts, coeffs1, XYbar, params, coeffs2, XYbar2, \
#             params2)
#pkl.dump(polsimpars, output)
#
#mcdata = (kmc, ellmc, zmc, Ymc, wmc, rmc, Tmc, cmc, imc, umc)
#pkl.dump(mcdata, output)
#
#output.close()
