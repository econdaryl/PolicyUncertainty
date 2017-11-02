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

from gssa import GSSA, XYfunc

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
pord = 3

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
    Zhist[t,0] = rho_z*Zhist[t-1,0] + sigma_z*np.random.normal(0., 1.)
    
coeffs1 = GSSA(params, Zhist, nobs)
    
# put SS values and starting values into numpy vectors
XYbar = np.array([kbar, ellbar])
X0 = np.array([kbar])
Y0 = np.array([ellbar])


## CHANGE POLICY

# set new tax rate
tau2 = .075

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
coeffs2 = GSSA(params2, Zhist, nobs)

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
    ellhist = np.zeros(nobs+1)
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
    ellfhist = np.zeros(nobs+2)
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
    khist[0] = k0
    zhist[0] = z0
    
    # unpack state1 and state2
    (kbar, ellbar) = XYbar
    (kbar2, ellbar2) = XYbar2
    
    # generate history of random shocks
    for t in range(1, nobs):
        zhist[t] = rho_z*zhist[t-1] + sigma_z*np.random.normal(0., 1.)
    
    pord = 3
    regtype = 'poly1' # functional form for X & Y functions 
    fittype = 'MVOLS'   # regression fitting method
    #pord = 3  # order of polynomial for fitting function
    ccrit = 1.0E-8  # convergence criteria for XY change
    damp = 0.01  # damping paramter for fixed point algorithm
    XYparams = (regtype, fittype, pord, nx, ny, nz, ccrit, damp)
    
    # generate histories for k and ell for the first ts-1 periods
    khist[0], ellhist[0] = XYfunc(k0, z0, XYparams, coeffs1)
    for t in range(0, ts-1):
        # inputs must be 1D numpy arrays and deviation from SS values
        # USE GSSA polynomials ######################################################
        khist[t+1], ellhist[t+1] = XYfunc(khist[t], zhist[t], XYparams, coeffs1)
        Yhist[t], whist[t], rhist[t], Thist[t], chist[t], ihist[t], \
            uhist[t] = \
            Modeldefs(khist[t+1], khist[t], ellhist[t], zhist[t], params)
        # Get 1-period ahead forecasts
        zfhist[t+1] = rho_z*zhist[t]
        kin = np.array([khist[t+1]])
        kfhist[t+2], ellfhist[t+2] = XYfunc(kin, zfhist[t+1], XYparams, coeffs1)
    for t in range(1, ts-1):
        Yfhist[t+1], wfhist[t+1], rfhist[t+1], Tfhist[t+1], cfhist[t+1], ifhist[t+1], \
            ufhist[t+1] = \
            Modeldefs(kfhist[t+2], kfhist[t+1], ellfhist[t+1], zfhist[t+1], params)

    for t in range(ts-1, nobs):
        # USE GSSA polynomials ######################################################
        khist[t+1], ellhist[t+1] = XYfunc(khist[t], zhist[t], XYparams, coeffs2)
        Yhist[t], whist[t], rhist[t], Thist[t], chist[t], ihist[t], \
            uhist[t] = \
            Modeldefs(khist[t+1], khist[t], ellhist[t], zhist[t], params2)
        # Get 1-period ahead forecasts
        zfhist[t+1] = rho_z*zhist[t]
        kin = np.array([khist[t+1]])
        kfhist[t+2], ellfhist[t+2] = XYfunc(kin, zfhist[t+1], XYparams, coeffs2)
    for t in range(ts, nobs):
        Yfhist[t+1], wfhist[t+1], rfhist[t+1], Tfhist[t+1], cfhist[t+1], ifhist[t+1], \
            ufhist[t+1] = \
            Modeldefs(kfhist[t+2], kfhist[t+1], ellfhist[t+1], zfhist[t+1], params2)
    
    return khist, ellhist, zhist, Yhist, whist, rhist, Thist, chist, ihist, \
        uhist, kfhist, ellfhist, zfhist, Yfhist, wfhist, rfhist, Tfhist, \
        cfhist, ifhist, ufhist


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

params4 = np.array([alpha, beta, gamma, delta, chi, theta, tau2, rho_z, 0.])

kpred, ellpred, zpred, Ypred, wpred, rpred, Tpred, cpred, ipred, upred, \
kf, ellf, zf, Yf, wf, rf, Tf, cf, invf, uf \
    = PolSim(initial, nobs, ts, coeffs1, XYbar, params3, coeffs2, XYbar2, \
           params4)

# begin Monte Carlos
# specify the number of simulations
nsim = 10000

# run first simulation and store in Monte Carlo matrices
kmc, ellmc, zmc, Ymc, wmc, rmc, Tmc, cmc, imc, umc, \
kfmc, ellfmc, zfmc, Yfmc, wfmc, rfmc, Tfmc, cfmc, ifmc, ufmc \
    = PolSim(initial, nobs, ts, coeffs1, XYbar, params, coeffs2, XYbar2, \
             params2)
# replace forecast with abs value of forecast error
for t in range(2, nobs):
    kfmc[t] = np.abs(kfmc[t] - kmc[t])
    ellfmc[t] = np.abs(ellfmc[t] - ellmc[t])
    zfmc[t] = np.abs(zfmc[t] - zmc[t])
    Yfmc[t] = np.abs(Yfmc[t] - Ymc[t])
    wfmc[t] = np.abs(wfmc[t] - wmc[t])
    rfmc[t] = np.abs(rfmc[t] - rmc[t])
    Tfmc[t] = np.abs(Tfmc[t] - Tmc[t])
    cfmc[t] = np.abs(cfmc[t] - cmc[t])
    ifmc[t] = np.abs(ifmc[t] - imc[t])
    ufmc[t] = np.abs(ufmc[t] - umc[t])

# caclulate mean forecast errors
foremeanmc = np.array([np.mean(kfmc[1:nobs]),
                       np.mean(ellfmc[1:nobs]),
                       np.mean(zfmc[1:nobs]), 
                       np.mean(Yfmc[1:nobs]),
                       np.mean(wfmc[1:nobs]), 
                       np.mean(rfmc[1:nobs]),
                       np.mean(Tfmc[1:nobs]), 
                       np.mean(cfmc[1:nobs]),
                       np.mean(ifmc[1:nobs]), 
                       np.mean(ufmc[1:nobs])])

for i in range(1, nsim):
    print('simulation # ', i)
    # run remaining simulations
    khist, ellhist, zhist, Yhist, whist, rhist, Thist, chist, ihist, uhist, \
    kfhist, ellfhist, zfhist, Yfhist, wfhist, rfhist, Tfhist, cfhist, ifhist, ufhist \
        = PolSim(initial, nobs, ts, coeffs1, XYbar, params, coeffs2, XYbar2, \
               params2)
    
    # replace forecast with abs value of forecast error
    for t in range(1, nobs):
        kfhist[t] = np.abs(kfhist[t] - khist[t])
        ellfhist[t] = np.abs(ellfhist[t] - ellhist[t])
        zfhist[t] = np.abs(zfhist[t] - zhist[t])
        Yfhist[t] = np.abs(Yfhist[t] - Yhist[t])
        wfhist[t] = np.abs(wfhist[t] - whist[t])
        rfhist[t] = np.abs(rfhist[t] - rhist[t])
        Tfhist[t] = np.abs(Tfhist[t] - Thist[t])
        cfhist[t] = np.abs(cfhist[t] - chist[t])
        ifhist[t] = np.abs(ifhist[t] - ihist[t])
        ufhist[t] = np.abs(ufhist[t] - uhist[t])
        
        # calculate mean forecast errors
        foremean = np.array([np.mean(kfhist[1:nobs]),
                             np.mean(ellfhist[1:nobs]),
                             np.mean(zfhist[1:nobs]), 
                             np.mean(Yfhist[1:nobs]),
                             np.mean(wfhist[1:nobs]), 
                             np.mean(rfhist[1:nobs]),
                             np.mean(Tfhist[1:nobs]), 
                             np.mean(cfhist[1:nobs]),
                             np.mean(ifhist[1:nobs]), 
                             np.mean(ufhist[1:nobs])]) 
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
    foremeanmc = np.vstack((foremeanmc, foremean))
    
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