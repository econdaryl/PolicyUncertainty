#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 16:08:19 2017

@author: klp4
"""

import numpy as np
import timeit

# import the modules from LinApp
from LinApp_FindSS import LinApp_FindSS
from LinApp_Deriv import LinApp_Deriv
from LinApp_Solve import LinApp_Solve
# from LinApp_SSL import LinApp_SSL

from Simple_ILA_Model_Funcs import Modeldefs, Modeldyn

# set name for external files written
name = 'ILALin'

# set clock for time to calcuate functions
startsolve = timeit.default_timer()

# set parameter values
alpha = .35
beta = .99
gamma = 2.5
delta = .02
chi = 1.5
theta = .33
rho_z = .9
sigma_z = .005
# set old and new tax rates
tau = .05
tau2 = .055

# make parameter list to pass to functions for baseline
params = np.array([alpha, beta, gamma, delta, chi, theta, tau, rho_z, sigma_z])

# set LinApp parameters
zbar = 0.
Zbar = np.array([zbar])
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

# set up steady state input vector for baseline
theta0 = np.array([kbar, kbar, kbar, ellbar, ellbar, 0., 0.])

# check SS solution
check = Modeldyn(theta0, params)
print ('check SS: ', check)
if np.max(np.abs(check)) > 1.E-6:
    print ('Have NOT found steady state')
    
# find the steady state values for the definitions
Ybar, wbar, rbar, Tbar, cbar, ibar, ubar = \
    Modeldefs(kbar, kbar, ellbar, 0., params)

# display all steady state values for baseline
print ('kbar:   ', kbar)
print ('ellbar: ', ellbar)
print ('Ybar:   ', Ybar)
print ('wbar:   ', wbar)
print ('rbar:   ', rbar)
print ('Tbar:   ', Tbar)
print ('cbar:   ', cbar)
print ('ibar:   ', ibar)
print ('ubar:   ', ubar)

# find the derivatives matrices
[AA, BB, CC, DD, FF, GG, HH, JJ, KK, LL, MM, WW, TT] = \
    LinApp_Deriv(Modeldyn, params, theta0, nx, ny, nz, logX)

# set value for NN    
NN = rho_z
    
# find the policy and jump function coefficients
PP, QQ, UU, RR, SS, VV = \
    LinApp_Solve(AA,BB,CC,DD,FF,GG,HH,JJ,KK,LL,MM,WW,TT,NN,Zbar,Sylv)
print ('P: ', PP)
print ('Q: ', QQ)
print ('R: ', RR)
print ('S: ', SS)
    
# put SS values and starting values into numpy vectors
XYbar = np.array([kbar, ellbar])
X0 = np.array([kbar])
Y0 = np.array([ellbar])


# -----------------------------------------------------------------------------
# CHANGE POLICY

# make parameter list to pass to functions for new tax
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

# find the new derivatives matrices
[AA2, BB2, CC2, DD2, FF2, GG2, HH2, JJ2, KK2, LL2, MM2, WW2, TT2] = \
    LinApp_Deriv(Modeldyn, params2, theta02, nx, ny, nz, logX)
    
# find the policy and jump function coefficients
PP2, QQ2, UU2, RR2, SS2, VV2 = \
    LinApp_Solve(AA2,BB2,CC2,DD2,FF2,GG2,HH2,JJ2,KK2,LL2,MM2,WW2,TT2,NN,Zbar, \
                 Sylv)
print ('P: ', PP2)
print ('Q: ', QQ2)
print ('R: ', RR2)
print ('S: ', SS2)

# calculate time to solve for functions
stopsolve = timeit.default_timer()
timesolve =  stopsolve - startsolve


# define a function that runs simulation with shift in tax rates
# -----------------------------------------------------------------------------
def PolSim(args):
    from LinApp_Sim import LinApp_Sim
    
    (initial, nobs, ts, coeffs1, state1, params1, coeffs2, state2, \
           params2) = args
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
    
    # preallocate forecast histories
    kfhist = np.zeros(nobs+1)
    ellfhist = np.zeros(nobs)
    zfhist = np.zeros(nobs)
    Yfhist = np.zeros(nobs)
    wfhist = np.zeros(nobs)
    rfhist = np.zeros(nobs)
    Tfhist = np.zeros(nobs)
    cfhist = np.zeros(nobs)
    ifhist = np.zeros(nobs)
    ufhist = np.zeros(nobs)
    
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
        kin = np.array([khist[t] - kbar])
        zin = np.array([zhist[t]])
        k, ell = LinApp_Sim(kin, zin, PP, QQ, UU, RR, SS, VV)
        # k and ell are deviations from SS values, so add these back.
        # they are also 1D numpy arrays, so pull out the values rather than 
        # use the arrays.
        khist[t+1] = k + kbar
        ellhist[t] = ell + ellbar
        Yhist[t], whist[t], rhist[t], Thist[t], chist[t], ihist[t], \
            uhist[t] = \
            Modeldefs(khist[t+1], khist[t], ellhist[t], zhist[t], params)
        
        # get 1-period ahead forecasts
        zfhist[t+1] = rho_z*zhist[t]
        kin = np.array([khist[t+1] - kbar])
        zin = np.array([zfhist[t]])
        kf, ellf = LinApp_Sim(kin, zin, PP, QQ, UU, RR, SS, VV)
        # k and ell are deviations from SS values, so add these back.
        # they are also 1D numpy arrays, so pull out the values rather than 
        # use the arrays.
        kfhist[t+2] = kf + kbar
        ellfhist[t+1] = ellf + ellbar
        Yfhist[t+1], wfhist[t+1], rfhist[t+1], Tfhist[t+1], cfhist[t+1], \
            ifhist[t], ufhist[t] = Modeldefs(kfhist[t+2], khist[t+1], \
            ellfhist[t+1], zfhist[t+1], params)
        
    
    for t in range(ts-1, nobs):
        kin = np.array([khist[t] - kbar2])
        zin = np.array([zhist[t]])
        k, ell = LinApp_Sim(kin, zin, PP2, QQ2, UU2, RR2, SS2, VV2)
        khist[t+1] = k + kbar2
        ellhist[t] = ell + ellbar2
        Yhist[t], whist[t], rhist[t], Thist[t], chist[t], ihist[t], \
            uhist[t] = \
            Modeldefs(khist[t+1], khist[t], ellhist[t], zhist[t], params2)
        
        if t < nobs - 1:
            # get 1-period ahead forecasts
            zfhist[t+1] = rho_z*zhist[t]
            kin = np.array([khist[t+1] - kbar])
            zin = np.array([zfhist[t]])
            kf, ellf = LinApp_Sim(kin, zin, PP2, QQ2, UU2, RR2, SS2, VV2)
            # k and ell are deviations from SS values, so add these back.
            # they are also 1D numpy arrays, so pull out the values rather than 
            # use the arrays.
            kfhist[t+2] = kf + kbar
            ellfhist[t+1] = ellf + ellbar
            Yfhist[t+1], wfhist[t+1], rfhist[t+1], Tfhist[t+1], cfhist[t+1], \
                ifhist[t], ufhist[t] = Modeldefs(kfhist[t+2], khist[t+1], \
                ellfhist[t+1], zfhist[t+1], params)
    
    
    return khist, ellhist, zhist, Yhist, whist, rhist, Thist, chist, ihist, \
        uhist, kfhist, ellfhist, zfhist, Yfhist, wfhist, rfhist, Tfhist, \
        cfhist, ifhist, ufhist
        
# -----------------------------------------------------------------------------
## RUN SIMULATIONS
from ILArunmc import runmc

# start clock for all MCs
startsim = timeit.default_timer()

# specify the number of observations per simulation
nobs = 120

# specify the period policy shifts
ts = 20

# specify initial values
k0 = kbar
z0 = 0.
initial = (k0, z0)

# set up coefficient lists
coeffs1 = (PP, QQ, UU, RR, SS, VV)
coeffs2 = (PP2, QQ2, UU2, RR2, SS2, VV2)

# parameters for tau1 portion
params3 = np.array([alpha, beta, gamma, delta, chi, theta, tau, rho_z, 0.])
# paramters for tau2 portion
params4 = np.array([alpha, beta, gamma, delta, chi, theta, tau2, rho_z, 0.])
# get list of arguments for predictions simulation
predargs = (initial, nobs, ts, coeffs1, XYbar, params3, coeffs2, XYbar2, \
           params4)
# get time zero prediction
kpred, ellpred, zpred, Ypred, wpred, rpred, Tpred, cpred, ipred, upred,  \
kf, ellf, zf, Yf, wf, rf, Tf, cf, invf, uf = PolSim(predargs)

# get list of arguments for monte carlos simulations 
simargs = (initial, nobs, ts, coeffs1, XYbar, params, coeffs2, XYbar2, \
               params2)

# specify the number of simulations
nsim = 10000

# specify the increment between MC reports
repincr = 100

# run the Monte Carlos
mcdata, histdata = runmc(PolSim, simargs, nsim, nobs, repincr)

# calculate time to simulate all MCs
stopsim = timeit.default_timer()
timesim = stopsim - startsim

preddata = (kpred, ellpred, zpred, Ypred, wpred, rpred, Tpred, cpred, ipred, \
        upred)
bardata = (kbar, ellbar, zbar, Ybar, wbar, rbar, Tbar, cbar, ibar, ubar)

# -----------------------------------------------------------------------------
# calculate and report statistics and charts from Monte Carlos  
from ILAmcanalysis import mcanalysis
avgdata, uppdata, lowdata = \
    mcanalysis(mcdata, preddata, bardata, histdata, name, nsim)

# -----------------------------------------------------------------------------
# save results in pickle file
import pickle as pkl

output = open(name + '.pkl', 'wb')

# write timing
pkl.dump(timesolve, output)
pkl.dump(timesim, output)
print('time to solve for policy functions: ', timesolve)
print('time to simulate', nsim, 'monte carlos: ', timesim)

# write policy simulation paramters
polsimpars = (initial, nobs, ts, coeffs1, XYbar, params, coeffs2, XYbar2, \
             params2, nsim, name)
pkl.dump(polsimpars, output)

# write monte carlo data
alldata = (mcdata, preddata, bardata, histdata, avgdata, uppdata, lowdata)
pkl.dump(alldata, output)

output.close()