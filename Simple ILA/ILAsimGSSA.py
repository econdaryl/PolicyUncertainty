# -*- coding: utf-8 -*-
'''
Created on Mon Nov  6 15:02:52 2017

@author: Daryl Larsen

Run a series of monte carlo simulations and analyzes them using GSSA
'''

import numpy as np
import pickle as pkl
import timeit
from ILArunmc import runmc
from ILAmcanalysis import mcanalysis

name = 'ILAsimGSSA'

def generateGSSA(k, z, args):
   
    '''
    This function generates values of k next period and ell this period given
    values for k and z this period.
    
    Inputs
    k - k this period
    z - z this period
    args - coeffs, pord, nx, ny, nz
    
    Outputs
    kp - k next period
    ell - ell this period
    '''
    # unpack args
    (coeffs, pord, nx, ny, nz) = args
    nX = nx+nz
    
    A = np.exp(z)
    Xin = np.append(k, A)
    Xvec = np.ones((1,1))
    for i in range(1, pord+1):
        Xvec = np.append(Xvec, Xin**i)
    for i in range (0, nX):
        for j in range(i+1, nX):
            temp = Xin[i]*Xin[j]
            Xvec = np.append(Xvec, temp)
    XYout = np.dot(Xvec, coeffs)
    kp = XYout[0:nx]
    ell = XYout[nx:nx+ny]
    if ell > 0.9999:
        ell = np.array([0.9999])
    elif ell < 0.0001:
        ell = np.array([0.0001])
    return kp, ell


# -----------------------------------------------------------------------------
# LOAD VALUES FROM SS AND GSSA
    
# load steady state values and parameters
infile = open('ILAfindss.pkl', 'rb')
(bar1, bar2, params1, params2, GSSAparams) = pkl.load(infile)
infile.close()

# unpack
[kbar1, ellbar1, Ybar1, wbar1, rbar1, Tbar1, cbar1, ibar1, ubar1] = bar1
[kbar2, ellbar2, Ybar2, wbar2, rbar2, Tbar2, cbar2, ibar2, ubar2] = bar2
[alpha, beta, gamma, delta, chi, theta, tau, rho_z, sigma_z] = params1
tau2 = params2[6]
(zbar, Zbar, NN, nx, ny, nz, logX, Sylv) = GSSAparams
pord = 2
# load GSSA coeffs
infile = open('ILAsolveGSSA.pkl', 'rb')
(coeffs1, coeffs2, timesolve) = pkl.load(infile)
infile.close()

# create args lists
XYbar1 = (kbar1, ellbar1)
XYbar2 = (kbar2, ellbar2)
args1 = (coeffs1, pord, nx, ny, nz)
args2 = (coeffs2, pord, nx, ny, nz)

# -----------------------------------------------------------------------------
# RUN MONTE CARLOS

# start clock for all MCs
startsim = timeit.default_timer()

# specify the number of observations per simulation
nobs = 120
# specify the period policy shifts
ts = 20
# specify the number of simulations
nsim = 100000
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
predargs = (initial, nobs, ts, generateGSSA, args1, args2, params3, params4)

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
bardata = (kbar1, ellbar1, zbar, Ybar1, wbar1, rbar1, Tbar1, cbar1, ibar1, 
           ubar1)
  
avgdata, uppdata, lowdata = \
    mcanalysis(mcdata, preddata, bardata, histdata, name, nsim)
    
# unpack
(kavg, ellavg, zavg, Yavg, wavg, ravg, Tavg, cavg, iavg, uavg, foremeanavg, \
     forevaravg, zformeanavg, zforvaravg, RMsqEerravg) = avgdata
(kupp, ellupp, zupp, Yupp, wupp, rupp, Tupp, cupp, iupp, uupp, foremeanupp, \
     forevarupp, zformeanupp, zforvarup) = uppdata
(klow, elllow, zlow, Ylow, wlow, rlow, Tlow, clow, ilow, ulow, foremeanlow, \
     forevarlow, zformeanlow, zformvarlow) = lowdata
    
foreperc = np.delete(foremeanavg, 2, 0)/np.abs(bar1)
print('1-period-ahead average forecast errors')
print(foreperc)

forevarc = (np.delete(forevaravg, 2, 0))**.5/np.abs(bar1)
print('1-period-ahead RMSE forecast errors')
print(forevarc)

zforperc = np.delete(zformeanavg, 2, 0)/np.abs(bar1)
print('period-0 average forecast errors')
print(zforperc)

zforvarc = (np.delete(zformeanavg, 2, 0))**.5/np.abs(bar1)
print('period-0 RMSE forecast errors')
print(zforvarc)

print('root mean squared Euler errors')
print(RMsqEerravg)

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