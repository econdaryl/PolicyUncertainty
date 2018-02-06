'''
Run a series of monte carlo simulations and analyzes them using the
GSSA method
'''

import numpy as np
import pickle as pkl
import timeit

from BMrunmc import runmc
from BMmcanalysis import mcanalysis
from gssa import poly1

name = 'BMsimGSSA'

def generateGSSA(k, z, args):
    (pord, nx, ny, nz, coeffs) = args
    An = np.exp(z)
    XZin = np.append(k, An)
    XYbasis = np.append(1., XZin)
    for i in range(1, pord):
        XYbasis = poly1(XZin, args)
    Xn = np.dot(XYbasis, coeffs)
    return Xn
# -----------------------------------------------------------------------------
# LOAD VALUES FROM SS AND LINEARIZATION
    
# load steady state values and parameters
infile = open('BMfindss.pkl', 'rb')
(bar1, bar2, params1, params2, LINparams) = pkl.load(infile)
infile.close()

# unpack
[kbar1, Ybar1, wbar1, rbar1, Tbar1, cbar1, ibar1, ubar1] = bar1
[kbar2, Ybar2, wbar2, rbar2, Tbar2, cbar2, ibar2, ubar2] = bar2
[alpha, beta, tau, rho_z, sigma_z] = params1
tau2 = params2[2]
(zbar, Zbar, NN, nx, ny, nz, logX, Sylv) = LINparams
    
# load Linearization coeffs
infile = open('BMsolveGSSA.pkl', 'rb')
(coeffs1, coeffs2, timesolve) = pkl.load(infile)
infile.close()

# create args lists
pord = 3
args1 = (pord, nx, ny, nz, coeffs1)
args2 = (pord, nx, ny, nz, coeffs2)

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
bardata = (kbar1, zbar, Ybar1, wbar1, rbar1, Tbar1, cbar1, ibar1, 
           ubar1)
  
avgdata, uppdata, lowdata = \
    mcanalysis(mcdata, preddata, bardata, histdata, name, nsim)
    
# unpack
(kavg, zavg, Yavg, wavg, ravg, Tavg, cavg, iavg, uavg, foremeanavg, \
     zformeanavg, RMsqEerravg) = avgdata
(kupp, zupp, Yupp, wupp, rupp, Tupp, cupp, iupp, uupp, foremeanupp, \
     zformeanupp) = uppdata
(klow, zlow, Ylow, wlow, rlow, Tlow, clow, ilow, ulow, foremeanlow, \
     zformeanlow) = lowdata
    
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
