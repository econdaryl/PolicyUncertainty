'''
Run a series of monte carlo simulations and analyzes them using the
linearization method
'''

import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
import timeit

from OLG3runmc import runmc
from OLG3mcanalysis import mcanalysis

name = 'OLG3simVFI_AL'

def generateVFI(k, z, args):
    
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
    (Vf, Pf, Jf, coeffsPF, coeffsJF) = coeffs
    
    # inputs must be 1D numpy arrays and deviation from SS values
    Xvec = np.array([[1.0], [k], [k**2], [k**3], [z], [z**2], [z**3], \
                     [k*z], [k**2*z], [k*z**2]])

    kp = np.vdot(Xvec, coeffsPF)
    ell= np.vdot(Xvec, coeffsJF)
    
    return kp, ell

# -----------------------------------------------------------------------------
# LOAD VALUES FROM SS AND VALUE FUNCTION ITERATION
    
# load steady state values and parameters
infile = open('OLG3findss.pkl', 'rb')
(bar1, bar2, params1, params2, VFIparams) = pkl.load(infile)
infile.close()

# load VFI coeffs
infile = open('OLG3solveVFI_11_Anthonylaptop.pkl', 'rb')
(coeffs1, coeffs2, timesolve) = pkl.load(infile)
infile.close()

# unpack
[k2bar1, k3bar1, l1bar1, l2bar1, Kbar1, \
    Lbar1, GDPbar1, wbar1, rbar1, T4bar1, Bbar1, c1bar1, c2bar1, c3bar1, \
    Cbar1, Ibar1, u1bar1, u2bar1, u3bar1] = bar1
[k2bar2, k3bar2, l1bar2, l2bar2, Kbar2, \
    Lbar2, GDPbar2, wbar2, rbar2, T4bar2, Bbar2, c1bar2, c2bar2, c3bar2, \
    Cbar2, Ibar2, u1bar2, u2bar2, u3bar2] = bar2
[alpha, beta, gamma, delta, chi, theta, tau, rho_z, \
    sigma_z, pi2, pi3, f1, f2, nx, ny, nz] = params1
tau2 = params2[6]
(zbar, Zbar, NN, nx, ny, nz, logX, Sylv) = VFIparams

# create args lists    (not sure about this part)
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
predargs = (initial, nobs, ts, generateVFI, args1, args2, params3, params4)

# get list of arguments for monte carlos simulations 
simargs = (initial, nobs, ts, generateVFI, args1, args2, params1, params2)

# run the Monte Carlos
mcdata, histdata, preddata, act = runmc(simargs, nsim, nobs, repincr)

# calculate time to simulate all MCs'/
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