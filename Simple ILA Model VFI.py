#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 16:08:19 2017

@author: klp4
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle as pkl

# import the modules from LinApp
from LinApp_FindSS import LinApp_FindSS

from Simple_ILA_Model_Funcs import Modeldefs, Modeldyn


# set name for external files written
name = 'ILAVFI'

# set parameter values
alpha = .35
beta = .99
gamma = 2.5
delta = .08
chi = 10.
theta = 2.
tau = .05   # the 1st stochastic shock
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


from rouwen import rouwen

# set up Markov approximation of AR(1) process using Rouwenhorst method
spread = 5.  # number of standard deviations above and below 0
znpts = 11
zstep = 4.*spread*sigma_z/(znpts-1)

# Markov transition probabilities, current z in cols, next z in rows
Pimat, zgrid = rouwen(rho_z, 0., zstep, znpts)

# discretize k
klow = .6*kbar
khigh = 1.4*kbar
knpts = 11
kgrid = np.linspace(klow, khigh, num = knpts)

# discretize ell
elllow = ellbar - .4
ellhigh = ellbar + .4
ellnpts = 11
ellgrid = np.linspace(elllow, ellhigh, num = ellnpts)

readVF = False

# initialize VF and PF
if readVF:
    infile = open('ILAVFI.pkl', 'rb')
    Vf1 = pkl.load(infile)
    infile.close()
else:
    Vf1 = np.ones((knpts, znpts)) * (-100)

Vf1new = np.zeros((knpts, znpts))
Pf1 = np.zeros((knpts, znpts))
Jf1 = np.zeros((knpts, znpts))

# set VF iteration parameters
#ccrit = 1.0E-20
ccrit = 1.0E-01
count = 0
dist = 100.
maxwhile = 4000  #2432?

# run the program to get the value function (VF1)
nconv = True 
while (nconv):
    count = count + 1
    if count > maxwhile:
        break
    for i1 in range(0, knpts): # over kt
        for i2 in range(0, znpts): # over zt, searching the value for the stochastic shock
            maxval = -100000000000
            for i3 in range(0, knpts): # over k_t+1
                for i4 in range(0, knpts): # over ell_t
                    Y, w, r, T, c, i, u = Modeldefs(kgrid[i3], kgrid[i1], \
                        ellgrid[i4], zgrid[i2], params)
                    temp = u
                    for i5 in range(0, znpts): # over z_t+1
                        temp = temp + beta * Vf1[i3,i5] * Pimat[i2,i5]
                    # print i, j, temp (keep all of them)
                    if np.iscomplex(temp):
                        temp = -1000000000
                    if np.isnan(temp):
                        temp = -1000000000
                    if temp > maxval:
                        maxval = temp
                        Vf1new[i1, i2] = temp
                        Pf1[i1, i2] = kgrid[i3]
                        Jf1[i1, i2] = ellgrid[i4]

    # calculate the new distance measure, we use maximum absolute difference
    dist = np.amax(np.abs(Vf1 - Vf1new))
    if dist < ccrit:
        nconv = False
    # report the results of the current iteration
    print ('iteration: ', count, 'distance: ', dist)
    
    # replace the value function with the new one
    Vf1 = 1.0*Vf1new

print ('Converged after', count, 'iterations') 
print ('Policy function at (', (knpts-1)/2, ',', (znpts-1)/2, ') should be', \
kgrid[int((knpts-1)/2)], 'and is', Pf1[int((knpts-1)/2), int((znpts-1)/2)])

# generate a history of Z's
nobs = 150
Zhist = np.zeros((nobs,1))
for t in range(1, nobs):
    Zhist[t,0] = rho_z*Zhist[t,0] + sigma_z*np.random.normal(0., 1.)
    
# put SS values and starting values into numpy vectors
XYbar = np.array([kbar, ellbar])
X0 = np.array([kbar])
Y0 = np.array([ellbar])


## CHANGE POLICY (PF1)
# see line 282 - 286 (done)

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

# Solve for new policy function using VFI

# create meshgrid
zmesh, kmesh = np.meshgrid(zgrid, kgrid)

# get PF2 and JF2
# find value function and transition function

# initialize VF2 and PF2
Vf2 = Vf1*1.
# Vf2 = np.ones((knpts, znpts)) * (-100)

    
# discretize k
klow = .6*kbar2
khigh = 1.4*kbar2
kgrid2 = np.linspace(klow, khigh, num = knpts)

# discretize ell
# discretize ell
elllow = ellbar2 - .4
ellhigh = ellbar2 + .4
ellgrid2 = np.linspace(elllow, ellhigh, num = ellnpts)

Vf2new = np.zeros((knpts, znpts))
Pf2 = np.zeros((knpts, znpts))
Jf2 = np.zeros((knpts, znpts))

# set VF iteration parameters
count = 0
dist = 100.

# run the program to get the value function (VF2)
nconv = True
while (nconv):
    count = count + 1
    if count > maxwhile:
        break
    for i1 in range(0, knpts): # over kt
        for i2 in range(0, znpts): # over zt, searching the value for the stochastic shock
            maxval = -100000000000
            for i3 in range(0, knpts): # over k_t+1
                for i4 in range(0, knpts): # over ell_t
                    Y, w, r, T, c, i, u = Modeldefs(kgrid2[i3], kgrid2[i1], \
                        ellgrid2[i4], zgrid[i2], params2)
                    temp = u
                    for i5 in range(0, znpts): # over z_t+1
                        temp = temp + beta * Vf2[i3,i5] * Pimat[i2,i5]
                    # print i, j, temp (keep all of them)
                    if np.iscomplex(temp):
                        temp = -1000000000
                    if np.isnan(temp):
                        temp = -1000000000
                    if temp > maxval:
                        maxval = temp
                        Vf2new[i1, i2] = temp
                        Pf2[i1, i2] = kgrid2[i3]
                        Jf2[i1, i2] = ellgrid2[i4]

    # calculate the new distance measure, we use maximum absolute difference
    dist = np.amax(np.abs(Vf2 - Vf2new))
    if dist < ccrit:
        nconv = False
    # report the results of the current iteration
    print ('iteration: ', count, 'distance: ', dist)
    
    # replace the value function with the new one
    Vf2 = 1.*Vf2new

print ('Converged after', count, 'iterations')
print ('Policy function at (', (knpts-1)/2, ',', (znpts-1)/2, ') should be', \
    kgrid2[int((knpts-1)/2)], 'and is', Pf2[int((knpts-1)/2), int((znpts-1)/2)])

Pfdiff = Pf1 - Pf2
Jfdiff = Jf1 - Jf2

# fit PF1 and PF2, Jf1 and JF2 with polynomials

# create meshgrid
kmesh, zmesh = np.meshgrid(kgrid, zgrid)

# create independent variables matrix (X)
X = np.ones(knpts*znpts)

temp = kmesh.flatten()
X = np.vstack((X,temp))

temp = kmesh**2
temp = temp.flatten()
X = np.vstack((X,temp))

temp = kmesh**3
temp = temp.flatten()
X = np.vstack((X,temp))

temp = zmesh.flatten()
X = np.vstack((X,temp))

temp = zmesh**2
temp = temp.flatten()
X = np.vstack((X,temp))

temp = zmesh**3
temp = temp.flatten()
X = np.vstack((X,temp))

temp = kmesh*zmesh
temp = temp.flatten()
X = np.vstack((X,temp))

temp = kmesh**2*zmesh
temp = temp.flatten()
X = np.vstack((X,temp))

temp = kmesh*zmesh**2
temp = temp.flatten()
X = np.vstack((X,temp))

# create 4 different dependent variables matrices (y's)
YPF1 = Pf1.flatten()
YJF1 = Jf1.flatten()
YPF2 = Pf2.flatten()
YJF2 = Jf2.flatten()


coeffsPF1 = np.dot(np.linalg.inv(np.dot(X,np.transpose(X))),np.dot(X,YPF1))
coeffsPF1 = coeffsPF1.reshape((10,1))

coeffsJF1 = np.dot(np.linalg.inv(np.dot(X,np.transpose(X))),np.dot(X,YJF1))
coeffsJF1 = coeffsJF1.reshape((10,1))

coeffsPF2 = np.dot(np.linalg.inv(np.dot(X,np.transpose(X))),np.dot(X,YPF2))
coeffsPF2 = coeffsPF2.reshape((10,1))

coeffsJF2 = np.dot(np.linalg.inv(np.dot(X,np.transpose(X))),np.dot(X,YJF2))
coeffsJF2 = coeffsJF2.reshape((10,1))


# save grids and polynomials
output = open("ILAVFI.pkl", "wb")
pkl.dump(Vf1, output)
pkl.dump(Vf2, output)
pkl.dump(Pf1, output)
pkl.dump(Pf2, output)
pkl.dump(Jf1, output)
pkl.dump(Jf2, output)
pkl.dump(coeffsPF1, output)
pkl.dump(coeffsPF2, output)
pkl.dump(coeffsJF1, output)
pkl.dump(coeffsJF2, output)
output.close()

def PolSim(initial, nobs, ts, coeffsPF1, coeffsJF1, state1, params1, \
           coeffsPF2, coeffsJF2, state2, \
           params2):
    
    '''
    Generates a history of k & ell with a switch in regime in period ts.
    
    Inputs
    -----------    
    initial: list of values for k & z (k0, z0) in the first period.
    nobs: number of periods to simulate.
    ts: period in which the shift occurs.
    PF1: the 1st policy function with the tax rate = 0.05 
    JF1: the 1st jump function with the tax rate = 0.05
    state1: numpy array of XYbar under the baseline regime.
    params1: list of parameters under the baseline regime.
    PF2: the 2nd policy function with the tax rate = 0.055 
    JF2: the 2nd jump function with the tax rate = 0.055
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
    zhist = np.zeros(nobs+1)
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
    zhist[0] = 0.
    
    # unpack state1 and state2
    (kbar, ellbar) = XYbar
    (kbar2, ellbar2) = XYbar2
    
    # generate history of random shocks
    for t in range(0, nobs):
        zhist[t+1] = rho_z*zhist[t] + sigma_z*np.random.normal(0., 1.)
        Xvec = np.array([[1.0], [khist[t]], [khist[t]**2], [khist[t]**3], \
                         [zhist[t]], [zhist[t]**2], [zhist[t]**3], \
                         [khist[t]*zhist[t]], [khist[t]**2*zhist[t]], \
                         [khist[t]*zhist[t]**2]])  
        if t < ts:
            khist[t+1] = np.vdot(Xvec, coeffsPF1)
            ellhist[t] = np.vdot(Xvec, coeffsJF1)
            Yhist[t], whist[t], rhist[t], Thist[t], chist[t], ihist[t], uhist[t] \
                = Modeldefs(khist[t+1], khist[t], ellhist[t], zhist[t], params1)
        else:
            khist[t+1] = np.vdot(Xvec, coeffsPF2)
            ellhist[t] = np.vdot(Xvec, coeffsJF2)
            Yhist[t], whist[t], rhist[t], Thist[t], chist[t], ihist[t], uhist[t] \
                = Modeldefs(khist[t+1], khist[t], ellhist[t], zhist[t], params2)
            
        
        
    return khist, ellhist, zhist, Yhist, whist, rhist, Thist, chist, ihist, \
        uhist
        
        
# parameters with zero variance for shocks
params3 = np.array([alpha, beta, gamma, delta, chi, theta, tau, rho_z, 0.])
params4 = np.array([alpha, beta, gamma, delta, chi, theta, tau2, rho_z, 0.])

# specify the number of observations per simulation
nobs = 1000

# specify the period policy shifts
ts = nobs

# specify initial values
k0 = kbar
z0 = 0.
initial = (k0, z0)

# find actual steady state for baseline
# simulate with zero shocks and see what k converges to in last period
kpred, ellpred, zpred, Ypred, wpred, rpred, Tpred, cpred, ipred, upred = \
    PolSim(initial, nobs, ts, coeffsPF1, coeffsJF1, XYbar, params3, \
           coeffsPF2, coeffsJF2, XYbar2, params3)

# find actual (uncertainty) steady state values for baseline
kact = kpred[nobs-1]
ellact = ellpred[nobs-1]
Yact, wact, ract, Tact, cact, iact, uact = \
    Modeldefs(kact, kact, ellact, 0., params)

# reset nobs and ts
ts = 20
nobs = 120
    
# respecify initial values
k0 = kact
z0 = 0.
initial = (k0, z0)

# get a time zero prediction
kpred, ellpred, zpred, Ypred, wpred, rpred, Tpred, cpred, ipred, upred = \
    PolSim(initial, nobs, ts, coeffsPF1, coeffsJF1, XYbar, params3, \
           coeffsPF2, coeffsJF2, XYbar2, params4)


# begin Monte Carlos
# specify the number of simulations
nsim = 100

# run first simulation and store in Monte Carlo matrices
kmc, ellmc, zmc, Ymc, wmc, rmc, Tmc, cmc, imc, umc \
    = PolSim(initial, nobs, ts, coeffsPF1, coeffsJF1, XYbar, params, \
      coeffsPF2, coeffsJF2, XYbar2, params2)

for i in range(1, nsim):
    # run remaining simulations
    khist, ellhist, zhist, Yhist, whist, rhist, Thist, chist, ihist, uhist = \
        PolSim(initial, nobs, ts, coeffsPF1, coeffsJF1, XYbar, params, \
        coeffsPF2, coeffsJF2, XYbar2, params2)
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


'''
# find the predicted path with no randomness
# run first simulation and store in Monte Carlo matrices
params3 = np.array([alpha, beta, gamma, delta, chi, theta, tau, rho_z, 0])
params4 = np.array([alpha, beta, gamma, delta, chi, theta, tau2, rho_z, 0.])

kpred, ellpred, zpred, Ypred, wpred, rpred, Tpred, cpred, ipred, upred \
    = PolSim(initial, nobs, ts, PF1, JF1, XYbar, params3, PF2, JF2, XYbar2, \
           params4)
'''

# plot predicted with upper and lower bounds

# create a list of time series to plot
data = (kavg/kbar, kupp/kbar, klow/kbar, khist/kbar, \
        ellavg/ellbar, ellupp/ellbar, elllow/ellbar, ellhist/ellbar, \
        zavg, zupp, zlow, zhist, \
        Yavg/Ybar, Yupp/Ybar, Ylow/Ybar, Yhist/Ybar, \
        wavg/wbar, wupp/wbar, wlow/wbar, whist/wbar, \
        ravg/rbar, rupp/rbar, rlow/rbar, rhist/rbar, \
        Tavg/Tbar, Tupp/Tbar, Tlow/Tbar, Thist/Tbar, \
        cavg/cbar, cupp/cbar, clow/cbar, chist/cbar, \
        iavg/ibar, iupp/ibar, ilow/ibar, ihist/ibar, \
        uavg/ubar, uupp/ubar, ulow/ubar, uhist/ubar)


# create a list of time series to plot
data = (kpred/kact, kupp/kact, klow/kact, khist/kact, \
        ellpred/ellact, ellupp/ellact, elllow/ellact, ellhist/ellact, \
        zpred, zupp, zlow, zhist, \
        Ypred/Yact, Yupp/Yact, Ylow/Yact, Yhist/Yact, \
        wpred/wact, wupp/wact, wlow/wact, whist/wact, \
        rpred/ract, rupp/ract, rlow/ract, rhist/ract, \
        Tpred/Tact, Tupp/Tact, Tlow/Tact, Thist/Tact, \
        cpred/cact, cupp/cact, clow/cact, chist/cact, \
        ipred/iact, iupp/iact, ilow/iact, ihist/iact, \
        upred/uact, uupp/uact, ulow/uact, uhist/uact)

# plot using Simple ILA Model Plot.py
from ILAplots import ILAplots
ILAplots(data, name)


'''
## Additional Work: plot grid approximation of policy functions and jump functions
# plot grid approximation of PF1
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(kmesh, zmesh, Pf1)
ax.view_init(30, 150)
plt.title('PF1 Grid')
plt.xlabel('k(t)')
plt.ylabel('z(t)')
plt.show()
plt.savefig('PF1 Grid.png')

# plot grid approximation of PF2
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(kmesh, zmesh, Pf2)
ax.view_init(30, 150)
plt.title('PF2 Grid')
plt.xlabel('k(t)')
plt.ylabel('z(t)')
plt.show()
plt.savefig('PF2 Grid.png')

# plot grid approximation of JF1
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(kmesh, zmesh, Jf1)
ax.view_init(30, 150)
plt.title('JF1 Grid')
plt.xlabel('k(t)')
plt.ylabel('z(t)')
plt.show()
plt.savefig('JF1 Grid.png')

# plot grid approximation of JF2
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(kmesh, zmesh, Jf2)
ax.view_init(30, 150)
plt.title('JF2 Grid')
plt.xlabel('k(t)')
plt.ylabel('z(t)')
plt.show()
plt.savefig('JF2 Grid.png')



## Get the polynomial approximations

PF1approx = 0.*Pf1
PF2approx = 0.*Pf2
JF1approx = 0.*Jf1
JF2approx = 0.*Jf2

for i in range(0,knpts):
    for j in range(0,znpts):
        temp = np.array([[1.0], [kmesh[i,j]], [kmesh[i,j]**2], \
                     [kmesh[i,j]**3], [zmesh[i,j]], [zmesh[i,j]**2], \
                     [zmesh[i,j]**3], [kmesh[i,j]*zmesh[i,j]], \
                     [zmesh[i,j]*kmesh[i,j]**2], [kmesh[i,j]*zmesh[i,j]**2]])
        PF1approx[i,j] = np.dot(np.transpose(coeffsPF1), temp)
        PF2approx[i,j] = np.dot(np.transpose(coeffsPF2), temp)
        JF1approx[i,j] = np.dot(np.transpose(coeffsJF1), temp)
        JF2approx[i,j] = np.dot(np.transpose(coeffsJF2), temp)
    
# plot polynomial approximations
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(kmesh, zmesh, PF1approx)
ax.view_init(30, 150)
plt.title('PF1 polynomial')
plt.xlabel('k(t)')
plt.ylabel('z(t)')
plt.show()
plt.savefig('PF1 Polynomial.png')


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(kmesh, zmesh, PF2approx)
ax.view_init(30, 150)
plt.title('PF2 polynomial')
plt.xlabel('k(t)')
plt.ylabel('z(t)')
plt.show()
plt.savefig('PF2 Polynomial.png')


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(kmesh, zmesh, JF1approx)
ax.view_init(30, 150)
plt.title('JF1 polynomial')
plt.xlabel('k(t)')
plt.ylabel('z(t)')
plt.show()
plt.savefig('JF1 Polynomial.png')


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(kmesh, zmesh, JF2approx)
ax.view_init(30, 150)
plt.title('JF2 polynomial')
plt.xlabel('k(t)')
plt.ylabel('z(t)')
plt.show()
plt.savefig('JF2 Polynomial.png')
'''