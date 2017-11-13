#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This program reads in paramter values and steady states from the file, 
ILAfindss.pkl.

It then calculates the value function coefficients for the policy and jump function
approximations using the VFI method.

The coefficients and time to solve are written to the file, ILAsolveVFI.pkl.

The baseline values have a 1 at the end of the variable name.
The values after the policy change have a 2 at the end. 
"""

import numpy as np
import timeit
import pickle as pkl

from Simple_ILA_Model_Funcs import Modeldyn
from Simple_ILA_Model_Funcs import Modeldefs

# import the modules from LinApp
from LinApp_FindSS import LinApp_FindSS

# -----------------------------------------------------------------------------
# READ IN VALUES FROM STEADY STATE CALCULATIONS

# load steady state values and parameters
infile = open('ILAfindss.pkl', 'rb')
(bar1, bar2, params1, params2, VFIparams) = pkl.load(infile)
infile.close()

# unpack
[kbar1, ellbar1, Ybar1, wbar1, rbar1, Tbar1, cbar1, ibar1, ubar1] = bar1
[kbar2, ellbar2, Ybar2, wbar2, rbar2, Tbar2, cbar2, ibar2, ubar2] = bar2
[alpha, beta, gamma, delta, chi, theta, tau, rho_z, sigma_z] = params1
tau2 = params2[6]
(zbar, Zbar, NN, nx, ny, nz, logX, Sylv) = VFIparams

# set clock for time to calcuate functions
startsolve = timeit.default_timer()

# set name for external files written
name = 'ILAsolveVFI'

# -----------------------------------------------------------------------------
# BASELINE

# set up steady state input vector for baseline
theta1 = np.array([kbar1, kbar1, kbar1, ellbar1, ellbar1, 0., 0.])

# check SS solution
check = Modeldyn(theta1, params1)
print ('check SS: ', check)
if np.max(np.abs(check)) > 1.E-6:
    print ('Have NOT found steady state')
    
# find the steady state values for the definitions
Ybar, wbar, rbar, Tbar, cbar, ibar, ubar = \
    Modeldefs(kbar1, kbar1, ellbar1, 0., params1)

# display all steady state values
print ('kbar:   ', kbar1)
print ('ellbar: ', ellbar1)
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
klow = .6*kbar1
khigh = 1.4*kbar1
knpts = 11
kgrid = np.linspace(klow, khigh, num = knpts)

# discretize ell
elllow = ellbar1 - .4
ellhigh = ellbar1 + .4
ellnpts = 11
ellgrid = np.linspace(elllow, ellhigh, num = ellnpts)

readVF = True

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
ccrit = 1.0E-01
count = 0
dist = 100.
maxwhile = 4000

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
                        ellgrid[i4], zgrid[i2], params1)
                    temp = u
                    for i5 in range(0, znpts): # over z_t+1
                        temp = temp + beta * Vf1[i3,i5] * Pimat[i5,i2]
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
XYbar = np.array([kbar1, ellbar1])
X0 = np.array([kbar1])
Y0 = np.array([ellbar1])



# -----------------------------------------------------------------------------
# CHANGE POLICY

# set new tax rate
 # already set at the begining

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
                        temp = temp + beta * Vf2[i3,i5] * Pimat[i5,i2]
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


# get OLS coefficient
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
    
# find the steady state values for the definitions
Ybar2, wbar2, rbar2, Tbar2, cbar2, ibar2, ubar2 = \
    Modeldefs(kbar2, kbar2, ellbar2, 0., params2)


# set up coefficient list before the policy change
coeffs1 = (Vf1, Pf1, Jf1, coeffsPF1, coeffsJF1)

# set up coefficient list after the policy change
coeffs2 = (Vf2, Pf2, Jf2, coeffsPF2, coeffsJF2)

# calculate time to solve for functions
stopsolve = timeit.default_timer()
timesolve =  stopsolve - startsolve
print('time to solve: ', timesolve)


# -----------------------------------------------------------------------------
# SAVE RESULTS

output = open(name + '.pkl', 'wb')

# write timing
pkl.dump((coeffs1, coeffs2, timesolve), output)

output.close()
 