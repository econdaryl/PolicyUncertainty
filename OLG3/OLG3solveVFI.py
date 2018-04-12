#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This program reads in paramter values and steady states from the file, 
ILAfindss.pkl.

It then calculates the linear coefficients for the policy and jump function
approximations using the LinApp toolkit.

The coefficients and time to solve are written to the file, ILAsolveVFI.pkl.

The baseline values have a 1 at the end of the variable name.
The values after the policy change have a 2 at the end. 
"""

import numpy as np
import timeit
import pickle as pkl

from OLG3funcs import Modeldefs

def fitfuncs(kgrid2, kgrid3, zgrid, nord, funclist):
    
    # unpack funclist
    (Pf11, Pf12, Jf11, Jf12) = funclist

    # fit PF1 and PF2, Jf1 and JF2 with polynomials
    
    # create meshgrid
    kmesh2, kmesh3, zmesh = np.meshgrid(kgrid2, kgrid3, zgrid)
    
    # create independent variables matrix (X)
    X = np.array([])
    count = 0
    for i in range(0, nord+1):
        for j in range(0, nord+1-i):
            for k in range(0, nord+1-i-j):
                count = count + 1
                temp = kmesh2**(i) * kmesh3**(j) * zmesh**(k)
                temp = temp.flatten()
                ncol = temp.size
                X = np.append(X,temp)
                X = X.reshape((count, ncol))
    
    # create 4 different dependent variables matrices (y's)
    YPf11 = Pf11.flatten()
    YPf12 = Pf12.flatten()
    YJf11 = Jf11.flatten()
    YJf12 = Jf12.flatten()

    # get OLS coefficiens
    coeffsPf11 = np.dot(np.linalg.inv(np.dot(X,np.transpose(X))),np.dot(X,YPf11))
    coeffsPf11 = np.reshape(coeffsPf11,(-1,1))
    
    coeffsPf12 = np.dot(np.linalg.inv(np.dot(X,np.transpose(X))),np.dot(X,YPf12))
    coeffsPf12 = np.reshape(coeffsPf12,(-1,1))
    
    coeffsJf11 = np.dot(np.linalg.inv(np.dot(X,np.transpose(X))),np.dot(X,YJf11))
    coeffsJf11 = np.reshape(coeffsJf11,(-1,1))
    
    coeffsJf12 = np.dot(np.linalg.inv(np.dot(X,np.transpose(X))),np.dot(X,YJf12))
    coeffsJf12 = np.reshape(coeffsJf12,(-1,1))
    
    return coeffsPf11, coeffsPf12, coeffsJf11, coeffsJf12



# -----------------------------------------------------------------------------
# READ IN VALUES FROM STEADY STATE CALCULATIONS

# load steady state values and parameters
infile = open('OLG3findss.pkl', 'rb')
(bar1, bar2, params1, params2, VFIparams) = pkl.load(infile)
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

# set clock for time to calcuate functions
startsolve = timeit.default_timer()

# set name for external files written
name = 'OLG3solveVFI_5'

# -----------------------------------------------------------------------------
# BASELINE

# set up steady state input vector for baseline
theta1 = np.array([k2bar1, k3bar1, k2bar1, k3bar1, k2bar1, \
                   k3bar1, l1bar1, l2bar1, l1bar1, l2bar1, 0., 0.])

from rouwen import rouwen

kfact= .05
elladd = .05

# set up Markov approximation of AR(1) process using Rouwenhorst method
spread = 3.  # number of standard deviations above and below 0
znpts = 3
zstep = 4.*spread*sigma_z/(znpts-1)

# Markov transition probabilities, current z in cols, next z in rows
Pimat, zgrid = rouwen(rho_z, 0., zstep, znpts)

# discretize k
klow = (1-kfact)*k2bar1  # changes
khigh = (1+kfact)*k2bar1 # changes
knpts = 3
kgrid = np.linspace(klow, khigh, num = knpts)

# discretize ell
elllow = l1bar1 - elladd  # what about l2bar1 and l3bar1?
ellhigh = l1bar1 + elladd # what about l2bar1 and l3bar1?
ellnpts = 3
ellgrid = np.linspace(elllow, ellhigh, num = ellnpts)

readVF = True

# initialize VF and PF
if readVF:    ## Need to change the code because of epanded variables
    infile = open('OLG3solveVFI_5.pkl', 'rb')
    pickled = pkl.load(infile)
    (coeffs1, coeffs2, timesolve) = pickled
    (Vf11, Vf12, Pf11, Pf12, Jf11, Jf12, coeffsPf11, coeffsPf12, coeffsJf11, \
        coeffsJf12) = coeffs1 
    (Vf21, Vf22, Pf21, Pf22, Jf21, Jf22, coeffsPf21, coeffsPf22, coeffsJf21, \
        coeffsJf22) = coeffs2
    infile.close()
else:
    Vf11 = np.ones((knpts, knpts, znpts)) * (-100000000000)
    Vf12 = np.ones((knpts, knpts, znpts)) * (-100000000000)
    Vf21 = np.ones((knpts, knpts, znpts)) * (-100000000000)
    Vf22 = np.ones((knpts, knpts, znpts)) * (-100000000000)

Vf11new = np.zeros((knpts, knpts, znpts))
Vf12new = np.zeros((knpts, knpts, znpts))
Pf11 = np.zeros((knpts, knpts, znpts))
Pf12 = np.zeros((knpts, knpts, znpts))
Jf11 = np.zeros((knpts, knpts, znpts))
Jf12 = np.zeros((knpts, knpts, znpts))

# set VF iteration parameters
ccrit = 1.0E-8
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
        for i2 in range(0, knpts):
            for i3 in range(0, znpts): # over zt, searching the value for the stochastic shock
                maxval1 = -100000000000
                maxval2 = -100000000000
                for i4 in range(0, knpts): # over k_t+1
                    for i5 in range(0, knpts):
                        for i6 in range(0, ellnpts): # over ell_t
                            for i7 in range(0, ellnpts):
                                Xp = np.array([kgrid[i4], kgrid[i5]])
                                X = np.array([kgrid[i1], kgrid[i2]])
                                Y = np.array([ellgrid[i6], ellgrid[i7]])
                                Z = np.array([zgrid[i3]])
                                K, L, GDP, w, r, T3, B, c1, c2, c3, C, I, \
                                    u1, u2, u3 = Modeldefs(Xp, X, Y, Z, params1)
                                temp1 = u1
                                temp2 = u2
                                for i8 in range(0, znpts): # over z_t+1
                                    temp1 = temp1 + beta * Vf11[i4,i5,i8] * Pimat[i8,i3]
                                    temp2 = temp2 + beta * Vf12[i4,i5,i8] * Pimat[i8,i3]
                                if np.iscomplex(temp1):
                                    temp1 = -1000000000
                                if np.isnan(temp1):
                                    temp1 = -1000000000
                                if np.iscomplex(temp2):
                                    temp2= -1000000000
                                if np.isnan(temp2):
                                    temp2 = -1000000000

                                if temp1 > maxval1:
                                    maxval1 = temp1
                                    Vf11new[i1, i2] = temp1
                                    Pf11[i1, i2, i3] = kgrid[i4]
                                    Jf11[i1, i2, i3] = ellgrid[i6]

                                if temp2 > maxval2:
                                    maxval2 = temp2
                                    Vf12new[i1, i2] = temp2
                                    Pf12[i1, i2, i3] = kgrid[i5] 
                                    Jf12[i1, i2, i3] = ellgrid[i7]
            
        # calculate the new distance measure, we use maximum absolute difference
    dist1 = np.amax(np.abs(Vf11 - Vf11new))
    dist2 = np.amax(np.abs(Vf12 - Vf12new))
    
    dist = np.max((dist1, dist2))
    if dist < ccrit:
        nconv = False
    # report the results of the current iteration
    print ('iteration: ', count, 'distance: ', dist)
    
    # replace the value function with the new one
    Vf11 = 1.0*Vf11new
    Vf12 = 1.0*Vf12new

print ('Converged after', count, 'iterations') 
               
    
# put SS values and starting values into numpy vectors
XYbar = np.array([k2bar1, k3bar1, l1bar1, l2bar1])
X0 = np.array([k2bar1, k3bar1])
Y0 = np.array([l1bar1, l2bar1])   


# -----------------------------------------------------------------------------
# CHANGE POLICY

# create meshgrid
zmesh, kmesh = np.meshgrid(zgrid, kgrid)
    
 # discretize k
klow = (1-kfact)*k2bar2   # what about k3bar2, k4bar2?
khigh = (1+kfact)*k2bar2  # what about k3bar2, k4bar2?
kgrid2 = np.linspace(klow, khigh, num = knpts)   
    
 # discretize ell
elllow = l1bar2 - elladd  # what about k3bar2, k4bar2?
ellhigh = l1bar2 + elladd # what about k3bar2, k4bar2?

ellgrid2 = np.linspace(elllow, ellhigh, num = ellnpts)

Vf21new = np.zeros((knpts, knpts, znpts))
Vf22new = np.zeros((knpts, knpts, znpts))
Pf21 = np.zeros((knpts, knpts, znpts))
Pf22 = np.zeros((knpts, knpts, znpts))
Jf21 = np.zeros((knpts, knpts, znpts))
Jf22 = np.zeros((knpts, knpts, znpts))

# set VF iteration parameters
count = 0
dist = 100.
    
# run the program to get the value function (VF1)
nconv = True 
while (nconv):
    count = count + 1
    if count > maxwhile:
        break
    for i1 in range(0, knpts): # over kt
        for i2 in range(0, knpts):
            for i3 in range(0, znpts): # over zt, searching the value for the stochastic shock
                maxval1 = -100000000000
                maxval2 = -100000000000
                for i4 in range(0, knpts): # over k_t+1
                    for i5 in range(0, knpts):
                        for i6 in range(0, ellnpts): # over ell_t
                            for i7 in range(0, ellnpts):
                                Xp = np.array([kgrid[i4], kgrid[i5]])
                                X = np.array([kgrid[i1], kgrid[i2]])
                                Y = np.array([ellgrid[i6], ellgrid[i7]])
                                Z = np.array([zgrid[i3]])
                                K, L, GDP, w, r, T3, B, c1, c2, c3, C, I, \
                                    u1, u2, u3 = Modeldefs(Xp, X, Y, Z, params1)
                                temp1 = u1
                                temp2 = u2
                                for i8 in range(0, znpts): # over z_t+1
                                    temp1 = temp1 + beta * Vf21[i4,i5,i8] * Pimat[i8,i3]
                                    temp2 = temp2 + beta * Vf22[i4,i5,i8] * Pimat[i8,i3]
                                if np.iscomplex(temp1):
                                    temp1 = -1000000000
                                if np.isnan(temp1):
                                    temp1 = -1000000000
                                if np.iscomplex(temp2):
                                    temp2= -1000000000
                                if np.isnan(temp2):
                                    temp2 = -1000000000

                                if temp1 > maxval1:
                                    maxval1 = temp1
                                    Vf21new[i1, i2] = temp1
                                    Pf21[i1, i2, i3] = kgrid[i4]
                                    Jf21[i1, i2, i3] = ellgrid[i6]

                                if temp2 > maxval2:
                                    maxval2 = temp2
                                    Vf22new[i1, i2] = temp2
                                    Pf22[i1, i2, i3] = kgrid[i5] 
                                    Jf22[i1, i2, i3] = ellgrid[i7]
            
        # calculate the new distance measure, we use maximum absolute difference
    dist1 = np.amax(np.abs(Vf21 - Vf21new))
    dist2 = np.amax(np.abs(Vf22 - Vf22new))
    
    dist = np.max((dist1, dist2))
    if dist < ccrit:
        nconv = False
    # report the results of the current iteration
    print ('iteration: ', count, 'distance: ', dist)
    
    # replace the value function with the new one
    Vf21 = 1.0*Vf11new
    Vf22 = 1.0*Vf12new

print ('Converged after', count, 'iterations')

print ('Policy function at (', (knpts-1)/2, ',', (znpts-1)/2, ') should be', \
    kgrid2[int((knpts-1)/2)], 'and is', Pf22[int((knpts-1)/2), int((znpts-1)/2)])

Pfdiff = Pf21 - Pf22
Jfdiff = Jf21 - Jf22

# fit PF1 and PF2, Jf1 and JF2 with polynomials
# run fitfuncs
nord = 2

funclist = (Pf11, Pf12, Jf11, Jf12) 

coeffsPf11, coeffsPf12, coeffsJf11, coeffsJf12 = \
    fitfuncs(kgrid, kgrid, zgrid, nord, funclist)
    
funclist = (Pf21, Pf22, Jf21, Jf22) 

coeffsPf21, coeffsPf22, coeffsJf21, coeffsJf22 = \
    fitfuncs(kgrid2, kgrid2, zgrid, nord, funclist)

# calculate time to solve for functions
stopsolve = timeit.default_timer()
timesolve =  stopsolve - startsolve
print('time to solve: ', timesolve)

# -----------------------------------------------------------------------------
# SAVE RESULTS

# save grids and polynomials
output = open(name + '.pkl', 'wb')

# set up coefficient list before the policy change
coeffs1 = (Vf11, Vf12, Pf11, Pf12, Jf11, Jf12, coeffsPf11, coeffsPf12, coeffsJf11, coeffsJf12)

# set up coefficient list after the policy change
coeffs2 = (Vf21, Vf22, Pf21, Pf22, Jf21, Jf22, coeffsPf21, coeffsPf22, coeffsJf21, coeffsJf22)

# write timing
pkl.dump((coeffs1, coeffs2, timesolve), output)

output.close()










    
    
    
    



