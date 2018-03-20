# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 14:38:20 2017

@author: Daryl Larsen

This program reads in parameter values and steady states from the file, 
ILAfindss.pkl.

It then calculates the linear coefficients for the policy and jump function
approximations using the GSSA toolkit.

The coefficients and time to solve are written to the file, ILAsolveGSSA.pkl.

The baseline values have a 1 at the end of the variable name.
The values after the policy change have a 2 at the end. 
"""

import numpy as np
import timeit
import pickle as pkl

# import the modules from GSSA

pord = 4
# -----------------------------------------------------------------------------
# GSSA
"""
Created on Wed Oct 11 16:55:35 2017
@author: Daryl Larsen
"""
'''
This program implements the Generalized Stochastic Simulation Algorthim from 
Judd, Maliar and Mailar (2011) "Numerically stable and accurate stochastic 
simulation approaches for solving dynamic economic models", Quantitative
Economics vol. 2, pp. 173-210.
'''
import matplotlib.pyplot as plt
from ILAfuncs import Modeldyn
'''
We test the algorithm with a simple DSGE model with endogenous labor.
'''

def poly1(Xin, XYparams):
    '''
    Includes polynomial terms up to order 'pord' for each element and quadratic 
    cross terms  One observation (row) at a time
    '''
    (pord, nx, ny, nz) = XYparams
    nX = nx + nz
    Xbasis = np.ones((1, 1))
    # generate polynomial terms for each element
    for i in range(1, pord+1):
        Xbasis = np.append(Xbasis, Xin**i)
    # generate cross terms
    for i in range (0, nX):
        for j in range(i+1, nX):
            temp = Xin[i]*Xin[j]
            Xbasis = np.append(Xbasis, temp)
    return Xbasis

def XYfunc(Xm, Zn, XYparams, coeffs):
    (pord, nx, ny, nz) = XYparams
    An = np.exp(Zn)
    XZin = np.append(Xm, An)
    XYbasis = np.append(1., XZin)
    for i in range(1, pord+1):
        XYbasis = poly1(XZin, XYparams)
    XYout = np.dot(XYbasis, coeffs)
    Xn = XYout[0:nx]
    Y = XYout[nx:nx+ny]
    if Y > 0.9999:
        Y = np.array([0.9999])
    elif Y < 0:
        Y = np.array([0])
    return Xn, Y
    
def MVOLS(Y, X):
    '''
    OLS regression with observations in rows
    '''
    XX = np.dot(np.transpose(X), X)
    XY = np.dot(np.transpose(X), Y)
    coeffs = np.linalg.solve(XX, XY)
    return coeffs
 
def GSSA(params, kbar, ellbar, GSSAparams, old_coeffs):
    regtype = 'poly1' # functional form for X & Y functions 
    fittype = 'MVOLS'   # regression fitting method
    ccrit = 1.0E-8  # convergence criteria for XY change
    damp = 0.05  # damping paramter for fixed point algorithm
    
    [alpha, beta, gamma, delta, chi, theta, tau, rho, sigma] = params
    (T, nx, ny, nz, pord, old) = GSSAparams

    Xstart = kbar
    XYparams = (pord, nx, ny, nz)
    cnumb = int((pord+1)*(nx+nz) + .5*(nx+nz-1)*(nx+nz-2))
    cnumb2 = int(3*(nx+nz) + .5*(nx+nz-1)*(nx+nz-2))
    #create history of Z's
    Z = np.zeros([T,nz])
    for t in range(1,T):
        Z[t,:] = rho*Z[t-1] + np.random.randn(1)*sigma
    if regtype == 'poly1' and old == False:
        coeffs = np.array([[  1.14283341e+03,   2.48767872e+01], \
                           [ -9.97029153e+01,  -3.18909650e+00], \
                           [ -3.13420276e+02,   1.38021793e+01], \
                           [  2.15133629e+00,   1.01468495e-01], \
                           [ -1.89909459e+00,   7.37298161e-01], \
                           [  1.63137901e+01,  -7.87431895e-01]])
    elif old == True:
        coeffs = old_coeffs
    
    if old == False and pord > 2:
        A = np.zeros((cnumb-cnumb2, nx+ny))
        coeffs = np.insert(coeffs, cnumb2 - 1, A)

    dist = 1.
    distold = 10000.
    count = 0
    damp = .01
    XYold = np.ones((T, nx+ny))

    while dist > ccrit:
        count = count + 1
        X = np.zeros((T+1, nx))
        Y = np.zeros((T, ny))
        Xin = np.zeros((T, nx+nz))
        A = np.exp(Z)
        x = np.zeros((T,(pord*2+2)))
        X[0], Y[0] = XYfunc(Xstart, Z[0], XYparams, coeffs)
        for t in range(1,T+1):
            X[t], Y[t-1] = XYfunc(X[t-1], Z[t-1], XYparams, coeffs)
            Xin[t-1,:] = np.concatenate((X[t-1], A[t-1]))
            x[t-1,:] = poly1(Xin[t-1,:], XYparams)
            X1 = X[0:T]
            Y1 = Y[0:T]
            # plot time series
        if count % 100 == 0:
            timeperiods = np.asarray(range(0,T))
            plt.subplot(2,1,1)
            plt.plot(timeperiods, X1, label='X')
            plt.axhline(y=kbar, color='r')
            plt.subplot(2,1,2)
            plt.plot(timeperiods, Y1, label='Y')
            plt.axhline(y=ellbar, color='g')
            plt.xlabel('time')
            plt.legend(loc=9, ncol=(nx+ny))
            plt.show()    
    
        # Generate consumption, lambda, and gamma series
        Gam = np.zeros((T-1, 1))
        Lam = np.zeros((T-1, 1))
        for t in range(0, T-1):
            theta0 = (X[t+2], X[t+1], X[t], Y[t+1], Y[t], Z[t+1], Z[t])
            Lam[t], Gam[t] = Modeldyn(theta0, params) + 1
            Gam[t] = 1/Gam[t]
        # (T-1)-by-1
    
        # update values for X and Y
        temp1 = np.mean(Gam)
        temp2 = np.mean(Lam)
        Xnew = (Gam)*X[1:T]
        Ynew = (Lam)*Y[1:T]
        XY = np.append(X[0:T], Y, axis=1)
        XYnew = np.append(Xnew, Ynew, axis = 1)
        temp = np.append(Gam, Lam, axis = 1)
        x = x[0:T-1,:]
        
        if fittype == 'MVOLS':
            coeffsnew = MVOLS(XYnew, x)
        
        dist = np.mean(np.abs(1-XY/XYold))
        print('count', count, 'distance', dist, 'Gam', temp1, 'Lam', temp2)

        # update coeffs
        XYold = XY*1.
        coeffs = (1-damp)*coeffs + damp*coeffsnew
        if count % 100 == 0:
            print('coeffs', coeffs)
    return coeffs
# -----------------------------------------------------------------------------
# READ IN VALUES FROM STEADY STATE CALCULATIONS

# load steady state values and parameters
infile = open('ILAfindss.pkl', 'rb')
(bar1, bar2, params1, params2, GSSAparams) = pkl.load(infile)
infile.close()

infile = open('Results_GSSA\\pord=3, tau1 = .05, tau2 = .055, OfficePC\\ILAsolveGSSA_3.pkl', 'rb')
(coeffs3a, coeffs3b, timesolve) = pkl.load(infile)
infile.close()

A = np.zeros((2,2))
coeffs3a = np.insert(coeffs3a, 2*pord-1, A, axis=0)
coeffs3b = np.insert(coeffs3b, 2*pord-1, A, axis=0)

try:
    infile = open('ILAsolveGSSA_4.pkl', 'rb')
    (coeffs4a, coeffs4b, timesolve) = pkl.load(infile)
    infile.close()
    old_pord = False
except FileNotFoundError:
    old_pord = True
    pass
old_pord = True
# unpack
[kbar1, ellbar1, Ybar1, wbar1, rbar1, Tbar1, cbar1, ibar1, ubar1] = bar1
[kbar2, ellbar2, Ybar2, wbar2, rbar2, Tbar2, cbar2, ibar2, ubar2] = bar2
[alpha, beta, gamma, delta, chi, theta, tau, rho_z, sigma_z] = params1
tau2 = params2[6]
(zbar, Zbar, NN, nx, ny, nz, logX, Sylv) = GSSAparams

# set clock for time to calcuate functions
startsolve = timeit.default_timer()

# set name for external files written
name = 'ILAsolveGSSA_4'

# -----------------------------------------------------------------------------
# BASELINE
T = 1000
old = True
GSSAparams = (T, nx, ny, nz, pord, old)
# find the policy and jump function coefficients
if old_pord == True:
    coeffs1 = GSSA(params1, kbar1, ellbar1, GSSAparams, coeffs3a)
elif old_pord == False:
    coeffs1 = GSSA(params1, kbar1, ellbar1, GSSAparams, coeffs4a)
# -----------------------------------------------------------------------------
# CHANGE POLICY

# find the policy and jump function coefficients
if old_pord == True:
    coeffs2 = GSSA(params2, kbar2, ellbar2, GSSAparams, coeffs3b)
elif old_pord == False:
    coeffs2 = GSSA(params2, kbar2, ellbar2, GSSAparams, coeffs4b)
print ('baseline coeffs')
print (coeffs1)
print  (' ')
print ('policy change coeffs')
print(coeffs2)
print  (' ')

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