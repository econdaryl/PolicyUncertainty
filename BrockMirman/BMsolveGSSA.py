#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Thu Sep 28 16:55:35 2017
@author: Daryl Larsen


This program reads in paramter values and steady states from the file, 
ILAfindss.pkl.

It then calculates the linear coefficients for the policy and jump function
approximations using the LinApp toolkit.

The coefficients and time to solve are written to the file, ILAsolveLIN.pkl.

The baseline values have a 1 at the end of the variable name.
The values after the policy change have a 2 at the end. 
"""

import timeit
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt

def poly1(Xin, XYparams):
    '''
    Includes polynomial terms up to order 'pord' for each element and quadratic 
    cross terms  One observation (row) at a time
    '''
    pord = XYparams[0]
    nx = XYparams[1]
    nz = XYparams[3]
    nX = nx + nz
    Xbasis = np.ones((1, 1))
    # generate polynomial terms for each element
    for i in range(1, pord):
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
    for i in range(1, pord):
        XYbasis = poly1(XZin, XYparams)
    Xn = np.dot(XYbasis, coeffs)
    return Xn

    
def MVOLS(Y, X):
    '''
    OLS regression with observations in rows
    '''
    XX = np.dot(np.transpose(X), X)
    XY = np.dot(np.transpose(X), Y)
    coeffs = np.dot(np.linalg.inv(XX), XY)
    return coeffs


def GSSA(params, kbar):  
    T = 10000
    regtype = 'poly1' # functional form for X & Y functions 
    fittype = 'MVOLS'   # regression fitting method
    pord = 3  # order of polynomial for fitting function
    ccrit = 1.0E-8  # convergence criteria for XY change
    nx = 1
    ny = 0
    nz = 1
    damp = 0.5  # damping paramqter for fixed point algorithm
    
    [alpha, beta, tau, rho, sigma] = params
    
    Xstart = kbar
    
    #create history of Z's
    Z = np.zeros([T,nz])
    for t in range(1,T):
        Z[t,:] = rho*Z[t-1] + np.random.randn(1)*sigma
    
    if regtype == 'poly1':
#        coeffs = np.array([[ -2.04961035e-02], \
#                           [  2.26920891e-01], \
#                           [  1.17409797e-01], \
#                           [ -6.27573544e-01], \
#                           [ -4.88424960e-05], \
#                           [  3.49581228e-01]])
        coeffs = np.array([[ kbar], \
                       [ .35], \
                       [.18095882], \
                       [ 0.], \
                       [ 0.], \
                       [ 0.]])
    
    dist = 1.
    distold = 2.
    count = 0
    Xold = np.ones((T-1, nx+ny))
    XYparams = (pord, nx, ny, nz)

    
    while dist > ccrit:
        count = count + 1
        X = np.zeros((T+1, nx))
        Xin = np.zeros((T, nx+nz))
        A = np.exp(Z)
        x = np.zeros((T,6))
        X[0] = XYfunc(Xstart, Z[0], XYparams, coeffs)
        for t in range(1,T+1):
            X[t] = XYfunc(X[t-1], Z[t-1], XYparams, coeffs)
            Xin[t-1,:] = np.concatenate((X[t-1], A[t-1]))
            x[t-1,:] = poly1(Xin[t-1,:], XYparams)
        # plot time series
        if count % 10 == 0:
            X1 = X[0:T]
            timeperiods = np.asarray(range(0,T))
            plt.plot(timeperiods, X1, label='X')
            plt.axhline(y=kbar, color='r')
            plt.title('time series')
            plt.xlabel('time')
            plt.legend(loc=9, ncol=(nx+ny))
            plt.show()    
    
        # Generate consumption and gamma series     
        c = (1-tau)*X[0:T]**alpha*A[0:T] - X[1:T+1]
        Gam = (beta*c[1:T]**(-1)*(alpha*X[1:T]**(alpha-1)*A[1:T]*(1-tau))) / \
            (c[0:T-1]**(-1))
        
        # update values for X and Y
        Xnew = Gam*X[1:T]
        x = x[0:T-1,:]
  
        if fittype == 'MVOLS':
            coeffsnew = MVOLS(Xnew, x)
        
        if dist < distold:
            damp = damp*1.05
            if damp > 1.:
                damp = 1.
        else:
            damp = damp*.8
            if damp < 0.001:
                damp = 0.001
        
        distold = 1.*dist

        # calculate distance between X and Xold
        dist = np.mean(np.abs(1-(Xnew/Xold)))
        print('count ', count, 'distance', dist, 'damp', damp)
    
        # update coeffs
        Xold = 1*Xnew
        coeffs = (1-damp)*coeffs + damp*coeffsnew
        if count % 10 == 0:
            print('coeffs', coeffs)
        coeffs = (1-damp)*coeffs + damp*coeffsnew
        
    return coeffs


# -----------------------------------------------------------------------------
# READ IN VALUES FROM STEADY STATE CALCULATIONS

# load steady state values and parameters
infile = open('BMfindss.pkl', 'rb')
(bar1, bar2, params1, params2, LINparams) = pkl.load(infile)
infile.close()

infile = open('BMsolveGSSA.pkl', 'rb')
(coeffsa, coeffsb, timesolve) = pkl.load(infile)
infile.close()

# unpack
[kbar1, Ybar1, wbar1, rbar1, Tbar1, cbar1, ibar1, ubar1] = bar1
[kbar2, Ybar2, wbar2, rbar2, Tbar2, cbar2, ibar2, ubar2] = bar2
[alpha, beta, tau, rho_z, sigma_z] = params1
tau2 = params2[2]
(zbar, Zbar, NN, nx, ny, nz, logX, Sylv) = LINparams

# set clock for time to calcuate functions
startsolve = timeit.default_timer()

# set name for external files written
name = 'BMsolveGSSA_KH'

# -----------------------------------------------------------------------------
# BASELINE
T = 10000
pord = 2
old = True
# set up steady state input vector for baseline
GSSAparams = (T, nx, ny, nz, pord, old)
coeffs1 = GSSA(params1, kbar1, GSSAparams, coeffsa)

# -----------------------------------------------------------------------------
# CHANGE POLICY

# set up coefficient list
coeffs2 = GSSA(params2, kbar2, GSSAparams, coeffsb)

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