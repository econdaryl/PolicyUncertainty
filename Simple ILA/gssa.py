# -*- coding: utf-8 -*-
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
import numpy as np
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
    XYold = np.ones((T-1, nx+ny))

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
        XY = np.append(Xnew, Ynew, axis = 1)
        x = x[0:T-1,:]
        
        if fittype == 'MVOLS':
            coeffsnew = MVOLS(XY, x)
        
        dist = np.mean(np.abs(1-XY/XYold))
        print('count', count, 'distance', dist, 'Gam', temp1, 'Lam', temp2)
        '''
        if dist < distold:
            damp = damp*1.05
            if damp > 1.:
                damp = 1.
        else:
            damp = damp*.8
            if damp < 0.001:
                damp = 0.001
        
        distold = 1.*dist
        '''
        # update coeffs
        XYold = XY*1.
        coeffs = (1-damp)*coeffs + damp*coeffsnew
        if count % 100 == 0:
            print('coeffs', coeffs)
    return coeffs