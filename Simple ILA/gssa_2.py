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
from LinApp_FindSS import LinApp_FindSS
from ILAfuncs import Modeldyn, Modeldefs
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
    for i in range(1, pord):
        XYbasis = poly1(XZin, XYparams)
    XYout = np.dot(XYbasis, coeffs)
    Xn = XYout[0:nx]
    Y = XYout[nx:nx+ny]
    if Y > 0.9999:
        Y = 0.9999
    elif Y < 0.0001:
        Y = 0.0001
    return Xn, Y
    
def MVOLS(Y, X):
    '''
    OLS regression with observations in rows
    '''
    XX = np.dot(np.transpose(X), X)
    XY = np.dot(np.transpose(X), Y)
    coeffs = np.linalg.solve(XX, XY)
    return coeffs
 
def GSSA(params, kbar, ellbar):
    T = 150
    regtype = 'poly1' # functional form for X & Y functions 
    fittype = 'MVOLS'   # regression fitting method
    pord = 2  # order of polynomial for fitting function
    ccrit = 1.0E-8  # convergence criteria for XY change
    nx = 1
    ny = 1
    nz = 1
    damp = 0.01  # damping paramter for fixed point algorithm
    
    [alpha, beta, gamma, delta, chi, theta, tau, rho, sigma] = params
    XYparams = (pord, nx, ny, nz)

    Xstart = kbar
    
    #create history of Z's
    Z = np.zeros([T,nz])
    for t in range(1,T):
        Z[t,:] = rho*Z[t-1] + np.random.randn(1)*sigma
    if regtype == 'poly1':
        coeffs = np.array([[0., 0.], \
                           [0.99, 0.95*ellbar], \
                           [0., 0.05*ellbar], \
                           [0., 0.]])

        #coeffs = np.array([[ -1.18633091e+00,   3.45793044e-01], \
        #                   [  9.89020856e-01,  -1.28876209e-02], \
        #                   [  1.88646740e+00,   5.12251894e-01], \
        #                   [ -1.16376098e-03,   2.74312088e-05], \
        #                   [ -4.35797445e-01,  -1.64555348e-01], \
        #                   [  1.99503799e-02,   2.30853778e-04]])
    dist = 1.
    distold = 2.
    count = 0
    XYold = np.ones((T-1, nx+ny))

    while dist > ccrit:
        count = count + 1
        X = np.zeros((T+1, nx))
        Y = np.zeros((T, ny))
        Xin = np.zeros((T, nx+nz))
        A = np.exp(Z)
        x = np.zeros((T,(pord*2)))
        X[0], Y[0] = XYfunc(Xstart, Z[0], XYparams, coeffs)
        for t in range(1,T+1):
            X[t], Y[t-1] = XYfunc(X[t-1], Z[t-1], XYparams, coeffs)
            Xin[t-1,:] = np.concatenate((X[t-1], A[t-1]))
            x[t-1,:] = poly1(Xin[t-1,:], XYparams)
        # plot time series
        if count % 200 == 0:
            X1 = X[0:T]
            timeperiods = np.asarray(range(0,T))
            plt.subplot(2,1,1)
            plt.plot(timeperiods, X1, label='X')
            plt.axhline(y=kbar, color='r')
            plt.subplot(2,1,2)
            plt.plot(timeperiods, Y, label='Y')
            plt.axhline(y=ellbar, color='g')
            plt.xlabel('time')
            plt.legend(loc=9, ncol=(nx+ny))
            plt.show()    
    
        # Generate consumption, lambda, and gamma series
        GDP = X[0:T]**alpha*(A[0:T]*Y[0:T])**(1-alpha)
        r = alpha*GDP[0:T] / X[0:T]
        w = (1-alpha)*GDP[0:T] / Y[0:T]
        Tax = tau*(w*Y + (r-delta)*X[0:T])
        c = (1-tau)*(w*Y + (r-delta)*X[0:T]) + X[0:T] + Tax - X[1:T+1]
        # T-by-1
        Lam = (c[0:T-1]**(-gamma)*(1-tau)*w[0:T-1]) / (chi*Y[0:T-1]**theta)
        # (T-1)-by-1
        Gam = (beta*c[1:T]**(-gamma)*(1 + (1-tau)*(r[1:T] - delta))) / (c[0:T-1]**(-gamma))
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
        
        dist = np.mean(np.abs(1-(XY/XYold)))
        #dist = np.abs(np.mean(X) - kbar)
        if count % 100 == 0:
            print('count ', count, 'distance', dist, 'damp', damp, 'Gam', temp1, 'Lam', temp2)
        
        #if dist < distold:
        #    damp = damp*1.05
        #    if damp > 1.:
        #        damp = 1.
        #else:
        #    damp = damp*.8
        #    if damp < 0.001:
        #        damp = 0.001

        distold = 1.*dist
    
        # update coeffs
        XYold = XY*1.
        coeffs = (1-damp)*coeffs + damp*coeffsnew
    return coeffs