# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 16:55:35 2017
@author: Daryl Larsen
"""
import numpy as np
import matplotlib.pyplot as plt

def poly1(Xin, XYparams):
    '''
    Includes polynomial terms up to order 'pord' for each element and quadratic 
    cross terms  One observation (row) at a time
    '''
    (coeffs, pord, nx, ny, nz) = XYparams
    nx = XYparams[2]
    nz = XYparams[4]
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
    (coeffs, pord, nx, ny, nz) = XYparams
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
    ccrit = 1.0E-6  # convergence criteria for XY change
    nx = 1
    ny = 0
    nz = 1
    damp = 0.01  # damping paramqter for fixed point algorithm
    
    [alpha, beta, tau, rho, sigma] = params
    
    Xstart = kbar
    
    #create history of Z's
    Z = np.zeros([T,nz])
    for t in range(1,T):
        Z[t,:] = rho*Z[t-1] + np.random.randn(1)*sigma
    
    if regtype == 'poly1':
        coeffs = np.array([[ -2.04961035e-02], \
                           [  2.26920891e-01], \
                           [  1.17409797e-01], \
                           [ -6.27573544e-01], \
                           [ -4.88424960e-05], \
                           [  3.49581228e-01]])
    
    dist = 1.
    distold = 2.
    count = 0
    Xold = np.ones((T-1, nx+ny))
    XYparams = (coeffs, pord, nx, ny, nz)

    
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
        Gam = (beta*c[1:T]**(-1)*(alpha*X[1:T]**(alpha-1)*A[1:T]*(1-tau))) / (c[0:T-1]**(-1))
        
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
        Xold = Xnew
        coeffs = (1-damp)*coeffs + damp*coeffsnew
        if count % 10 == 0:
            print('coeffs', coeffs)
        coeffs = (1-damp)*coeffs + damp*coeffsnew
        
    return coeffs