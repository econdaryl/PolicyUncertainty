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
from OLGfuncs import Modeldyn
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
    for i in range(1, 3):
        Xbasis = np.append(Xbasis, Xin**i)
        #print('Xbasis', Xbasis)
    # generate cross terms
    for i in range (0, nX):
        #for j in range(nx, nX):
        for j in range(i+1, nX):
            temp = Xin[i]*Xin[j]
            Xbasis = np.append(Xbasis, temp)
    return Xbasis

def XYfunc(Xm, Zn, XYparams, coeffs):
    (pord, nx, ny, nz) = XYparams
    An = np.exp(Zn)
    XZin = np.append(Xm, An)
    #print('XZin', XZin)
    XYbasis = np.append(1., XZin)
    for i in range(1, pord):
        XYbasis = poly1(XZin, XYparams)
    XYout = np.dot(XYbasis, coeffs)
    Xn = XYout[0:nx]
    Yn = XYout[nx:nx+ny]
    return Xn, Yn
    
def MVOLS(Y, X):
    '''
    OLS regression with observations in rows
    '''
    XX = np.dot(np.transpose(X), X)
    XY = np.dot(np.transpose(X), Y)
    coeffs = np.linalg.solve(XX, XY)
    return coeffs
 
def GSSA(params, kbar, ellbar):
    T = 10000
    regtype = 'poly1' # functional form for X & Y functions 
    fittype = 'MVOLS'   # regression fitting method
    pord = 3  # order of polynomial for fitting function
    ccrit = 1.0E-6  # convergence criteria for XY change
    damp = 0.01  # damping paramter for fixed point algorithm
    
    [alpha, beta, gamma, delta, chi, theta, tau, rho, \
    sigma, pi2, pi3, pi4, f1, f2, f3, nx, ny, nz] = params
    nx = int(nx)
    ny = int(ny)
    nz = int(nz)
    XYparams = (pord, nx, ny, nz)

    Xstart = kbar
    
    #create history of Z's
    Z = np.zeros((T,nz))
    for t in range(1,T):
        Z[t,:] = rho*Z[t-1] + np.random.randn(1)*sigma
    if regtype == 'poly1':
        cnumb = int(pord*(nx+nz) + .5*(nx+nz-1)*(nx+nz-2))
        coeffs = np.ones((cnumb,(nx+ny)))*.01
        for i in range(0, nx+ny) :
            coeffs[:, i] = coeffs[:, i]*(i+1)
            for j in range(0, cnumb):
                coeffs[j,i] = coeffs[j,i] + np.random.randn(1)*.005
            #coeffs[:,i] = coeffs[:,i]*np.random.exponential(1)
            #for j in range(0, cnumb) :
            #    coeffs[j,i] = coeffs[j,i]*np.random.randn(1)*0.05
    print('coeffs', coeffs)
 
    dist = 1.
    distold = 2.
    count = 0
    damp = .01
    XYold = np.ones((T-1, nx+ny))

    while dist > ccrit:
        count = count + 1
        X = np.zeros((T+1, nx))
        Y = np.zeros((T+1, ny))
        Xin = np.zeros((T, nx+nz))
        A = np.exp(Z)
        x = np.zeros((T,(pord*5)))
        X[0, :], Y[0, :] = XYfunc(Xstart, Z[0], XYparams, coeffs)
        for t in range(1,T+1):
            X[t, :], Y[t, :] = XYfunc(X[t-1, :], Z[t-1, :], XYparams, coeffs)
            Xin[t-1,:] = np.concatenate((X[t-1], A[t-1]))
            x[t-1,:] = poly1(Xin[t-1,:], XYparams)
        X1 = X[0:T, :]
        Y1 = Y[0:T, :]
        Lam1 = np.zeros((T-1, 1))
        Lam2 = np.zeros((T-1, 1))
        Lam3 = np.zeros((T-1, 1))
        Gam1 = np.zeros((T-1, 1))
        Gam2 = np.zeros((T-1, 1))
        Gam3 = np.zeros((T-1, 1))
        for t in range(0, T-1):
            inmat = np.concatenate((X[t+2, :], X[t+1, :], X[t, :], Y[t+1, :], Y[t, :], Z[t+1, :], Z[t, :]))
            Lam1[t], Lam2[t], Lam3[t], Gam1[t], Gam2[t], Gam3[t] = (Modeldyn(inmat, params) + 1)
        Gam = np.hstack((Gam1, Gam2, Gam3))
        print('Gam', Gam)
        Lam = np.hstack((Lam1, Lam2, Lam3))
        # plot time series
        if count % 10 == 0:
            timeperiods = np.asarray(range(0,T))
            plt.subplot(2,1,1)
            plt.plot(timeperiods, X1, label='X')
            #plt.axhline(y=kbar, color='r')
            plt.subplot(2,1,2)
            plt.plot(timeperiods, Y1, label='Y')
            #plt.axhline(y=ellbar, color='g')
            plt.xlabel('time')
            # plt.legend(loc=9, ncol=(nx+ny))
            plt.show()    
        
        #Generate Gamma and lambda series
        #B = (1+r-delta)*((1-pi2)*k2 + (1-pi3)*pi2*k3 + (1-pi4)*pi3*k4) \
        #/(1+pi2+pi3+pi4)
        #c1 = (1-tau)*(w*f1*l1) + B - k2p
        #c2 = (1-tau)*(w*f2*l2) + B + (1+r-delta)*k2 - k3p
        #c3 = (1-tau)*(w*f3*l3) + B + (1+r-delta)*k3 - k4p
        #c4 = (1+r-delta)*k4 + B + T4

    
        # update values for X and Y
        Xnew = (Gam)*X[1:T, :]
        Ynew = (Lam)*Y[1:T, :]
        XY = np.append(Xnew, Ynew, axis = 1)
        x = x[0:T-1,:]
        
        if fittype == 'MVOLS':
            coeffsnew = MVOLS(XY, x)
        
        dist = np.mean(np.abs(1-XY/XYold))
        print('count ', count, 'distance', dist, 'distold', distold, \
              'damp', damp)
        
        if dist < distold:
            damp = damp*1.05
            if damp > 1.:
                damp = 1.
        else:
            damp = damp*.8
            if damp > .001:
                damp = .001

        distold = 1.*dist
    
        # update coeffs
        XYold = XY
        coeffs = (1-damp)*coeffs + damp*coeffsnew
        if count % 10 == 0:
            print('coeffs', coeffs)
    return coeffs