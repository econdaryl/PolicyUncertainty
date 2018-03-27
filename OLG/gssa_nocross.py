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
import sys
'''
We test the algorithm with a simple DSGE model with endogenous labor.
'''

def poly1(Xin, XYparams):
    '''
    Includes polynomial terms up to order 'pord' for each element and quadratic 
    cross terms  One observation (row) at a time
    '''
    (pord, nx, ny, nz, kbar) = XYparams
    Xbasis = np.ones((1, 1))
    # generate polynomial terms for each element
    for i in range(1, pord+1):
        Xbasis = np.append(Xbasis, Xin**i)
        #print('Xbasis', Xbasis)
    return Xbasis

def XYfunc(Xm, Zn, XYparams, coeffs):
    (pord, nx, ny, nz, kbar) = XYparams
    An = np.exp(Zn)
    XZin = np.append(Xm, An)
    #print('XZin', XZin)
    XYbasis = np.append(1., XZin)
    for i in range(1, pord+1):
        XYbasis = poly1(XZin, XYparams)
    XYout = np.dot(XYbasis, coeffs)
    Xn = XYout[0:nx]
    Y = XYout[nx:nx+ny]
    #temp = type(Y)
    #print('Y is ', temp)
    Y[Y > 0.9999] = 0.9999
    Y[Y < 0.0001] = 0.0001
    #print('Xn', Xn, 'Y', Y)
    #if np.isnan(Xn).any():
    #    sys.exit()
    #for i in range(0, ny):
        #temp = Xn[i] < -1. or Y[i] > 1.
        #print(temp)
    #    if Xn[i] < (kbar[i] - 0.001):
    #        Xn[i] = kbar[i] - 0.001
    #    elif Xn[i] > (kbar[i] + 0.001):
    #        Xn[i] = kbar[i] + 0.001
    #    else:
    #        continue
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
    damp = 0.01  # damping paramter for fixed point algorithm
    
    [alpha, beta, gamma, delta, chi, theta, tau, rho, \
    sigma, pi2, pi3, pi4, f1, f2, f3, nx, ny, nz] = params
    (T, nx, ny, nz, pord, old) = GSSAparams
    (kbar2, kbar3, kbar4) = kbar
    (ellbar1, ellbar2, ellbar3) = ellbar
    nx = int(nx)
    ny = int(ny)
    nz = int(nz)
    XYparams = (pord, nx, ny, nz, kbar)

    Xstart = kbar
    
    #create history of Z's
    Z = np.zeros((T,nz))
    for t in range(1,T):
        Z[t,:] = rho*Z[t-1] + np.random.randn(1)*sigma
    if regtype == 'poly1' and old == False:
        coeffs = np.array([[0., 0., 0., 0.95*ellbar1, 0.95*ellbar2, 0.95*ellbar3], \
                           [0.95, 0., 0., 0., 0., 0.], \
                           [0., 0.95, 0., 0., 0., 0.], \
                           [0., 0., 0.95, 0., 0., 0.], \
                           [0.1*kbar2, 0.05*kbar3, 0.05*kbar4, 0.04*ellbar1, 0.04*ellbar2, 0.04*ellbar3], \
                           [0., 0., 0., 0., 0., 0.], \
                           [0., 0., 0., 0., 0., 0.], \
                           [0., 0., 0., 0., 0., 0.], \
                           [0., 0., 0., 0., 0., 0.]])
    else:
        coeffs = old_coeffs
    #print('coeffs', coeffs)
 
    dist = 1.
    distold = 2.
    count = 0
    damp = .01
    XYold = np.ones((T-1, nx+ny))

    while dist > ccrit and count < 10000:
        count = count + 1
        X = np.zeros((T+1, nx))
        Y = np.zeros((T, ny))
        Xin = np.zeros((T, nx+nz))
        A = np.exp(Z)
        x = np.zeros((T,9))
        X[0, :], Y[0, :] = XYfunc(Xstart, Z[0], XYparams, coeffs)
        for t in range(1,T+1):
            X[t, :], Y[t-1, :] = XYfunc(X[t-1, :], Z[t-1, :], XYparams, coeffs)
            Xin[t-1,:] = np.concatenate((X[t-1], A[t-1]))
            x[t-1,:] = poly1(Xin[t-1,:], XYparams)
        X1 = X[0:T, :]
        #Lam1 = np.zeros((T-1, 1))
        #Lam2 = np.zeros((T-1, 1))
        #Lam3 = np.zeros((T-1, 1))
        #Gam1 = np.zeros((T-1, 1))
        #Gam2 = np.zeros((T-1, 1))
        #Gam3 = np.zeros((T-1, 1))
        #for t in range(0, T-1):
        #    inmat = np.concatenate((X[t+2, :], X[t+1, :], X[t, :], Y[t+1, :], Y[t, :], Z[t+1, :], Z[t, :]))
        #    Lam1[t], Lam2[t], Lam3[t], Gam1[t], Gam2[t], Gam3[t] = (Modeldyn(inmat, params) + 1)
        #Gam = np.hstack((Gam1, Gam2, Gam3))
        #print('Gam', Gam)
        #Lam = np.hstack((Lam1, Lam2, Lam3))
        # plot time series
        if count % 100 == 0:
            timeperiods = np.asarray(range(0,T))
            plt.subplot(2,1,1)
            plt.plot(timeperiods, X1, label='X')
            plt.axhline(y=kbar2, color='k')
            plt.axhline(y=kbar3, color='r')
            plt.axhline(y=kbar4, color='b')
            plt.subplot(2,1,2)
            plt.plot(timeperiods, Y, label='Y')
            plt.axhline(y=ellbar1, color='k')
            plt.axhline(y=ellbar2, color='r')
            plt.axhline(y=ellbar3, color='b')
            plt.xlabel('time')
            plt.legend(loc=9, ncol=(nx+ny))
            plt.show()    
        
        #Generate Gamma and lambda series
        k2 = np.reshape(X[:, 0], (T+1, 1))
        k3 = np.reshape(X[:, 1], (T+1, 1))
        k4 = np.reshape(X[:, 2], (T+1, 1))
        l1 = np.reshape(Y[:, 0], (T, 1))
        l2 = np.reshape(Y[:, 1], (T, 1))
        l3 = np.reshape(Y[:, 2], (T, 1))
        for t in range(0, T):
            if l1[t] > 0.9999:
                l1[t] = 0.9999
            elif l1[t] < 0.0001:
                l1[t] = 0.0001
            if l2[t] > 0.9999:
                l2[t] = 0.9999
            elif l2[t] < 0.0001:
                l2[t] = 0.0001  
            if l3[t] > 0.9999:
                l3[t] = 0.9999
            elif l3[t] < 0.0001:
                l3[t] = 0.0001
        K = k2 + pi2*k3 + pi3*k4
        L = f1*l1 + pi2*f2*l2 + pi3*f3*l3
        GDP = K[0:T]**alpha*(A*L)**(1-alpha)
        r = alpha*GDP / K[0:T]
        w = (1-alpha)*GDP / L
        T4 = tau*w*L
        B = (1+r-delta)*((1-pi2)*k2[0:T] + (1-pi3)*pi2*k3[0:T] + (1-pi4)*pi3*k4[0:T]) \
        /(1+pi2+pi3+pi4)
        c1 = (1-tau)*(w*f1*l1) + B - k2[1:T+1]
        c2 = (1-tau)*(w*f2*l2) + B + (1+r-delta)*k2[0:T] - k3[1:T+1]
        c3 = (1-tau)*(w*f3*l3) + B + (1+r-delta)*k3[0:T] - k4[1:T+1]
        c4 = (1+r-delta)*k4[0:T] + B + T4
        # T-by-1
        El1 = (c1[0:T-1]**(-gamma)*(1-tau)*w[0:T-1]*f1) / (chi*l1[0:T-1]**theta)
        El2 = (c2[0:T-1]**(-gamma)*(1-tau)*w[0:T-1]*f2) / (chi*l2[0:T-1]**theta)
        El3 = (c3[0:T-1]**(-gamma)*(1-tau)*w[0:T-1]*f3) / (chi*l3[0:T-1]**theta)
        Ek2 = (c1[0:T-1]**(-gamma)) / (beta*c2[1:T]**(-gamma)*(1 + r[1:T] - delta))
        Ek3 = (beta*c3[1:T]**(-gamma)*(1 + r[1:T] - delta)) / (c2[0:T-1]**(-gamma))
        Ek4 = (beta*c4[1:T]**(-gamma)*(1 + r[1:T] - delta)) / (c3[0:T-1]**(-gamma))
        # T-1-by-1
        Gam = np.hstack((Ek2, Ek3, Ek4))
        #print('Gam', Gam)
        Lam = np.hstack((El1, El2, El3))

        # update values for X and Y
        temp1 = np.mean(Gam)
        temp2 = np.mean(Lam)
        Xnew = (Gam)*X[1:T, :]
        Ynew = (Lam)*Y[1:T, :]
        XY = np.append(Xnew, Ynew, axis = 1)
        x = x[0:T-1,:]
        
        if fittype == 'MVOLS':
            coeffsnew = MVOLS(XY, x)
        
        dist = np.mean(np.abs(1-XY/XYold))
        print('count ', count, 'distance', dist, \
              'Gam', temp1, 'Lam', temp2)
        
        if dist < distold:
            damp = damp*1.05
        else:
            damp = damp*.8
        
        if damp > 1.:
            damp = 1.
        elif damp < .001:
            damp = .001

        distold = 1.*dist
    
        # update coeffs
        XYold = XY
        coeffs = (1-damp)*coeffs + damp*coeffsnew
        if count % 100 == 0:
            print('coeffs', coeffs)
    return coeffs