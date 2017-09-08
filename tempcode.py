#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 16:08:19 2017

@author: klp4
"""

import numpy as np
import matplotlib.pyplot as plt

# import the modules from LinApp
from LinApp_FindSS import LinApp_FindSS

def Modeldefs(Xp, X, Y, Z, params):
    '''
    This function takes vectors of endogenous and exogenous state variables
    along with a vector of 'jump' variables and returns explicitly defined
    values for consumption, gdp, wages, real interest rates, and transfers
    
    Inputs are:
        Xp: value of capital in next period
        X: value of capital this period
        Y: value of labor this period
        Z: value of productivity this period
        params: list of parameter values
    
    Output are:
        Y: GDP
        w: wage rate
        r: rental rate on capital
        T: transfer payments
        c: consumption
        i: investment
        u: utiity
    '''
    
    # unpack input vectors
    kp = Xp
    k = X
    ell = Y
    z = Z
    
    # unpack params
    [alpha, beta, gamma, delta, chi, theta, tau, rho_z, sigma_z] = params
    
    # find definintion values
    Y = k**alpha*(np.exp(z)*ell)**(1-alpha)
    w = (1-alpha)*Y/ell
    r = alpha*Y/k
    T = tau*(w*ell + (r - delta)*k)
    c = (1-tau)*(w*ell + (r - delta)*k) + k + T - kp
    i = Y - c
    u = c**(1-gamma)/(1-gamma) - chi*ell**(1+theta)/(1+theta)
    return Y, w, r, T, c, i, u


def Modeldyn(theta0, params):
    '''
    This function takes vectors of endogenous and exogenous state variables
    along with a vector of 'jump' variables and returns values from the
    characterizing Euler equations.
    
    Inputs are:
        theta: a vector containng (Xpp, Xp, X, Yp, Y, Zp, Z) where:
            Xpp: value of capital in two periods
            Xp: value of capital in next period
            X: value of capital this period
            Yp: value of labor in next period
            Y: value of labor this period
            Zp: value of productivity in next period
            Z: value of productivity this period
        params: list of parameter values
    
    Output are:
        Euler: a vector of Euler equations written so that they are zero at the
            steady state values of X, Y & Z.  This is a 2x1 numpy array. 
    '''
    
    # unpack theat0
    (Xpp, Xp, X, Yp, Y, Zp, Z) = theta0
    
    # unpack params
    [alpha, beta, gamma, delta, chi, theta, tau, rho_z, sigma_z] = params
    
    # find definitions for now and next period
    ell = Y
    Y, w, r, T, c, i, u = Modeldefs(Xp, X, Y, Z, params)
    Yp, wp, rp, Tp, cp, ip, up = Modeldefs(Xpp, Xp, Yp, Zp, params)
    
    # Evaluate Euler equations
    E1 = (c**(-gamma)*(1-tau)*w) / (chi*ell**theta) - 1
    E2 = (c**(-gamma)) / (beta*cp**(-gamma)*(1 + (1-tau)*(rp - delta))) - 1
    
    return np.array([E1, E2])


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


"""That's where I start"""

def rouwen(rho, mu, step, num):
    '''
    Adapted from Lu Zhang and Karen Kopecky. Python by Ben Tengelsen.
    Construct transition probability matrix for discretizing an AR(1)
    process. This procedure is from Rouwenhorst (1995), which works
    well for very persistent processes.

    INPUTS:
    rho  - persistence (close to one)
    mu   - mean and the middle point of the discrete state space
    step - step size of the even-spaced grid
    num  - number of grid points on the discretized process

    OUTPUT:
    dscSp  - discrete state space (num by 1 vector)
    transP - transition probability matrix over the grid
    '''

    # discrete state space
    dscSp = np.linspace(mu -(num-1)/2*step, mu +(num-1)/2*step, num).T

    # transition probability matrix
    q = p = (rho + 1)/2.
    transP = np.array([[p**2, p*(1-q), (1-q)**2], \
                    [2*p*(1-p), p*q+(1-p)*(1-q), 2*q*(1-q)], \
                    [(1-p)**2, (1-p)*q, q**2]]).T


    while transP.shape[0] <= num - 1:

        # see Rouwenhorst 1995
        len_P = transP.shape[0]
        transP = p * np.vstack((np.hstack((transP, np.zeros((len_P, 1)))), np.zeros((1, len_P+1)))) \
                + (1 - p) * np.vstack((np.hstack((np.zeros((len_P, 1)), transP)), np.zeros((1, len_P+1)))) \
                + (1 - q) * np.vstack((np.zeros((1, len_P+1)), np.hstack((transP, np.zeros((len_P, 1)))))) \
                + q * np.vstack((np.zeros((1, len_P+1)), np.hstack((np.zeros((len_P, 1)), transP))))

        transP[1:-1] /= 2.


    # ensure columns sum to 1
    if np.max(np.abs(np.sum(transP, axis=1) - np.ones(transP.shape))) >= 1e-12:
        print('Problem in rouwen routine!')
        return None
    else:
        return transP.T, dscSp

# set up Markov approximation of AR(1) process using Rouwenhorst method
spread = 5.  # number of standard deviations above and below 0
znpts = 11
zstep = 4.*spread*sigma_z/(znpts-1)
# Markov transition probabilities, current z in cols, next z in rows
Pimat, zgrid = rouwen(rho_z, 0., zstep, znpts)

# discretize k
klow = .5*kbar
khigh = 1.5*kbar
knpts = 11
kgrid = np.linspace(klow, khigh, num = knpts)

# discretize ell
elllow = 0.0
ellhigh = 1.0
ellnpts = 11
ellgrid = np.linspace(elllow, ellhigh, num = ellnpts)

# initialize VF and PF
VF1 = np.ones((knpts, znpts)) * (-100)
VF1new = np.zeros((knpts, znpts))
PF1 = np.zeros((knpts, znpts))
JF1 = np.zeros((knpts, znpts))

# set VF iteration parameters
ccrit = 1.0E-10
maxit = 1000
count = 0
dist = 100.
maxwhile = 100000

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
                    r = alpha*kgrid[i1]**(alpha-1)*(np.exp(zgrid[i2])*ellgrid[i4] )**(1-alpha)
                    w = ((1-alpha)*kgrid[i1]**alpha*np.exp(zgrid[i2]*(1-alpha))) / ellgrid[i4] 
                    t = tau * (w * ellgrid[i4] + (r - delta)*kgrid[i1])
                    c = (1 - tau) * (w*ellgrid[i4] + (r-delta)*kgrid[i1]) + kgrid[i1] + t - kgrid[i3]
                    temp = -1/c**sigma_z
                    for i5 in range(0, znpts): # over z_t+1
                        temp = temp + beta * VF1[i3,i5] * Pimat[i2,i5] # check why it's not working
                    # print i, j, temp (keep all of them)
                    if np.iscomplex(temp):
                        temp = -1000000000
                    if np.isnan(temp):
                        temp = -1000000000
                    if temp > maxval:
                        maxval = temp
                        VF1new[i1, i2] = temp
                        PF1[i1, i2] = kgrid[i3]
                        JF1[i1, i2] = ellgrid[i4]

    # calculate the new distance measure, we use maximum absolute difference
    dist = np.amax(np.abs(VF1 - VF1new))
    if dist > ccrit:
        nconv = True
    # report the results of the current iteration
    print 'iteration: ', count, 'distance: ', dist
    
    # replace the value function with the new one
    VF1 = 1.*VF1new

print 'Converged after', count, 'iterations'
print 'Policy function at (', (knpts-1)/2, ',', (znpts-1)/2, ') should be', \
    kgrid[(knpts-1)/2], 'and is', PF1[(knpts-1)/2, (znpts-1)/2]
