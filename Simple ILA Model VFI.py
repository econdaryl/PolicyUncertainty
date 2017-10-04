#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 16:08:19 2017

@author: klp4
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

a = np.array([[1.23,2],[1,2]])
# import the modules from LinApp
from LinApp_FindSS import LinApp_FindSS

def Modeldefs1(Xp, X, Y, Z, params):
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


def Modeldyn1(theta0, params):
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
    Y, w, r, T, c, i, u = Modeldefs1(Xp, X, Y, Z, params)
    Yp, wp, rp, Tp, cp, ip, up = Modeldefs1(Xpp, Xp, Yp, Zp, params)
    
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
XYbar = LinApp_FindSS(Modeldyn1, params, guessXY, Zbar, nx, ny)
(kbar, ellbar) = XYbar

# set up steady state input vector
theta0 = np.array([kbar, kbar, kbar, ellbar, ellbar, 0., 0.])

# check SS solution
check = Modeldyn1(theta0, params)
print ('check SS: ', check)
if np.max(np.abs(check)) > 1.E-6:
    print ('Have NOT found steady state')
    
# find the steady state values for the definitions
Ybar, wbar, rbar, Tbar, cbar, ibar, ubar = \
    Modeldefs1(kbar, kbar, ellbar, 0., params)

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

#value = np.array([[-6.547,-22.65,-66.89,-148.9,-240.7,-281.7,-238.4,-145.5,-64.00,-20.81,-5.641], \
#[-5.675,-21.50,-65.14,-146.9,-238.3,-279.2,-236.2,-143.7,-62.51,-19.87,-4.970], \
#[-5.111,-20.70,-63.83,-145.1,-236.2,-277.0,-234.2,-142.1,-61.39,-19.22,-4.530], \
#[-4.745,-19.94,-62.68,-143.5,-234.4,-275.1,-232.5,-140.7,-60.41,-18.61,-4.221], \
#[-4.387,-19.29,-61.68,-142.0,-232.8,-273.4,-230.9,-139.5,-59.54,-18.07,-3.939], \
#[-4.149,-18.71,-60.82,-140.8,-231.3,-271.9,-229.5,-138.3,-58.78,-17.59,-3.738], \
#[-3.921,-18.23,-60.10,-139.7,-230.0,-270.4,-228.2,-137.3,-58.13,-17.18,-3.555], \
#[-3.738,-17.82,-59.43,-138.6,-228.7,-269.1,-227.0,-136.4,-57.57,-16.83,-3.401], \
#[-3.582,-17.48,-58.78,-137.7,-227.5,-267.8,-225.9,-135.5,-57.02,-16.54,-3.269], \
#[-3.435,-17.14,-58.17,-136.8,-226.4,-266.7,-224.8,-134.7,-56.50,-16.25,-3.150], \
#[-3.313,-16.84,-57.60,-136.0,-225.3,-265.6,-223.8,-133.9,-56.00,-16.00,-3.042]])

#Vf1 = np.ones((knpts, znpts)) * (-100)

Vf1 = np.array([[-6.791, -25.196, -78.671, -180.857, -2.972, -349.678, -294.754, -177.271, -75.644, -23.312, -5.879], \
                [-5.918, -24.039, -76.906, -178.830, -2.948, -347.090, -292.421, -175.429, -74.134, -22.368,	-5.209], \
                [-5.355, -23.242, -75.586, -177.032, -2.926, -344.857, -290.405, -173.845, -73.003, -21.715,	-4.768], \
                [-4.988, -22.485, -74.429, -175.440, -2.907, -342.886, -288.621, -172.442, -72.013, -21.105,	-4.460], \
                [-4.630, -21.829, -73.412, -173.976, -2.891, -341.123, -287.019, -171.164, -71.138, -20.564,	-4.177], \
                [-4.392, -21.249, -72.536, -172.656, -2.875, -339.521, -285.577, -169.955, -70.364, -20.084,	-3.976], \
                [-4.164, -20.764, -71.797, -171.511, -2.862, -338.047, -284.260, -168.886, -69.697, -19.675,	-3.793], \
                [-3.982, -20.353, -71.123, -170.474, -2.849, -336.677, -283.034, -167.939, -69.124, -19.324,	-3.639], \
                [-3.826, -20.006, -70.470, -169.505, -2.837, -335.399, -281.883, -167.056, -68.572, -19.025,	-3.507], \
                [-3.678, -19.664, -69.859, -168.600, -2.825, -334.209, -280.792, -166.224, -68.045, -18.738,	-3.388], \
                [-3.556, -19.367, -69.285, -167.751, -2.814, -333.119, -279.735, -165.441, -67.544, -18.483,	-3.279]])
Vf1new = np.zeros((knpts, znpts))

Pf1 = np.zeros((knpts, znpts))
#Pf1 = np.array([[2.073, 2.073, 2.073, 2.488, 2.488, 2.488,  2.488,	2.488, 2.079, 2.073, 2.073], \
#			[2.073, 2.488, 2.488, 2.903, 2.903, 2.903, 2.903,	 2.903, 2.488, 2.488, 2.073], \
#			[2.488, 2.488, 2.903, 3.318, 3.318, 3.318, 3.318,	 3.318, 2.903, 2.903, 2.488], \
#			[2.488, 2.903, 3.318, 3.733, 3.733, 3.733, 3.733,	 3.733, 3.318, 2.903, 2.488], \
#			[2.903, 3.318, 3.733, 3.733, 4.147, 4.147, 4.147,	 4.147, 3.733, 3.318, 2.903], \
#			[3.318, 3.733, 4.147, 4.147, 4.562, 4.562, 4.562, 4.147, 4.147, 3.733, 3.318], \
#			[3.733, 4.147, 4.562, 4.562, 4.977, 4.977, 4.977,	 4.562, 4.562, 4.147, 3.733], \
#			[3.733, 4.562, 4.562, 4.977, 5.392, 5.395, 5.392,	 4.977, 4.977, 4.562, 3.733], \
#			[4.147, 4.977, 4.977, 5.392, 5.807, 5.807, 5.807,	 5.392, 4.977, 4.977, 4.147], \
#			[4.562, 4.977, 5.392, 5.807, 5.807, 6.221, 6.221,	 5.807, 5.392, 4.977, 4.562], \
#			[4.562, 5.392, 5.807, 6.221, 6.221, 6.221, 6.221,	 6.221, 5.807, 5.392, 4.562]])
Jf1 = np.zeros((knpts, znpts))

# set VF iteration parameters
#ccrit = 1.0E-20
ccrit = 1.0E-20
count = 0
dist = 100.
maxwhile = 3103 #is the convergent number

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
                    Y, w, r, T, c, i, u = Modeldefs1(kgrid[i3], kgrid[i1], \
                        ellgrid[i4], zgrid[i2], params)
                    temp = u
                    for i5 in range(0, znpts): # over z_t+1
                        temp = temp + beta * Vf1[i3,i5] * Pimat[i2,i5]
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
kgrid[int((knpts-1)/2)], 'and is', Pf1[int((knpts-1)/2)], int((znpts-1)/2))

# generate a history of Z's
nobs = 150
Zhist = np.zeros((nobs,1))
for t in range(1, nobs):
    Zhist[t,0] = rho_z*Zhist[t,0] + sigma_z*np.random.normal(0., 1.)
    
# put SS values and starting values into numpy vectors
XYbar = np.array([kbar, ellbar])
X0 = np.array([kbar])
Y0 = np.array([ellbar])


## CHANGE POLICY (PF1)
# see line 282 - 286 (done)

# set new tax rate
tau2 = .055

# make parameter list to pass to functions
params2 = np.array([alpha, beta, gamma, delta, chi, theta, tau2, rho_z, 
                    sigma_z])

# find new steady state
# use the old steady state values of k and ell for our guess
guessXY = XYbar

# find the steady state values using LinApp_FindSS
XYbar2 = LinApp_FindSS(Modeldyn1, params2, guessXY, Zbar, nx, ny)
(kbar2, ellbar2) = XYbar2

# set up steady state input vector
theta02 = np.array([kbar2, kbar2, kbar2, ellbar2, ellbar2, 0., 0.])

# check SS solution
check = Modeldyn1(theta02, params2)
print ('check SS: ', check)
if np.max(np.abs(check)) > 1.E-6:
    print ('Have NOT found steady state')
    
# find the steady state values for the definitions
Ybar2, wbar2, rbar2, Tbar2, cbar2, ibar2, ubar2 = \
    Modeldefs1(kbar2, kbar2, ellbar2, 0., params2)

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
# VF2 = np.zeros((knpts, znpts))
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
                    Y, w, r, T, c, i, u = Modeldefs1(kgrid[i3], kgrid[i1], \
                        ellgrid[i4], zgrid[i2], params)
                    temp = u
                    for i5 in range(0, znpts): # over z_t+1
                        temp = temp + beta * Vf2[i3,i5] * Pimat[i2,i5]
                    # print i, j, temp (keep all of them)
                    if np.iscomplex(temp):
                        temp = -1000000000
                    if np.isnan(temp):
                        temp = -1000000000
                    if temp > maxval:
                        maxval = temp
                        Vf2new[i1, i2] = temp
                        Pf2[i1, i2] = kgrid[i3]
                        Jf2[i1, i2] = ellgrid[i4]

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
    kgrid[int((knpts-1)/2)], 'and is', Pf2[int((knpts-1)/2), int((znpts-1)/2)])



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


coeffsPF1 = np.dot(np.linalg.inv(np.dot(X,np.transpose(X))),np.dot(X,YPF1))
coeffsPF1 = coeffsPF1.reshape((10,1))

coeffsJF1 = np.dot(np.linalg.inv(np.dot(X,np.transpose(X))),np.dot(X,YJF1))
coeffsJF1 = coeffsJF1.reshape((10,1))

coeffsPF2 = np.dot(np.linalg.inv(np.dot(X,np.transpose(X))),np.dot(X,YPF2))
coeffsPF2 = coeffsPF2.reshape((10,1))

coeffsJF2 = np.dot(np.linalg.inv(np.dot(X,np.transpose(X))),np.dot(X,YJF2))
coeffsJF2 = coeffsJF2.reshape((10,1))



def PolSim(initial, nobs, ts, PF1, JF1, state1, params1, PF2, JF2, state2, \
           params2):
    
    '''
    Generates a history of k & ell with a switch in regime in period ts.
    
    Inputs
    -----------    
    initial: list of values for k & z (k0, z0) in the first period.
    nobs: number of periods to simulate.
    ts: period in which the shift occurs.
    PF1: the 1st policy function with the tax rate = 0.05 
    JF1: the 1st jump function with the tax rate = 0.05
    state1: numpy array of XYbar under the baseline regime.
    params1: list of parameters under the baseline regime.
    PF2: the 2nd policy function with the tax rate = 0.055 
    JF2: the 2nd jump function with the tax rate = 0.055
    state2: numpy array of XYbar2 under the new regime.
    params2: list of parameters under the new regime.
    
    Returns
    --------
    khist: 2D-array, dtype=float
        nobs-by-1 matrix containing the values of k
    
    ellhist: 2D-array, dtype=float
        nobs-by-1 matrix vector containing the values of ell 
        
    zhist: 2D-array, dtype=float
        nobs-by-1 matrix vector containing the values of z 
    '''
    
    # preallocate histories
    khist = np.zeros(nobs+1)
    ellhist = np.zeros(nobs)
    zhist = np.zeros(nobs+1)
    Yhist = np.zeros(nobs)
    whist = np.zeros(nobs)
    rhist = np.zeros(nobs)
    Thist = np.zeros(nobs)
    chist = np.zeros(nobs)
    ihist = np.zeros(nobs)
    uhist = np.zeros(nobs)
    
    # upack simulation parameters
    rho_z = params1[7] 
    sigma_z = params1[8]
    
    # set starting values
    khist[0] = k0
    zhist[0] = 0.
    
    # unpack state1 and state2
    (kbar, ellbar) = XYbar
    (kbar2, ellbar2) = XYbar2
    
    # generate history of random shocks
    for t in range(0, nobs):
        zhist[t+1] = rho_z*zhist[t] + sigma_z*np.random.normal(0., 1.)
        Xvec = np.array([[1.0], [khist[t]], [khist[t]**2], [khist[t]**3], \
                         [zhist[t]], [zhist[t]**2], [zhist[t]**3], \
                         [khist[t]*zhist[t]], [khist[t]**2*zhist[t]], \
                         [khist[t]*zhist[t]**2]])  
        if t < ts:
            khist[t+1] = np.vdot(Xvec, coeffsPF1)
            ellhist[t] = np.vdot(Xvec, coeffsJF1)
            Yhist[t], whist[t], rhist[t], Thist[t], chist[t], ihist[t], uhist[t] \
                = Modeldefs1(khist[t+1], khist[t], ellhist[t], zhist[t], params1)
        else:
            khist[t+1] = np.vdot(Xvec, coeffsPF2)
            ellhist[t] = np.vdot(Xvec, coeffsJF2)
            Yhist[t], whist[t], rhist[t], Thist[t], chist[t], ihist[t], uhist[t] \
                = Modeldefs1(khist[t+1], khist[t], ellhist[t], zhist[t], params2)
            
        
        
    return khist, ellhist, zhist, Yhist, whist, rhist, Thist, chist, ihist, \
        uhist


# specify the number of simulations and observations per simulation
nsim = 100
nobs = 150

# specify the period policy shifts
ts = 20

# specify initial values
k0 = kbar
z0 = 0.
initial = (k0, z0)

# begin Monte Carlos

# run first simulation and store in Monte Carlo matrices
kmc, ellmc, zmc, Ymc, wmc, rmc, Tmc, cmc, imc, umc \
    = PolSim(initial, nobs, ts, Pf1, Jf1, XYbar, params, Pf2, Jf2, XYbar2, \
           params2)

for i in range(1, nsim):
    # run remaining simulations
    khist, ellhist, zhist, Yhist, whist, rhist, Thist, chist, ihist, uhist = \
        PolSim(initial, nobs, ts, Pf1, Jf1, XYbar, params, Pf2, Jf2, XYbar2, \
           params2)
    # stack results in Monte Carlo matrices
    kmc = np.vstack((kmc, khist))
    ellmc = np.vstack((ellmc, ellhist))
    zmc = np.vstack((zmc, zhist))
    Ymc = np.vstack((Ymc, Yhist))
    wmc = np.vstack((wmc, whist))
    rmc = np.vstack((rmc, rhist))
    Tmc = np.vstack((Tmc, Thist))
    cmc = np.vstack((cmc, chist))
    imc = np.vstack((imc, ihist))
    umc = np.vstack((umc, uhist))
    
# now sort the Monte Carlo matrices over the rows
kmc = np.sort(kmc, axis = 0)
ellmc = np.sort(ellmc, axis = 0)
zmc = np.sort(zmc, axis = 0)
Ymc = np.sort(Ymc, axis = 0)
wmc = np.sort(wmc, axis = 0)
rmc = np.sort(rmc, axis = 0)
Tmc = np.sort(Tmc, axis = 0)
cmc = np.sort(cmc, axis = 0)
imc = np.sort(imc, axis = 0)
umc = np.sort(umc, axis = 0)

# find the average values for each variable in each time period across 
# Monte Carlos
kavg = np.mean(kmc, axis = 0)
ellavg = np.mean(ellmc, axis = 0)
zavg = np.mean(zmc, axis = 0)
Yavg = np.mean(Ymc, axis = 0)
wavg = np.mean(wmc, axis = 0)
ravg = np.mean(rmc, axis = 0)
Tavg = np.mean(Tmc, axis = 0)
cavg = np.mean(cmc, axis = 0)
iavg = np.mean(imc, axis = 0)
uavg = np.mean(umc, axis = 0)

# find the rows for desired confidence bands
conf = .1
low = int(np.floor((conf/2)*nsim))
high = nsim - low

# find the upper and lower confidence bands for each variable
kupp = kmc[high,:]
ellupp = ellmc[high,:]
zupp = zmc[high,:]
Yupp = Ymc[high,:]
wupp = wmc[high,:]
rupp = rmc[high,:]
Tupp = Tmc[high,:]
cupp = cmc[high,:]
iupp = imc[high,:]
uupp = umc[high,:]

klow = kmc[low,:]
elllow = ellmc[low,:]
zlow = zmc[low,:]
Ylow = Ymc[low,:]
wlow = wmc[low,:]
rlow = rmc[low,:]
Tlow = Tmc[low,:]
clow = cmc[low,:]
ilow = imc[low,:]
ulow = umc[low,:]

'''
# find the predicted path with no randomness
# run first simulation and store in Monte Carlo matrices
params3 = np.array([alpha, beta, gamma, delta, chi, theta, tau, rho_z, 0])
params4 = np.array([alpha, beta, gamma, delta, chi, theta, tau2, rho_z, 0.])

kpred, ellpred, zpred, Ypred, wpred, rpred, Tpred, cpred, ipred, upred \
    = PolSim(initial, nobs, ts, PF1, JF1, XYbar, params3, PF2, JF2, XYbar2, \
           params4)
'''

# plot predicted with upper and lower bounds
plt.subplot(2,2,1)
plt.plot(range(kavg.size), kavg, 'k-',
         range(kupp.size), kupp, 'k:',
         range(klow.size), klow, 'k:')
plt.title('k')

plt.subplot(2,2,2)
plt.plot(range(ellavg.size), ellavg, 'k-',
         range(ellupp.size), ellupp, 'k:',
         range(elllow.size), elllow, 'k:')
plt.title('ell')

plt.subplot(2,2,3)
plt.plot(range(zavg.size), zavg, 'k-',
         range(zupp.size), zupp, 'k:',
         range(zlow.size), zlow, 'k:')
plt.title('z')

plt.subplot(2,2,4)
plt.plot(range(Yavg.size), Yavg, 'k-',
         range(Yupp.size), Yupp, 'k:',
         range(Ylow.size), Ylow, 'k:')
plt.title('Y')

# save high quality version to external file
plt.savefig('ILAfig1.eps', format='eps', dpi=2000)

plt.show()

plt.subplot(3,2,1)
plt.plot(range(wavg.size), wavg, 'k-',
         range(wupp.size), wupp, 'k:',
         range(wlow.size), wlow, 'k:')
plt.title('w')

plt.subplot(3,2,2)
plt.plot(range(ravg.size), ravg, 'k-',
         range(rupp.size), rupp, 'k:',
         range(rlow.size), rlow, 'k:')
plt.title('r')

plt.subplot(3,2,3)
plt.plot(range(Tavg.size), Tavg, 'k-',
         range(Tupp.size), Tupp, 'k:',
         range(Tlow.size), Tlow, 'k:')
plt.title('T')

plt.subplot(3,2,4)
plt.plot(range(cavg.size), cavg, 'k-',
         range(cupp.size), cupp, 'k:',
         range(clow.size), clow, 'k:')
plt.title('c')

plt.subplot(3,2,5)
plt.plot(range(iavg.size), iavg, 'k-',
         range(iupp.size), iupp, 'k:',
         range(ilow.size), ilow, 'k:')
plt.title('iT')

plt.subplot(3,2,6)
plt.plot(range(uavg.size), uavg, 'k-',
         range(uupp.size), uupp, 'k:',
         range(ulow.size), ulow, 'k:')
plt.title('u')

# save high quality version to external file
plt.savefig('ILAfig2.eps', format='eps', dpi=2000)

plt.show()


# plot avgicted with typical simulation
plt.subplot(2,2,1)
plt.plot(range(khist.size), khist, 'k-',
         range(kavg.size), kavg, 'r-')
plt.title('k')

plt.subplot(2,2,2)
plt.plot(range(ellhist.size), ellhist, 'k-',
         range(ellavg.size), ellavg, 'r-')
plt.title('ell')

plt.subplot(2,2,3)
plt.plot(range(zhist.size), zhist, 'k-',
         range(zavg.size), zavg, 'r-')
plt.title('z')

plt.subplot(2,2,4)
plt.plot(range(Yhist.size), Yhist, 'k-',
         range(Yavg.size), Yavg, 'r-')
plt.title('Y')

# save high quality version to external file
plt.savefig('ILAfig3.eps', format='eps', dpi=2000)

plt.show()

plt.subplot(3,2,1)
plt.plot(range(whist.size), whist, 'k-',
         range(wavg.size), wavg, 'r-')
plt.title('w')

plt.subplot(3,2,2)
plt.plot(range(rhist.size), rhist, 'k-',
         range(ravg.size), ravg, 'r-')
plt.title('r')

plt.subplot(3,2,3)
plt.plot(range(Thist.size), Thist, 'k-',
         range(Tavg.size), Tavg, 'r-')
plt.title('T')

plt.subplot(3,2,4)
plt.plot(range(chist.size), chist, 'k-',
         range(cavg.size), cavg, 'r-')
plt.title('c')

plt.subplot(3,2,5)
plt.plot(range(ihist.size), ihist, 'k-',
         range(iavg.size), iavg, 'r-')
plt.title('iT')

plt.subplot(3,2,6)
plt.plot(range(uhist.size), uhist, 'k-',
         range(uavg.size), uavg, 'r-')
plt.title('u')

# save high quality version to external file
plt.savefig('ILAfig4.eps', format='eps', dpi=2000)

plt.show()

'''
## Additional Work: plot grid approximation of policy functions and jump functions
# plot grid approximation of PF1
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(kmesh, zmesh, PF1)
ax.view_init(30, 150)
plt.title('PF1 Grid')
plt.xlabel('k(t)')
plt.ylabel('z(t)')
plt.show()

# plot grid approximation of PF2
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(kmesh, zmesh, PF2)
ax.view_init(30, 150)
plt.title('PF2 Grid')
plt.xlabel('k(t)')
plt.ylabel('z(t)')
plt.show()

# plot grid approximation of JF1
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(kmesh, zmesh, JF1)
ax.view_init(30, 150)
plt.title('JF1 Grid')
plt.xlabel('k(t)')
plt.ylabel('z(t)')
plt.show()

# plot grid approximation of JF2
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(kmesh, zmesh, JF2)
ax.view_init(30, 150)
plt.title('JF2 Grid')
plt.xlabel('k(t)')
plt.ylabel('z(t)')
plt.show()



## Get the polynomial approximations

PF1approx = 0.*PF1
PF2approx = 0.*PF2
JF1approx = 0.*JF1
JF2approx = 0.*JF2

for i in range(0,knpts):
    for j in range(0,znpts):
        temp = np.array([[1.0], [kmesh[i,j]], [kmesh[i,j]**2], \
                     [kmesh[i,j]**3], [zmesh[i,j]], [zmesh[i,j]**2], \
                     [zmesh[i,j]**3], [kmesh[i,j]*zmesh[i,j]], \
                     [zmesh[i,j]*kmesh[i,j]**2], [kmesh[i,j]*zmesh[i,j]**2]])
        PF1approx[i,j] = np.dot(np.transpose(coeffsPF1), temp)
        PF2approx[i,j] = np.dot(np.transpose(coeffsPF2), temp)
        JF1approx[i,j] = np.dot(np.transpose(coeffsJF1), temp)
        JF2approx[i,j] = np.dot(np.transpose(coeffsJF2), temp)
    
# plot polynomial approximations
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(kmesh, zmesh, PF1approx)
ax.view_init(30, 150)
plt.title('PF1 polynomial')
plt.xlabel('k(t)')
plt.ylabel('z(t)')
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(kmesh, zmesh, PF2approx)
ax.view_init(30, 150)
plt.title('PF2 polynomial')
plt.xlabel('k(t)')
plt.ylabel('z(t)')
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(kmesh, zmesh, JF1approx)
ax.view_init(30, 150)
plt.title('JF1 polynomial')
plt.xlabel('k(t)')
plt.ylabel('z(t)')
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(kmesh, zmesh, JF2approx)
ax.view_init(30, 150)
plt.title('JF2 polynomial')
plt.xlabel('k(t)')
plt.ylabel('z(t)')
plt.show()
'''