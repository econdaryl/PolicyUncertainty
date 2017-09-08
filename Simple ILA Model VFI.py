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

# Alternative approach
#print (count, distance)
#distance = np.mean(np.abs(value/newval - 1.0))
#for i in range(0, npts):
#value[i] = newval[i]


#
## generate a history of Z's
#nobs = 250
#Zhist = np.zeros((nobs,1))
#for t in range(1, nobs):
#    Zhist[t,0] = rho_z*Zhist[t,0] + sigma_z*np.random.normal(0., 1.)
#    
## put SS values and starting values into numpy vectors
#XYbar = np.array([kbar, ellbar])
#X0 = np.array([kbar])
#Y0 = np.array([ellbar])
#
#
#
### CHANGE POLICY (PF1)
## see line 282 - 286
#
## set new tax rate
#tau2 = .055
#
## make parameter list to pass to functions
#params2 = np.array([alpha, beta, gamma, delta, chi, theta, tau2, rho_z, 
#                    sigma_z])
#
## find new steady state
## use the old steady state values of k and ell for our guess
#guessXY = XYbar
#
## find the steady state values using LinApp_FindSS
#XYbar2 = LinApp_FindSS(Modeldyn, params2, guessXY, Zbar, nx, ny)
#(kbar2, ellbar2) = XYbar2
#
## set up steady state input vector
#theta02 = np.array([kbar2, kbar2, kbar2, ellbar2, ellbar2, 0., 0.])
#
## check SS solution
#check = Modeldyn(theta02, params2)
#print ('check SS: ', check)
#if np.max(np.abs(check)) > 1.E-6:
#    print ('Have NOT found steady state')
#    
## find the steady state values for the definitions
#Ybar2, wbar2, rbar2, Tbar2, cbar2, ibar2, ubar2 = \
#    Modeldefs(kbar2, kbar2, ellbar2, 0., params2)
#
## display all steady state values
#print ('kbar:   ', kbar2)
#print ('ellbar: ', ellbar2)
#print ('Ybar:   ', Ybar2)
#print ('wbar:   ', wbar2)
#print ('rbar:   ', rbar2)
#print ('Tbar:   ', Tbar2)
#print ('cbar:   ', cbar2)
#print ('ibar:   ', ibar2)
#print ('ubar:   ', ubar2)
#
## Solve for new policy function using VFI
#
## caclulate and plot closed form solution for policy function
#TruePF = np.zeros((kpts, zpts))
#for i in range (0, kpts):
#    for j in range(0, zpts):
#        TruePF[i, j] = alpha*beta*kgrid[i1]**alpha*np.exp(zgrid[i2])
#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#ax.plot_surface(kmesh, zmesh, TruePF)
#ax.view_init(30, 150)
#plt.title('True Policy Function')
#plt.xlabel('k(t)')
#plt.ylabel('z(t)')
#plt.show()
#
## get PF2 and JF2
## find value function and transition function
#nconv = True
#while (conv):
#    #count = count + 1
#    #if count > maxwhile:
#        #break
#    for i1 in range(0, kpts): # over kt
#        for i2 in range(0, zpts): # over zt, searching the value for the stochastic shock
#            maxval = -100000000000
#            for i3 in range(0, kpts): # over k_t+1
#                for i4 in range(0, kpts): # over ell_t
#                    temp = 0
#                    for i5 in range(0, zpts): # over z_t+1
#                        r = alpha*kgrid[i1]**(alpha-1)*(np.exp(zgrid[i2])*ell)**(1-alpha)
#                        w = ((1-alpha)*kgrid[i1]**alpha*np.exp(zgrid[i2]*(1-alpha))) / ellgrid[i4] 
#                        t = tau2 * (w * ellgrid[i4] + (r - delta)*kgrid[i1])
#                        c = (1 - tau) * (w*ellgrid[i4] + (r-delta)*kgrid[i1]) + kgrid[i1] + t - kgrid[i3]
#                        temp = temp + (-1/c**sigma_z) + beta * VF1[i3,i5] * Pimat[i2,i5]) # check why it's not working
#                        # V(k;theta) = (((r - delta)*(1-tau)+1)/c**sigma_z) 
#                        # u_c(C) = 1/c**sigma_z
#                        
#                        #if theta == 1:
#                            #temp = (c**(1-gamma)/(1-gamma) - chi*ell**(1+theta)/(1+theta) 
#                            #+ beta*value[i3]) 
#                            #temp = (c**(1-gamma)/(1-gamma) - chi*ell**(1+theta)/(1+theta) 
#                            #       + beta*(((r - delta)*(1-tau)+1)/c))
#                            #else:
#                            #    temp = (((kvec[i]**alpha+(1-delta)*kvec[i]-kvec[j]*(1+g)
#                            #              *(1+n))**(1-theta)-1)/(1-theta)+beta*value[j])
#                            #    temp = (((kvec[i]**alpha+(1-delta)*kvec[i]-kvec[j]*(1+g)
#                            #              *(1+n))**(1-theta)-1)/(1-theta)+beta*
#                                # print i, j, temp (keep all of them)
#                                if np.iscomplex(temp):
#                                    temp = -1000000000
#                                if np.isnan(temp):
#                                    temp = -1000000000
#                                if temp > maxval:
#                                    maxval = temp
#                                    VF2[i1, i2] = temp # or VF?
#                                    PF2[i1, i2] = kgrid[i3]
#                                    JF2[i1, i2] = ellgrid[i4]
#
#if sum(VF1 - VF2): #2
#    nconv = False
## report the results of the current iteration
#print 'iteration: ', iters, 'distance: ', dist
#
## replace the value function with the new one
#VF1 = 1.*VF2
#                                 
##print (count, distance)
##distance = np.mean(np.abs(value/newval - 1.0))
##for i in range(0, npts):
##value[i] = newval[i]
#
#
#def PolSim(initial, nobs, ts, PF1, JF1, state1, params1, PF2, JF2, state2, \
#           params2):
#    from LinApp_Sim import LinApp_Sim
#    '''
#    Generates a history of k & ell with a switch in regime in period ts.
#    
#    Inputs
#    -----------    
#    initial: list of values for k & z (k0, z0) in the first period.
#    nobs: number of periods to simulate.
#    ts: period in which the shift occurs.
#    PF1: the 1st policy function with the tax rate = 0.05 
#    JF1: the 1st jump function with the tax rate = 0.05
#    state1: numpy array of XYbar under the baseline regime.
#    params1: list of parameters under the baseline regime.
#    PF2: the 2nd policy function with the tax rate = 0.055 
#    JF2: the 2nd jump function with the tax rate = 0.055
#    state2: numpy array of XYbar2 under the new regime.
#    params2: list of parameters under the new regime.
#    
#    Returns
#    --------
#    khist: 2D-array, dtype=float
#        nobs-by-1 matrix containing the values of k
#    
#    ellhist: 2D-array, dtype=float
#        nobs-by-1 matrix vector containing the values of ell 
#        
#    zhist: 2D-array, dtype=float
#        nobs-by-1 matrix vector containing the values of z 
#    '''
#    
#    # preallocate histories
#    khist = np.zeros(nobs+1)
#    ellhist = np.zeros(nobs)
#    zhist = np.zeros(nobs)
#    Yhist = np.zeros(nobs)
#    whist = np.zeros(nobs)
#    rhist = np.zeros(nobs)
#    Thist = np.zeros(nobs)
#    chist = np.zeros(nobs)
#    ihist = np.zeros(nobs)
#    uhist = np.zeros(nobs)
#    
#    # set starting values
#    khist[0] = k0
#    zhist[0] = z0
#    
#    # unpack state1 and state2
#    (kbar, ellbar) = XYbar
#    (kbar2, ellbar2) = XYbar2
#    
#    # generate history of random shocks
#    for t in range(1, nobs):
#        zhist[t] = rho_z*zhist[t] + sigma_z*np.random.normal(0., 1.)
#        
#    # generate histories for k and ell for the first ts-1 periods
#    for t in range(0, ts-1):
#        # is this actually k_t-1?
#        khist[t] = beta * ( (r[t+1]-delta)*(1 - tau) / c[t+1]*sigma_z + 1) # i dont think that's correct
#        
#    for t in range(ts-1, nobs):
#        ########
#        ellhist[t] = (w[t]*(1-tau)/ (chi*c[t])**(1/theta)
#        
#    return khist, ellhist, zhist, Yhist, whist, rhist, Thist, chist, ihist, \
#        uhist
#
#
## specify the number of simulations and observations per simulation
#nsim = 10000
#nobs = 120
#
## specify the period policy shifts
#ts = 20
#
## specify initial values
#k0 = kbar
#z0 = 0.
#initial = (k0, z0)
#
## set up coefficient lists
#coeffs1 = (PP, QQ, UU, RR, SS, VV)
#coeffs2 = (PP2, QQ2, UU2, RR2, SS2, VV2)
#
## begin Monte Carlos
#
## run first simulation and store in Monte Carlo matrices
#kmc, ellmc, zmc, Ymc, wmc, rmc, Tmc, cmc, imc, umc \
#    = PolSim(initial, nobs, ts, coeffs1, XYbar, params, coeffs2, XYbar2, \
#             params2)
#
#for i in range(1, nsim):
#    # run remaining simulations
#    khist, ellhist, zhist, Yhist, whist, rhist, Thist, chist, ihist, uhist = \
#        PolSim(initial, nobs, ts, coeffs1, XYbar, params, coeffs2, XYbar2, \
#               params2)
#    # stack results in Monte Carlo matrices
#    kmc = np.vstack((kmc, khist))
#    ellmc = np.vstack((ellmc, ellhist))
#    zmc = np.vstack((zmc, zhist))
#    Ymc = np.vstack((Ymc, Yhist))
#    wmc = np.vstack((wmc, whist))
#    rmc = np.vstack((rmc, rhist))
#    Tmc = np.vstack((Tmc, Thist))
#    cmc = np.vstack((cmc, chist))
#    imc = np.vstack((imc, ihist))
#    umc = np.vstack((umc, uhist))
#    
## now sort the Monte Carlo matrices over the rows
#kmc = np.sort(kmc, axis = 0)
#ellmc = np.sort(ellmc, axis = 0)
#zmc = np.sort(zmc, axis = 0)
#Ymc = np.sort(Ymc, axis = 0)
#wmc = np.sort(wmc, axis = 0)
#rmc = np.sort(rmc, axis = 0)
#Tmc = np.sort(Tmc, axis = 0)
#cmc = np.sort(cmc, axis = 0)
#imc = np.sort(imc, axis = 0)
#umc = np.sort(umc, axis = 0)
#
## find the average values for each variable in each time period across 
## Monte Carlos
#kavg = np.mean(kmc, axis = 0)
#ellavg = np.mean(ellmc, axis = 0)
#zavg = np.mean(zmc, axis = 0)
#Yavg = np.mean(Ymc, axis = 0)
#wavg = np.mean(wmc, axis = 0)
#ravg = np.mean(rmc, axis = 0)
#Tavg = np.mean(Tmc, axis = 0)
#cavg = np.mean(cmc, axis = 0)
#iavg = np.mean(imc, axis = 0)
#uavg = np.mean(umc, axis = 0)
#
## find the rows for desired confidence bands
#conf = .1
#low = int(np.floor((conf/2)*nsim))
#high = nsim - low
#
## find the upper and lower confidence bands for each variable
#kupp = kmc[high,:]
#ellupp = ellmc[high,:]
#zupp = zmc[high,:]
#Yupp = Ymc[high,:]
#wupp = wmc[high,:]
#rupp = rmc[high,:]
#Tupp = Tmc[high,:]
#cupp = cmc[high,:]
#iupp = imc[high,:]
#uupp = umc[high,:]
#
#klow = kmc[low,:]
#elllow = ellmc[low,:]
#zlow = zmc[low,:]
#Ylow = Ymc[low,:]
#wlow = wmc[low,:]
#rlow = rmc[low,:]
#Tlow = Tmc[low,:]
#clow = cmc[low,:]
#ilow = imc[low,:]
#ulow = umc[low,:]
#
## plot
#plt.subplot(2,2,1)
#plt.plot(range(kavg.size), kavg, 'k-',
#         range(kupp.size), kupp, 'k:',
#         range(klow.size), klow, 'k:')
#plt.title('k')
#
#plt.subplot(2,2,2)
#plt.plot(range(ellavg.size), ellavg, 'k-',
#         range(ellupp.size), ellupp, 'k:',
#         range(elllow.size), elllow, 'k:')
#plt.title('ell')
#
#plt.subplot(2,2,3)
#plt.plot(range(zavg.size), zavg, 'k-',
#         range(zupp.size), zupp, 'k:',
#         range(zlow.size), zlow, 'k:')
#plt.title('z')
#
#plt.subplot(2,2,4)
#plt.plot(range(Yavg.size), Yavg, 'k-',
#         range(Yupp.size), Yupp, 'k:',
#         range(Ylow.size), Ylow, 'k:')
#plt.title('Y')
#
## save high quality version to external file
#plt.savefig('ILAfig1.eps', format='eps', dpi=2000)
#
#plt.show()
#
#plt.subplot(3,2,1)
#plt.plot(range(wavg.size), wavg, 'k-',
#         range(wupp.size), wupp, 'k:',
#         range(wlow.size), wlow, 'k:')
#plt.title('w')
#
#plt.subplot(3,2,2)
#plt.plot(range(ravg.size), ravg, 'k-',
#         range(rupp.size), rupp, 'k:',
#         range(rlow.size), rlow, 'k:')
#plt.title('r')
#
#plt.subplot(3,2,3)
#plt.plot(range(Tavg.size), Tavg, 'k-',
#         range(Tupp.size), Tupp, 'k:',
#         range(Tlow.size), Tlow, 'k:')
#plt.title('T')
#
#plt.subplot(3,2,4)
#plt.plot(range(cavg.size), cavg, 'k-',
#         range(cupp.size), cupp, 'k:',
#         range(clow.size), clow, 'k:')
#plt.title('c')
#
#plt.subplot(3,2,5)
#plt.plot(range(iavg.size), iavg, 'k-',
#         range(iupp.size), iupp, 'k:',
#         range(ilow.size), ilow, 'k:')
#plt.title('iT')
#
#plt.subplot(3,2,6)
#plt.plot(range(uavg.size), uavg, 'k-',
#         range(uupp.size), uupp, 'k:',
#         range(ulow.size), ulow, 'k:')
#plt.title('u')
#
## save high quality version to external file
#plt.savefig('ILAfig2.eps', format='eps', dpi=2000)
#
#plt.show()
#
## plot
#plt.subplot(2,2,1)
#plt.plot(range(khist.size), khist, 'k-',
#         range(kavg.size), kavg, 'r-')
#plt.title('k')
#
#plt.subplot(2,2,2)
#plt.plot(range(ellhist.size), ellhist, 'k-',
#         range(ellavg.size), ellavg, 'r-')
#plt.title('ell')
#
#plt.subplot(2,2,3)
#plt.plot(range(zhist.size), zhist, 'k-',
#         range(zavg.size), zavg, 'r-')
#plt.title('z')
#
#plt.subplot(2,2,4)
#plt.plot(range(Yhist.size), Yhist, 'k-',
#         range(Yavg.size), Yavg, 'r-')
#plt.title('Y')
#
## save high quality version to external file
#plt.savefig('ILAfig3.eps', format='eps', dpi=2000)
#
#plt.show()
#
#plt.subplot(3,2,1)
#plt.plot(range(whist.size), whist, 'k-',
#         range(wavg.size), wavg, 'r-')
#plt.title('w')
#
#plt.subplot(3,2,2)
#plt.plot(range(rhist.size), rhist, 'k-',
#         range(ravg.size), ravg, 'r-')
#plt.title('r')
#
#plt.subplot(3,2,3)
#plt.plot(range(Thist.size), Thist, 'k-',
#         range(Tavg.size), Tavg, 'r-')
#plt.title('T')
#
#plt.subplot(3,2,4)
#plt.plot(range(chist.size), chist, 'k-',
#         range(cavg.size), cavg, 'r-')
#plt.title('c')
#
#plt.subplot(3,2,5)
#plt.plot(range(ihist.size), ihist, 'k-',
#         range(iavg.size), iavg, 'r-')
#plt.title('iT')
#
#plt.subplot(3,2,6)
#plt.plot(range(uhist.size), uhist, 'k-',
#         range(uavg.size), uavg, 'r-')
#plt.title('u')
#
## save high quality version to external file
#plt.savefig('ILAfig4.eps', format='eps', dpi=2000)
#
#plt.show()
