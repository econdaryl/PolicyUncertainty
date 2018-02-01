#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 18:41:27 2018

@author: klp4
"""

import numpy as np
import pickle as pkl

from rouwen import rouwen

# copied from BMsimEX.py
def generateEX(k, z, args):
    
    '''
    This function generates values of k next period and ell this period given
    values for k and z this period.
    
    Inputs
    k - k this period
    z - z this period
    args - lists of linear coeffiecients and the steady state values.
    
    Outputs
    kp - k next period
    '''
    
    # unpack args
    (params, XYbar) = args
    [alpha, beta, tau, rho_z, sigma_z] = params

    kp = alpha*beta*(1-tau)*np.exp(z)*k**alpha
    
    return kp


# READ IN VALUES FROM STEADY STATE CALCULATIONS

# load steady state values and parameters
infile = open('BMfindss.pkl', 'rb')
(bar1, bar2, params1, params2, VFIparams) = pkl.load(infile)
infile.close()

# unpack
[kbar1, Ybar1, wbar1, rbar1, Tbar1, cbar1, ibar1, ubar1] = bar1
[kbar2, Ybar2, wbar2, rbar2, Tbar2, cbar2, ibar2, ubar2] = bar2
[alpha, beta, tau, rho_z, sigma_z] = params1
tau2 = params2[2]
(zbar, Zbar, NN, nx, ny, nz, logX, Sylv) = VFIparams


# create args lists
XYbar1 = kbar1
XYbar2 = kbar2
args1 = (params1, XYbar1)
args2 = (params2, XYbar2)

#  SET UP GRIDS FOR PLOTS

kfact= .05

# set up Markov approximation of AR(1) process using Rouwenhorst method
spread = 3.  # number of standard deviations above and below 0
znpts = 11
zstep = 4.*spread*sigma_z/(znpts-1)

# Markov transition probabilities, current z in cols, next z in rows
Pimat, zgrid = rouwen(rho_z, 0., zstep, znpts)

# discretize k
klow = (1-kfact)*kbar1
khigh = (1+kfact)*kbar1
knpts = 11
kgrid = np.linspace(klow, khigh, num = knpts)

# COMPUTE PF1 & PF2

Pf1EX = np.zeros((knpts, znpts))
Pf2EX = np.zeros((knpts, znpts))

for ik in range(0,knpts):
    for iz in range(0,znpts):
        Pf1EX[ik,iz] = generateEX(kgrid[ik], zgrid[iz], args1)
        Pf2EX[ik,iz] = generateEX(kgrid[ik], zgrid[iz], args2)
        
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# CREATE 3D PLOTS

# create meshgrid
kmesh, zmesh = np.meshgrid(kgrid, zgrid)

# plot grid approximation of Vf1
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(kmesh, zmesh, Pf1EX)
ax.view_init(30, 150)
plt.title('Pf1 Exact')
plt.xlabel('k(t)')
plt.ylabel('z(t)')
plt.show()
plt.savefig('BMPf1EX.pdf')      

# plot grid approximation of Vf1
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(kmesh, zmesh, Pf1EX)
ax.view_init(30, 150)
plt.title('Pf2 Exact')
plt.xlabel('k(t)')
plt.ylabel('z(t)')
plt.show()
plt.savefig('BMPf2EX.pdf')     
        