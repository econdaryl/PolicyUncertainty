#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 18:41:27 2018

@author: klp4
"""

import numpy as np
import pickle as pkl

from rouwen import rouwen

# copied from BMsimGSSA.py
def generateGSSA(k, z, args):
    from gssa import poly1
    (coeffs, pord, nx, ny, nz) = args
    An = np.exp(z)
    XZin = np.append(k, An)
    XYbasis = np.append(1., XZin)
    for i in range(1, pord):
        XYbasis = poly1(XZin, args)
    Xn = np.dot(XYbasis, coeffs)
    return Xn

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

# READ IN VALUES FROM LINERIZATION SOLUTION

# load Linearization coeffs
infile = open('BMsolveGSSA.pkl', 'rb')
(coeffs1, coeffs2, timesolve) = pkl.load(infile)
infile.close()

# create args lists
pord = 3
args1 = (coeffs1, pord, nx, ny, nz)
args2 = (coeffs2, pord, nx, ny, nz)

# SET UP GRIDS FOR PLOTS

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

Pf1GSSA = np.zeros((knpts, znpts))
Pf2GSSA = np.zeros((knpts, znpts))

for ik in range(0,knpts):
    for iz in range(0,znpts):
        Pf1GSSA[ik,iz] = generateGSSA(kgrid[ik], zgrid[iz], args1)
        Pf2GSSA[ik,iz] = generateGSSA(kgrid[ik], zgrid[iz], args2)
        
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# CREATE 3D PLOTS

# create meshgrid
kmesh, zmesh = np.meshgrid(kgrid, zgrid)

# plot grid approximation of Vf1
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(kmesh, zmesh, Pf1GSSA)
ax.view_init(30, 150)
plt.title('Pf1 GSSA')
plt.xlabel('k(t)')
plt.ylabel('z(t)')
plt.show()
plt.savefig('Results\GSSA\BMPf1GSSA.pdf')      

# plot grid approximation of Vf1
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(kmesh, zmesh, Pf1GSSA)
ax.view_init(30, 150)
plt.title('Pf2 GSSA')
plt.xlabel('k(t)')
plt.ylabel('z(t)')
plt.show()
plt.savefig('Results\GSSA\BMPf2GSSA.pdf')     
        