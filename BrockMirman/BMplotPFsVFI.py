#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 18:41:27 2018

@author: klp4
"""

import numpy as np
import pickle as pkl

from rouwen import rouwen

# copied from BMsimVFI.py
def generateVFI(k, z, args):
    
    '''
    This function generates values of k next period and ell this period given
    values for k and z this period.
    Inputs
    k - k this period
    z - z this period
    args - lists of linear coeffiecients and the steady state values.
    Outputs
    kp - k next period
    ell - ell this period
    '''
     # unpack args

    (coeffs, XYbar) = args
    (Vf, Pf, coeffsPF) = coeffs

    # inputs must be 1D numpy arrays and deviation from SS values
    Xvec = np.array([[1.0], [k], [k**2], [k**3], [z], [z**2], [z**3], \
                     [k*z], [k**2*z], [k*z**2]])
    
    kp = np.vdot(Xvec, coeffsPF)
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

# READ IN VALUES FROM LINERIZATION SOLUTION

# load VFI coeffs
infile = open('BMsolveVFI.pkl', 'rb')
(coeffs1, coeffs2, timesolve) = pkl.load(infile)
infile.close()

# create args lists
XYbar1 = kbar1
XYbar2 = kbar2
args1 = (coeffs1, XYbar1)
args2 = (coeffs2, XYbar2)

#---------------------------SET UP GRIDS FOR PLOTS-----------------------------

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

Pf1VFI = np.zeros((knpts, znpts))
Pf2VFI = np.zeros((knpts, znpts))

for ik in range(0,knpts):
    for iz in range(0,znpts):
        Pf1VFI[ik,iz] = generateVFI(kgrid[ik], zgrid[iz], args1)
        Pf2VFI[ik,iz] = generateVFI(kgrid[ik], zgrid[iz], args2)
        
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# CREATE 3D PLOTS

# create meshgrid
kmesh, zmesh = np.meshgrid(kgrid, zgrid)



"""try to get the grid approximation and plot them"""
# fit PF1 and PF2, Jf1 and JF2 with polynomials

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
YPF1 = Pf1VFI.flatten()
YPF2 = Pf2VFI.flatten()

coeffsPf1 = np.dot(np.linalg.inv(np.dot(X,np.transpose(X))),np.dot(X,YPF1))
coeffsPf1 = coeffsPf1.reshape((10,1))

coeffsPf2 = np.dot(np.linalg.inv(np.dot(X,np.transpose(X))),np.dot(X,YPF2))
coeffsPf2 = coeffsPf2.reshape((10,1))



## Get the polynomial approximations

Pf1VFIapprox = 0.*Pf1VFI
Pf2VFIapprox = 0.*Pf2VFI

for i in range(0,knpts):
    for j in range(0,znpts):
        temp = np.array([[1.0], [kmesh[i,j]], [kmesh[i,j]**2], \
                     [kmesh[i,j]**3], [zmesh[i,j]], [zmesh[i,j]**2], \
                     [zmesh[i,j]**3], [kmesh[i,j]*zmesh[i,j]], \
                     [zmesh[i,j]*kmesh[i,j]**2], [kmesh[i,j]*zmesh[i,j]**2]])
        Pf1VFIapprox[i,j] = np.dot(np.transpose(coeffsPf1), temp)
        Pf2VFIapprox[i,j] = np.dot(np.transpose(coeffsPf2), temp)


# plot grid approximation of Vf1
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(kmesh, zmesh, Pf1VFI)
ax.view_init(30, 150)
plt.title('Pf1 discrete')
plt.xlabel('k(t)')
plt.ylabel('z(t)')
plt.show()
plt.savefig('BMPf1VFIdiscrete.pdf')      

# plot grid approximation of Vf2
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(kmesh, zmesh, Pf2VFI)
ax.view_init(30, 150)
plt.title('Pf2 discrete')
plt.xlabel('k(t)')
plt.ylabel('z(t)')
plt.show()
plt.savefig('BMPf2VFIdiscrete.pdf')     
        
# plot polynomial approximation of Vf1
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(kmesh, zmesh, Pf1VFIapprox)
ax.view_init(30, 150)
plt.title('Pf1 polynomial')
plt.xlabel('k(t)')
plt.ylabel('z(t)')
plt.show()
plt.savefig('BMPf1VFIpolynomial.pdf')      

# plot polynomial approximation of Vf2
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(kmesh, zmesh, Pf2VFIapprox)
ax.view_init(30, 150)
plt.title('Pf2 polynomial')
plt.xlabel('k(t)')
plt.ylabel('z(t)')
plt.show()
plt.savefig('BMPf2VFIpolynomial.pdf')    

        