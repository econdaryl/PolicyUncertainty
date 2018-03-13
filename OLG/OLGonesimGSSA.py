# -*- coding: utf-8 -*-
"""
Created on Fri Mar  2 13:31:59 2018

@author: dblarsen
"""

import numpy as np
import pickle as pkl

def generateGSSA(k, z, args):
    from gssa import poly1
    
    (pord, nx, ny, nz, coeffs) = args
    polyargs = (pord, nx, ny, nz)
    An = np.exp(z)
    XZin = np.append(k, An)
    XYbasis = np.append(1., XZin)
    for i in range(1, pord):
        XYbasis = poly1(XZin, polyargs)
    XYout = np.dot(XYbasis, coeffs)
    Xn = XYout[0:nx]
    Y = XYout[nx:nx+ny]
    return Xn, Y


infile = open('OLGfindss.pkl', 'rb')
(bar1, bar2, params1, params2, LINparams) = pkl.load(infile)
infile.close()

# unpack
[k2bar1, k3bar1, k4bar1, l1bar1, l2bar1, l3bar1, Kbar1, \
    Lbar1, GDPbar1, wbar1, rbar1, T4bar1, Bbar1, c1bar1, c2bar1, c3bar1, \
    c4bar1, Cbar1, Ibar1, u1bar1, u2bar1, u3bar1, u4bar1] = bar1
[k2bar2, k3bar2, k4bar2, l1bar2, l2bar2, l3bar2, Kbar2, \
    Lbar2, GDPbar2, wbar2, rbar2, T4bar2, Bbar2, c1bar2, c2bar2, c3bar2, \
    c4bar2, Cbar2, Ibar2, u1bar2, u2bar2, u3bar2, u4bar2] = bar2
[alpha, beta, gamma, delta, chi, theta, tau, rho, \
    sigma, pi2, pi3, pi4, f1, f2, f3, nx, ny, nz] = params1
tau2 = params2[6]
(zbar, Zbar, NN, nx, ny, nz, logX, Sylv) = LINparams

T = 120

infile = open('OLGsolveGSSA.pkl', 'rb')
(coeffs1, coeffs2, timesolve) = pkl.load(infile)
infile.close()

pord = 3
args1 = (pord, nx, ny, nz, coeffs1)
args2 = (pord, nx, ny, nz, coeffs2)

Z = np.zeros((T,nz))
for t in range(1,T):
    Z[t,:] = rho*Z[t-1] + np.random.randn(1)*sigma

kbar1 = (k2bar1, k3bar1, k4bar1)
lbar1 = (l1bar1, l2bar1, l3bar1)
Xstart = kbar1
X = np.zeros((T+1, nx))
Y = np.zeros((T, ny))
A = np.exp(Z)
x = np.zeros((T,(pord*5)))
X[0, :], Y[0, :] = generateGSSA(Xstart, Z[0], args1)
for t in range(1,T+1):
    X[t, :], Y[t-1, :] = generateGSSA(X[t-1, :], Z[t-1, :], args1)
X1 = X[0:T, :]

    
import matplotlib.pyplot as plt
timeperiods = np.asarray(range(0,T))
plt.subplot(2,1,1)
plt.plot(timeperiods, X1, label='X')
plt.axhline(y=k2bar1, color='k')
plt.axhline(y=k3bar1, color='w')
plt.axhline(y=k4bar1, color='b')
plt.subplot(2,1,2)
plt.plot(timeperiods, Y, label='Y')
plt.axhline(y=l1bar1, color='k')
plt.axhline(y=l2bar1, color='w')
plt.axhline(y=l3bar1, color='b')
plt.xlabel('time')
plt.legend(loc=9, ncol=(nx+ny))
plt.show()
    
