# -*- coding: utf-8 -*-
"""
Econ 581 Project

Created on Oct 31 2017 

Modeling Natural Disasters and Economic Responses

"""
import numpy as np

def rouwen(rho, mu, step, num):
    '''
    Adapted from Lu Zhang and Karen Kopecky. Python by Ben Tengelsen.
    Construct transition probability matrix for discretizing an AR(1)
    process. This procedure is from Rouwenhorst (1995), which works
    wl for very persistent processes.

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
        transP = p*np.vstack((np.hstack((transP,np.zeros((len_P,1)))), np.zeros((1, len_P+1)))) \
        + (1-p)*np.vstack((np.hstack((np.zeros((len_P, 1)), transP)), np.zeros((1, len_P+1)))) \
        + (1-q)*np.vstack((np.zeros((1, len_P+1)), np.hstack((transP, np.zeros((len_P, 1)))))) \
        + q*np.vstack((np.zeros((1, len_P+1)), np.hstack((np.zeros((len_P, 1)), transP))))

        transP[1:-1] /= 2.

    # ensure columns sum to 1
    if np.max(np.abs(np.sum(transP, axis=1) - np.ones(transP.shape))) >= 1e-12:
        print('Problem in rouwen routine!')
        return None
    else:
        return transP.T, dscSp

#import os
#retval = os.getcwd()
#print(retval)
#os.chdir('/Users/Austin/Desktop/Desktop/School/Econ 581/')
#help(os.chdir)

'''
Beginning of Code
Create a Impulse Response funciton to simulate a natural disaster and a destruction to capital. 
Create two models that have these IRF's under different government tax policies. 
The first tax policy is a low tax policy
The second tax policy is a high tax policy. 

'''
def Modeldefs(Xp, X, Y, Z, params):
  
    
    # unpack input vectors
    kp = Xp
    k = X
    l = Y
    z = Z
    
    # truncate labor if necessary
    if l > 0.9999:
        l = 0.9999
    elif l < 0.0001:
        l = 0.0001
    
    # unpack params
    [alpha, beta, gamma, delta, chi, theta, tau1, rho, sigma] = params
    
    # find definintion values
    GDP = (k**alpha*(np.exp(z)*l)**(1-alpha))
    #GDP = (1-tax)*(k**alpha*(np.exp(z)*l)**(1-alpha))
    # Y = Z*K**(alpha)*(np.exp(g + z)*L)**(1-alpha)
    w = (1-alpha)*GDP/l
    r = alpha*GDP/k
    #R = (alpha*Y/K) - delta
    T = tau1*(w*l + (r - delta)*k) 
    #T = tau*(w*l + (r - delta)*k) +  (tax*((k**alpha*(np.exp(z)*l)**(1-alpha))))
    c = (1-tau1)*(w*l + (1 + r - delta)*k) + k + T - kp
    i = GDP - c
    u = (c**(1-gamma)-1)/(1-gamma) - chi*l**(1+theta)/(1+theta)

    return GDP, w, r, c, i, u, T




def Modeldyn(theta0, params):    
    # unpack theat0
    (Xpp, Xp, X, Yp, Y, Zp, Z) = theta0
    
    # unpack params
    [alpha, beta, gamma, delta, chi, theta, rho, tau1, sigma] = params
    
    # find definitions for now and next period
    l = Y
    if l > 1:
        l = 0.9999
    elif l < 0.0001:
        l = 0.0001
    GDP, w, r, c, i, u, T = Modeldefs(Xp, X, Y, Z, params)
    GDPp, wp, rp, cp, ip, up, Tp = Modeldefs(Xpp, Xp, Yp, Zp, params)
    
    # Evaluate Euler equations
    E1 = (c**(-gamma)*w*(1-tau1)) / (chi*l**theta) - 1
    E2 = (c**(-gamma)) / (beta*cp**(-gamma)*(1 + (1-tau1)*(rp - delta))) - 1

    return np.array([E1, E2])


# import the modules from LinApp
from LinApp_FindSS import LinApp_FindSS
from LinApp_Deriv import LinApp_Deriv
from LinApp_Solve import LinApp_Solve
from LinApp_SSL import LinApp_SSL

# set parameter values
alpha = .35
beta = .96
#Beta = .96
gamma = 2.5
delta = .05
#Delta = .05
#g = .03 Labor Augmenting Technological Progress
chi = 10.
theta = 2.
rho = .9
sigma = .01
CapDes = -.498


'''
Tax Policies
'''
#Normal Tax Policy
tau1 = .25

# make parameter list to pass to functions
params1 = np.array([alpha, beta, gamma, delta, chi, theta, tau1, rho, sigma])


# set LinApp parameters
Zbar = np.array([0.])
nx = 1  # number of X variables
ny = 1  # number of Y variables
nz = 1  # number of Z variables
logX = 0  # 1 if log-linearizing, otherwise 0
Sylv = 0  # just set this to 0 for now.

# take a guess for steady state values of k and l
guessXY = np.array([.1, .25])

# find the steady state values using LinApp_FindSS
XYbar = LinApp_FindSS(Modeldyn, params1, guessXY, Zbar, nx, ny)
(kbar, lbar) = XYbar

# set up steady state input vector
theta0 = np.array([kbar, kbar, kbar, lbar, lbar, 0., 0.])

# check SS solution
check = Modeldyn(theta0, params1)
print ('check SS: ', check)
if np.max(np.abs(check)) > 1.E-6:
    print ('Have NOT found steady state')
    
# find the steady state values for the definitions
Ybar, wbar, rbar, cbar, ibar, ubar, Tbar = \
    Modeldefs(kbar, kbar, lbar, 0., params1)

# display all steady state values
print ('kbar:   ', kbar)
print ('lbar:   ', lbar)
print ('Ybar:   ', Ybar)
print ('wbar:   ', wbar)
print ('rbar:   ', rbar)
print ('cbar:   ', cbar)
print ('ibar:   ', ibar)
print ('ubar:   ', ubar)
print ('Tbar:   ', Tbar)


# find the derivatives matrices
[AA, BB, CC, DD, FF, GG, HH, JJ, KK, LL, MM, WW, TT] = \
    LinApp_Deriv(Modeldyn, params1, theta0, nx, ny, nz, logX)

# set value for NN    
NN = rho
    
# find the policy and jump function coefficients
PP, QQ, UU, RR, SS, VV = \
    LinApp_Solve(AA, BB, CC, DD, FF, GG, HH, JJ, KK, LL, \
                 MM, WW, TT, NN, Zbar, Sylv)
print ('P: ', PP)
print ('Q: ', QQ)
print ('R: ', RR)
print ('S: ', SS)


# set up Markov approximation of AR(1) process using Rouwenhorst method
spread = 5.  # number of standard deviations above and below 0
znpts = 11
zstep = 4.*spread*sigma/(znpts-1)

# Markov transition probabilities, current z in cols, next z in rows
Pimat, zgrid = rouwen(rho, 0., zstep, znpts)

# discretize k
klow = .7*kbar
khigh = 1.3*kbar
knpts = 11
kgrid = np.linspace(klow, khigh, num = knpts)

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# create meshgrid
kmesh, zmesh = np.meshgrid(kgrid, zgrid)

# initialize grids for policy and jump functions
PF = np.zeros_like(kmesh)
JF = np.zeros_like(kmesh)

# use linearized equations to fill in values
PF = PP*zmesh + QQ*zmesh + kbar
JF = RR*zmesh + SS*zmesh + lbar





#-----------------------------------------------------------------------------



# set number of observations
nobs = 250

# create a history of z's using equation (7)
zhist = np.zeros(nobs+1)
epshist = sigma*np.random.normal(0., 1., nobs+1)
#Create Impulse Response Function
epshist[40] = CapDes

'''
Creating a new variable called ts, which stand for Tax shifts after the natural disaster.
This is beginning of the after disaster tax change
'''
ts = 41
tau2 = .05
params2 = np.array([alpha, beta, gamma, delta, chi, theta, tau2, rho, sigma])


zhist[0] = epshist[0]
for t in range(1,nobs+1):
    zhist[t] = rho*zhist[t-1] + epshist[t]
    
# LinApp_SSL requires that Zhist be a 2-dimensional array
Zhist = np.reshape(zhist, (nobs+1, 1))

# Linapp_SSL also requires that starting values be arrays
k0 = np.array([[kbar]])
l0 = np.array([[lbar]])
khist = np.zeros(nobs)
khist, lhist =  LinApp_SSL(k0, Zhist ,XYbar, logX, \
    PP, QQ, UU, l0, RR, SS, VV)

# create histories of remaining variables
Yhist = np.zeros(nobs)
whist = np.zeros(nobs)
rhist = np.zeros(nobs)
chist = np.zeros(nobs)
ihist = np.zeros(nobs)
uhist = np.zeros(nobs)
Thist = np.zeros(nobs)
for t in range(0,nobs):
    Yhist[t], whist[t], rhist[t], chist[t], ihist[t], uhist[t], Thist[t] = \
        Modeldefs(khist[t+1], khist[t], lhist[t], zhist[t], params1)
        
# delete last observation
khist = khist[0:nobs]
zhist = zhist[0:nobs]
lhist = lhist[0:nobs]






#----------------------------------------------------------------------------

""" Graphs for the variables"""

# plot time series
time = range(0, nobs)

fig = plt.figure()
plt.subplot(2,1,1)
plt.plot(time, khist, color = 'orange', label='Capital')
plt.title('Capital Stock')

plt.subplot(2,1,2)
plt.plot(time, lhist, color='#1324dd', label='Labor')
plt.title('Labor')
plt.tight_layout()
plt.show()
#fig.savefig('/Users/Austin/Desktop/k_Grids')

fig = plt.figure()
plt.subplot(2,1,1)
plt.plot(time, zhist, color='#bcd613', label='z')
plt.title('Productivity')

plt.subplot(2,1,2)
plt.plot(time, Yhist, color='#12d6d2' ,label='Y')
plt.title('GDP')
plt.tight_layout()
plt.show()
#fig.savefig('/Users/Austin/Desktop/zy_Grids')

fig = plt.figure()
plt.subplot(2,1,1)
plt.plot(time, whist, color='#128ed6', label='w')
plt.title('Wage Rate')

plt.subplot(2,1,2)
plt.plot(time, rhist, color='#4471ed', label='r')
plt.title('Rental Rate')
plt.tight_layout()
plt.show()
#fig.savefig('/Users/Austin/Desktop/wr_Grids')

fig = plt.figure()
plt.subplot(2,1,1)
plt.plot(time, chist, color='#cc1e95', label='c')
plt.title('Consumption')

plt.subplot(2,1,2)
plt.plot(time, ihist, color='#ccc91d' ,label='i')
plt.title('Investment')
plt.tight_layout()
plt.show()
#fig.savefig('/Users/Austin/Desktop/ci_Grids')

fig = plt.figure()
plt.subplot(2,1,1)
plt.plot(time, uhist, color='#008dc9', label='u')
plt.title('Utility')
plt.tight_layout()
plt.show()
#fig.savefig('/Users/Austin/Desktop/u_Grids')

fig = plt.figure()
plt.subplot(2,1,2)
plt.plot(time, Thist, color='#008dc9', label='u')
plt.title('Government Revenue')
plt.tight_layout()
plt.show()

'''
These are the graphs showing that I am using a Linear approzimation method.
'''
# plot grid approximation of PF
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(kmesh, zmesh, PF)
ax.view_init(30, 240)
plt.title('PF')
plt.xlabel('k(t)')
plt.ylabel('z(t)')
plt.show()
#plt.savefig('HW06 PF Linear.png')

# plot grid approximation of JF
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(kmesh, zmesh, JF)
ax.view_init(30, 240)
plt.title('JF')
plt.xlabel('k(t)')
plt.ylabel('z(t)')
plt.show()
#plt.savefig('HW06 JF Linear.png')

#
#def PolSim(initial, nobs, ts, coeffs1, state1, params1, coeffs2, state2, \
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
#    coeffs1: list of (PP, QQ, UU, RR, SS, VV) under the baseline regime.
#    state1: numpy array of XYbar under the baseline regime.
#    params1: list of parameters under the baseline regime.
#    coeffs2: list of (PP2, QQ2, UU2, RR2, SS2, VV2) under the new regime.
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
#    # preallocate forecast histories
#    kfhist = np.zeros(nobs+1)
#    ellfhist = np.zeros(nobs)
#    zfhist = np.zeros(nobs)
#    Yfhist = np.zeros(nobs)
#    wfhist = np.zeros(nobs)
#    rfhist = np.zeros(nobs)
#    Tfhist = np.zeros(nobs)
#    cfhist = np.zeros(nobs)
#    ifhist = np.zeros(nobs)
#    ufhist = np.zeros(nobs)
#    
#    # upack simulation parameters
#    rho_z = params1[7] 
#    sigma_z = params1[8]
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
#        # inputs must be 1D numpy arrays and deviation from SS values
#        kin = np.array([khist[t] - kbar])
#        zin = np.array([zhist[t]])
#        k, ell = LinApp_Sim(kin, zin, PP, QQ, UU, RR, SS, VV)
#        # k and ell are deviations from SS values, so add these back.
#        # they are also 1D numpy arrays, so pull out the values rather than 
#        # use the arrays.
#        khist[t+1] = k + kbar
#        ellhist[t] = ell + ellbar
#        Yhist[t], whist[t], rhist[t], Thist[t], chist[t], ihist[t], \
#            uhist[t] = \
#            Modeldefs(khist[t+1], khist[t], ellhist[t], zhist[t], params)
#        
#        # get 1-period ahead forecasts
#        zfhist[t+1] = rho_z*zhist[t]
#        kin = np.array([khist[t+1] - kbar])
#        zin = np.array([zfhist[t]])
#        kf, ellf = LinApp_Sim(kin, zin, PP, QQ, UU, RR, SS, VV)
#        # k and ell are deviations from SS values, so add these back.
#        # they are also 1D numpy arrays, so pull out the values rather than 
#        # use the arrays.
#        kfhist[t+2] = kf + kbar
#        ellfhist[t] = ellf + ellbar
#        Yfhist[t+1], wfhist[t+1], rfhist[t+1], Tfhist[t+1], cfhist[t+1], \
#            ifhist[t], ufhist[t] = Modeldefs(kfhist[t+2], khist[t+1], \
#            ellfhist[t+1], zfhist[t+1], params)
#        
#    
#    for t in range(ts-1, nobs):
#        kin = np.array([khist[t] - kbar2])
#        zin = np.array([zhist[t]])
#        k, ell = LinApp_Sim(kin, zin, PP2, QQ2, UU2, RR2, SS2, VV2)
#        khist[t+1] = k + kbar2
#        ellhist[t] = ell + ellbar2
#        Yhist[t], whist[t], rhist[t], Thist[t], chist[t], ihist[t], \
#            uhist[t] = \
#            Modeldefs(khist[t+1], khist[t], ellhist[t], zhist[t], params2)
#        
#        if t < nobs - 1:
#            # get 1-period ahead forecasts
#            zfhist[t+1] = rho_z*zhist[t]
#            kin = np.array([khist[t+1] - kbar])
#            zin = np.array([zfhist[t]])
#            kf, ellf = LinApp_Sim(kin, zin, PP2, QQ2, UU2, RR2, SS2, VV2)
#            # k and ell are deviations from SS values, so add these back.
#            # they are also 1D numpy arrays, so pull out the values rather than 
#            # use the arrays.
#            kfhist[t+2] = kf + kbar
#            ellfhist[t] = ellf + ellbar
#            Yfhist[t+1], wfhist[t+1], rfhist[t+1], Tfhist[t+1], cfhist[t+1], \
#                ifhist[t], ufhist[t] = Modeldefs(kfhist[t+2], khist[t+1], \
#                ellfhist[t+1], zfhist[t+1], params)
#    
#    
#    return khist, ellhist, zhist, Yhist, whist, rhist, Thist, chist, ihist, \
#        uhist, kfhist, ellfhist, zfhist, Yfhist, wfhist, rfhist, Tfhist, \
#        cfhist, ifhist, ufhist
