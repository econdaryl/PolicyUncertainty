# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 14:52:30 2017

@author: Anthony Yim
"""
"""
The Purpose of this code is to generate a Value Function matrix, a Policy 
Function , and a Jump Function matrix.

Once I finish coding for VF, PF, and JF, then I will put the code into the 
master python fileSimple ILA Model VFY.py

In other words, I can play around with the code here.
"""

"""Value Function"""

"""
Notice that there there should be
2 VFs
2 PFs
2 JFs
"""
import numpy as np
knpts = 25 # just a quasi code
znpts = 25

# find value function and transition function
converge = .001
distance = 1.0
maxwhile = 100
count = 0
value = np.zeros(knpts, znpts)
newval = np.zeros(knpts, znpts)

# set up the 2-dimensional arrays 
PF1 = np.zeros(knpts, znpts)
JF1 = np.zeros(knpts, znpts)

# Discretize z
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
znpts = 25
zstep = 4.*spread*sigma/(znpts-1)
# Markov transition probabilities, current z in cols, next z in rows
Pimat, zgrid = rouwen(rho, 0., zstep, znpts)

    
# discretize k
klow = .5*Kbar
khigh = 1.5*Kbar
knpts = 25
kgrid = np.linspace(klow, khigh, num = knpts)

# discretize ell
elllow = 0.0
ellhigh = 1.0
ellnpts = 25
ellgrid = np.linspace(elllow, ellhigh, num = ellnpts)



# initialize VF and PF

VF = np.zeros((knpts, znpts))
VFnew = np.zeros((knpts, znpts))
PF = np.zeros((knpts, znpts))

# set VF iteration parameters
ccrit = 1.0E-5
maxit = 1000
dist = 1.0E+99
iters = 0



# iterate to find true VF
nconv = True
while (nconv):
    # set new value function to zeros
    VFnew.fill(0.0)
    # update increment counter
    iters = iters + 1
    # search over k(t) - i index, and z(t) - j index
    for i in range (0, knpts):
        for j in range(0, znpts):
            # set initial maximum to a large negative number
            maxval = -1.0E+98
            # search over k(t+1) - m index
            for m in range(0, knpts):
                # get current period utility
                yout, rat, wag, con, inv, u =  \
                    modeldefs(kgrid[i], kgrid[m], zgrid[j], *mparams)
                # get expected value
                val = u
                # weighted sum over possible values of z(t+1) - n index
                for n in range (0, znpts):
                    # sum over all possible value of z(t+1) with Markov probs
                    val = val + beta*Pimat[n, j]*VF[m, n]
                    # if this exceeds previous maximum do replacements
                if val > maxval:
                    maxval = val          # new maximum value
                    VFnew[i, j] = val     # write this to appropriate cell in VFnew
                    PF[i, j] = kgrid[m]   # write value of k(t+1) into PF
                    
    # calculate the new distance measure, we use maximum absolute difference
    dist = np.amax(np.abs(VF - VFnew))
    # report the results of the current iteration
    print ('iteration: ', iters, 'distance: ', dist)
    # replace the value function with the new one
    VF = 1.*VFnew
"""Policy Function"""


"""Junp Function"""