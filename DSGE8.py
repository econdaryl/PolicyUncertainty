import numpy as np
from rouwen import rouwen
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def modeldefs(k, kp, z, *mparams):
    # This function takes the following inputs:
    #  k - capital stock today
    #  kp - capital stock next period
    #  z - productivity today
    # It returns the following:
    #  y - output
    #  r - rental rate on capital
    #  w - wage rate
    #  c - consumption
    #  i - investment
    #  u - household utility
    
    # find variables
    y = k**alpha*np.exp(z)            # y(t)
    r = alpha*y/k                     # r(t)
    w = (1-alpha)*y                   # w(t)
    c = w + r*k - kp                  # c(t)
    i = y - c                         # i(t)
    u = np.log(c)                     # u(t)
    return y, r, w, c, i, u


"""
Main Program
"""

# set parameter values
alpha = .33
beta = .95
rho = .9
sigma = .02
mparams = (alpha, beta, rho, sigma)

# find the steady state
A = beta*alpha
Kbar = A**(1/(1-alpha))
print 'Kbar: ', Kbar

# get other steady state values
zbar = 0.
ybar, rbar, wbar, cbar, ibar, ubar = modeldefs(Kbar, Kbar, zbar, *mparams)
print 'ybar: ', ybar
print 'rbar: ', rbar
print 'wbar: ', wbar
print 'cbar: ', cbar
print 'ibar: ', ibar
print 'ubar: ', ubar
print ' '

# set up grid for k
klow = .5*Kbar
khigh = 1.5*Kbar
knpts = 25
kgrid = np.linspace(klow, khigh, num = knpts)

# set up Markov approximation of AR(1) process using Rouwenhorst method
spread = 5.  # number of standard deviations above and below 0
znpts = 25
zstep = 4.*spread*sigma/(znpts-1)
# Markov transition probabilities, current z in cols, next z in rows
Pimat, zgrid = rouwen(rho, 0., zstep, znpts)

# initialize VF and PF
VF = np.zeros((knpts, znpts))
VFnew = np.zeros((knpts, znpts))
PF = np.zeros((knpts, znpts))

# set VF iteration parameters
ccrit = 1.0E-10
maxit = 1000
dist = 1.0E+99
iters = 0

# iterate to find true VF
while (dist > ccrit) and (iters < maxit):
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
    print 'iteration: ', iters, 'distance: ', dist
    # replace the value function with the new one
    VF = 1.*VFnew
    
print 'Converged after', iters, 'iterations'
print 'Policy function at (', (knpts-1)/2, ',', (znpts-1)/2, ') should be', \
    kgrid[(knpts-1)/2], 'and is', PF[(knpts-1)/2, (znpts-1)/2]
    
# create meshgrid
zmesh, kmesh = np.meshgrid(zgrid, kgrid)

# plot grid approximation of policy function
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(kmesh, zmesh, PF)
ax.view_init(30, 150)
plt.title('Policy Function Grid')
plt.xlabel('k(t)')
plt.ylabel('z(t)')
plt.show()

# caclulate and plot closed form solution for policy function
TruePF = np.zeros((knpts, znpts))
for i in range (0, knpts):
    for j in range(0, znpts):
        TruePF[i, j] = alpha*beta*kgrid[i]**alpha*np.exp(zgrid[j])
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(kmesh, zmesh, TruePF)
ax.view_init(30, 150)
plt.title('True Policy Function')
plt.xlabel('k(t)')
plt.ylabel('z(t)')
plt.show()