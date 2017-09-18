'''
Solutions to Econ581 homework #3
Value Function Iteration

Problem Set 1
Using the non-stochastic version of the discrete-time Ramsey-Cass-
Koopmans find both the value function and the transition function using 
a N-point grid on k.  Let k_1=.2*kbar and k_N=5*kbar.  Find the points 
between by creating equidistant points in the natural log of k.

Starting at an initial value of k that is 30 percent below the steady 
state value for k, use the transition function you found to simulate the 
transition of the economy from this initial state to the steady state.  
Plot the time paths for k,y,c,i,w & r.

Repeat the analysis above for a starting value of k that is 50 percent 
above the steady state.

Use the following parameters:
g=.025, n=.01, δ=.1, α=.33, θ=2.5, ρ=.05, N=21
g=.025, n=.01, δ=.1, α=.33, θ=2.5, ρ=.05, N=1001

Comment on how these results compare with those from homework #2.

Problem Set 2
Rather than rounding to the nearest grid point, fit the transition 
function with a 3rd-order polynomial, so that we have 
k_(t+1)= β_0+β_1 k_t^ +β_2 k_t^2+β_3 k_t^3.

Repeat the simulations from above.  Comment on how these results compare 
with those from problem set 1 and from homework #2.
'''

import numpy as np
import matplotlib.pyplot as plt

# set parameter values
#  model
g = .025
n = .01
delta = .1
alpha = .33
theta = 1.0
rho = .05
beta = (1+g)**(1-theta)*(1+n)/(1+rho)
# program
nobs = 50       # number of periods in simulatipon
start = .7      # starting value for simulation (proportional to kbar)
low = .2        # low end of grid (proportional to kbar)
high = 5        # high end of grid (proportional to kbar)
npts = 101      # number of points in the grid
gridsim = 0     # set to 1 to use grid, 0 to use polynomial fit

# calculate steady state values
kbar = (((1+rho)*(1+g)**theta-1+delta)/alpha)**(1/(alpha-1))
ybar = kbar**alpha
rbar = alpha*ybar/kbar
wbar = (1-alpha)*ybar
cbar = wbar + (1+rbar-delta)*kbar - (1+g)*(1+n)*kbar
ibar = ybar - cbar

# set up grid for k
klow = low*kbar   # low end of grid
khigh = high*kbar # high end of grid
kincr = np.log(khigh/klow)/(npts-1);
kvec = np.zeros(npts)
kvec[0] = klow
for i in range (1, npts):
    kvec[i] = kvec[i-1]*(1+kincr)


# find value function and transition function
converge = .001;
distance = 1.0;
maxwhile = 100;
count = 0;
value = np.zeros(npts);
newval = np.zeros(npts);
trans = np.zeros(npts);
while distance > converge:
    count = count + 1
    if count > maxwhile:
        break
    for i in range(0, npts):
        maxval = -10000000000;
        for j in range(0, npts):
            if theta == 1:
                temp = np.log(kvec[i]**alpha+(1-delta)*kvec[i]-kvec[j]*(1+g)
                   *(1+n))+beta*value[j];
            else:
                temp = ((kvec[i]**alpha+(1-delta)*kvec[i]-kvec[j]*(1+g)
                   *(1+n))**(1-theta)-1)/(1-theta)+beta*value[j];
            # print i, j, temp
            if np.iscomplex(temp):
               temp = -1000000000
            if np.isnan(temp):
                temp = -1000000000
            if temp > maxval:
                maxval = temp
                newval[i] = temp
                trans[i] = kvec[j]
    print count, distance
    distance = np.mean(np.abs(value/newval - 1.0))
    for i in range(0, npts):
        value[i] = newval[i]

# fit a polynomial
reg = np.stack((kvec**0, kvec, kvec**2, kvec**3))
XtX = np.dot(reg,np.transpose(reg))
XtY = np.dot(reg,trans)
coeffs = np.dot(np.linalg.inv(XtX),XtY)
tpoly = np.zeros(npts)
for i in range(0, npts):
    tpoly[i] = np.dot(np.stack((1, kvec[i], kvec[i]**2, kvec[i]**3)),coeffs)

# plot value function and transition function
plt.subplot(2, 1, 1)
plt.plot(kvec, value)
plt.title('Value Function')
plt.ylabel('V(t)')
plt.xlabel('k(t)')
plt.subplot(2, 1, 2)
plt.plot(kvec, trans, 'b.', label='grid')
plt.plot(kvec, tpoly, 'g', label='polynomial')
plt.plot(kvec, kvec, 'r--', label='45-degree')
plt.title('Transition Function')
plt.ylabel('k(t+1)')
plt.xlabel('k(t)')
plt.legend(loc=2)
plt.show()

# perform simulation
k = np.zeros(nobs+1)
y = np.zeros(nobs)
r = np.zeros(nobs)
w = np.zeros(nobs)
i = np.zeros(nobs)
c = np.zeros(nobs)
k[0] = start*kbar
for t in range(0, nobs):
    if gridsim:
        # perform simulation with grid
        tmp = abs(kvec-k[t]);
        idx = np.argmin(tmp); #index of closest value
        k[t+1] = trans[idx];
    else:
        # perform simulation with polynomial fit
        k[t+1] = np.dot(np.stack((1, k[t], k[t]**2, k[t]**3)),coeffs)
    y[t] = k[t]**alpha;
    r[t] = alpha*y[t]/k[t];
    w[t] = (1-alpha)*y[t];
    i[t] = k[t+1]*(1+g)-(1-delta)*k[t];
    c[t] = y[t]-i[t];
# remove final k (only needed to get i)
k = k[0:nobs];

# plot data
t = range(0, nobs)
plt.plot(t, y, label='y')
plt.plot(t, c, label='c')
plt.plot(t, i, label='i')
plt.plot(t, k, label='k')
plt.plot(t, r, label='r')
plt.plot(t, w, label='w')
plt.xlabel('time')
plt.legend(loc=9, ncol=6, bbox_to_anchor=(0., 1.02, 1., .102))
plt.show(3)