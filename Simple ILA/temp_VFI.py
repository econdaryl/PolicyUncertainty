# My Code(VFI) - need to modify the code to calculate the forecasting error for each variable

import numpy as np

def PolSim(initial, nobs, ts, coeffsPF1, coeffsJF1, state1, params1, \
           coeffsPF2, coeffsJF2, state2, \
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

    # preallocate forecast histories
    kfhist = np.zeros(nobs+1)
    ellfhist = np.zeros(nobs)
    zfhist = np.zeros(nobs+1)
    Yfhist = np.zeros(nobs)
    wfhist = np.zeros(nobs)
    rfhist = np.zeros(nobs)
    Tfhist = np.zeros(nobs)
    cfhist = np.zeros(nobs)
    ifhist = np.zeros(nobs)
    ufhist = np.zeros(nobs)
    
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
                = Modeldefs(khist[t+1], khist[t], ellhist[t], zhist[t], params1)
                
        else:
            khist[t+1] = np.vdot(Xvec, coeffsPF2)
            ellhist[t] = np.vdot(Xvec, coeffsJF2)
            Yhist[t], whist[t], rhist[t], Thist[t], chist[t], ihist[t], uhist[t] \
                = Modeldefs(khist[t+1], khist[t], ellhist[t], zhist[t], params2)

    # get 1-period ahead forecasts
    for t in range(0, nobs):
        zfhist[t+1] = rho_z*zhist[t]
        Xvec2 = np.array([[1.0], [khist[t+1]], [khist[t+1]**2], [khist[t+1]**3], \
                  [zfhist[t+1]], [zfhist[t+1]**2], [zfhist[t+1]**3], \
                  [khist[t+1]*zfhist[t+1]], [khist[t]**2*zfhist[t+1]], \
                  [khist[t+1]*zfhist[t+1]**2]])
    
        if t < ts:
            kfhist[t+2] = np.vdot(Xvec2, coeffsPF1)
            ellfhist[t] = np.vdot(Xvec2, coeffsJF1)
            (Yfhist[t+1], wfhist[t+1], rfhist[t+1],
             Tfhist[t+1], cfhist[t+1], ifhist[t], 
             ufhist[t]) = Modeldefs(
                     kfhist[t+2], khist[t+1], ellfhist[t+1], zfhist[t+1], params)
            
        else:
            khist[t+1] = np.vdot(Xvec2, coeffsPF2)
            ellhist[t] = np.vdot(Xvec2, coeffsJF2)
            Yhist[t], whist[t], rhist[t], Thist[t], chist[t], ihist[t], uhist[t] \
                = Modeldefs(khist[t+1], khist[t], ellhist[t], zhist[t], params2)
                  
    return khist, ellhist, zhist, Yhist, whist, rhist, Thist, chist, ihist, \
           uhist, kfhist, ellfhist, zfhist, Yfhist, wfhist, rfhist, Tfhist, \
           cfhist, ifhist, ufhist,
                  
                  
                  
                  