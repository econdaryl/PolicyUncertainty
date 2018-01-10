#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Run Monte Carlos for Simple ILA Model
'''
import numpy as np
import pickle as pkl

from BMpolsim import polsim

def runmc(simargs, nsim, nobs, repincr):
    '''
    This function returns all the results from a set of Monte Carlo simulations
    of the Simple ILA model.
    
    This function reads values from the following pkl files:
        ILAfindss.pkl - steady state values and parameters
    
    Inputs:
    -----------  
    funcname: name of the policy simulation function to be used.
        The function must be set up to take a single argument which is a list
    args: the list of arguments to be used by funcname
    nsim: the number of Monte Carlo simulations to run
    nobs: the number of observations in each simulation
    repincr:  the increment between MC reports (helps to see how fast the
        simulations run)
    
    Outputs:
    -----------  
    mcdata: a list of numpy arrays with simulations in the rows and
        observations in the columns
    histdata: a list of 1-dimensional numpy arrays for the final simulation 
    preddata: a list of 1-dimensional numpy arrays for the time-zero prediction
        of the variable's history
    '''
    
    # load steady state values and parameters
    infile = open('BMfindss.pkl', 'rb')
    (bar1, bar2, params1, params2, LINparams) = pkl.load(infile)
    infile.close()
    
    # unpack
    [kbar1, Ybar1, wbar1, rbar1, Tbar1, cbar1, ibar1, ubar1] = bar1
    [kbar2, Ybar2, wbar2, rbar2, Tbar2, cbar2, ibar2, ubar2] = bar2
    [alpha, beta, tau, rho_z, sigma_z] = params1
    tau2 = params2[2]
    (zbar, Zbar, NN, nx, ny, nz, logX, Sylv) = LINparams
    
    (initial, nobs, ts, generateLIN, args1, args2, params1, params2) = simargs
    
    # get time zero prediction
    # parameters for tau1 portion
    params3 = np.array([alpha, beta, tau, rho_z, 0.])
    # paramters for tau2 portion
    params4 = np.array([alpha, beta, tau2, rho_z, 0.])
    
    # find actual steady state for baseline
    # get list of arguments for predictions simulation
    predargs = (initial, nobs, nobs, generateLIN, args1, args2, params3, \
                params3)
    # simulate with zero shocks and see what k converges to in last period
    kpred, zpred, Ypred, wpred, rpred, Tpred, cpred, ipred, upred, \
    kf, zf, Yf, wf, rf, Tf, cf, invf, uf, MsqEerr = polsim(predargs)
    
    # find actual (uncertainty) steady state values for baseline
    kact = kpred[nobs-1]
    Yact = Ypred[nobs-1]
    wact = wpred[nobs-1]
    ract = rpred[nobs-1]
    Tact = Tpred[nobs-1]
    cact = cpred[nobs-1]
    iact = ipred[nobs-1]
    uact = upred[nobs-1]
    act = (kact, Yact, wact, ract, Tact, cact, iact, uact)
    
    initial = (kact, 0.)

    # get NEW list of arguments for predictions simulation
    predargs = (initial, nobs, ts, generateLIN, args1, args2, params3, params4)
    
    # find predicted series
    kpred, zpred, Ypred, wpred, rpred, Tpred, cpred, ipred, upred, \
    kf, zf, Yf, wf, rf, Tf, cf, invf, uf, MsqEerr = polsim(predargs)
    
    # preallocate mc matrices
    kmc = np.zeros((nsim, nobs+1))
    zmc = np.zeros((nsim, nobs))
    Ymc = np.zeros((nsim, nobs))
    wmc = np.zeros((nsim, nobs))
    rmc = np.zeros((nsim, nobs))
    Tmc = np.zeros((nsim, nobs))
    cmc = np.zeros((nsim, nobs))
    imc = np.zeros((nsim, nobs))
    umc = np.zeros((nsim, nobs)) 
    foremeanmc = np.zeros((nsim, 9)) 
    zformeanmc = np.zeros((nsim, 9))
    RMsqEerrmc = np.zeros((nsim, nx+ny))
                                       
    # run remaining simulations                                
    for i in range(0, nsim):
        if np.fmod(i, repincr) == 0.:
            print('mc #:', i, 'of', nsim)
        khist, zhist, Yhist, whist, rhist, Thist, chist, ihist, \
        uhist, kfhist, zfhist, Yfhist, wfhist, rfhist, Tfhist, \
        cfhist, ifhist, ufhist, RMsqEerrhist = polsim(simargs)
            
        # replace 1-period ahead forecast with abs value of forecast error
        for t in range(1, nobs):
            kfhist[t] = np.abs(kfhist[t] - khist[t])
            zfhist[t] = np.abs(zfhist[t] - zhist[t])
            Yfhist[t] = np.abs(Yfhist[t] - Yhist[t])
            wfhist[t] = np.abs(wfhist[t] - whist[t])
            rfhist[t] = np.abs(rfhist[t] - rhist[t])
            Tfhist[t] = np.abs(Tfhist[t] - Thist[t])
            cfhist[t] = np.abs(cfhist[t] - chist[t])
            ifhist[t] = np.abs(ifhist[t] - ihist[t])
            ufhist[t] = np.abs(ufhist[t] - uhist[t])
            
        # caclulate mean 1-period ahead forecast errors
        foremean = np.array([np.mean(kfhist[1:nobs]),
                             np.mean(zfhist[1:nobs]), 
                             np.mean(Yfhist[1:nobs]),
                             np.mean(wfhist[1:nobs]), 
                             np.mean(rfhist[1:nobs]),
                             np.mean(Tfhist[1:nobs]), 
                             np.mean(cfhist[1:nobs]),
                             np.mean(ifhist[1:nobs]), 
                             np.mean(ufhist[1:nobs])])  
    
        # caclulate mean period zero forecast errors
        zformean = np.array([np.mean(khist[1:nobs] - kpred[1:nobs]),
                             np.mean(zhist[1:nobs] - zpred[1:nobs]), 
                             np.mean(Yhist[1:nobs] - Ypred[1:nobs]),
                             np.mean(whist[1:nobs] - wpred[1:nobs]), 
                             np.mean(rhist[1:nobs] - rpred[1:nobs]),
                             np.mean(Thist[1:nobs] - Tpred[1:nobs]), 
                             np.mean(chist[1:nobs] - cpred[1:nobs]),
                             np.mean(ihist[1:nobs] - ipred[1:nobs]), 
                             np.mean(uhist[1:nobs] - upred[1:nobs])])  
            
        # store results in Monte Carlo matrices
        kmc[i,:] = khist
        zmc[i,:] = zhist
        Ymc[i,:] = Yhist
        wmc[i,:] = whist
        rmc[i,:] = rhist
        Tmc[i,:] = Thist
        cmc[i,:] = chist
        imc[i,:] = ihist
        umc[i,:] = uhist
        foremeanmc[i,:] = foremean
        zformeanmc[i,:] = zformean
        RMsqEerrmc[i,:] = np.mean(RMsqEerrhist[1:nobs,:])
        
        mcdata = (kmc, zmc, Ymc, wmc, rmc, Tmc, cmc, imc, umc, \
                  foremeanmc, zformeanmc, RMsqEerrmc)
        
        histdata = (khist, zhist, Yhist, whist, rhist, Thist, chist, \
                    ihist, uhist)
        
        preddata = (kpred, zpred, Ypred, wpred, rpred, Tpred, cpred, \
                    ipred, upred)
        
    return mcdata, histdata, preddata, act