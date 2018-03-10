#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Run Monte Carlos for Simple ILA Model
'''
import numpy as np
import pickle as pkl

from OLGpolsim import polsim

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
    [alpha, beta, gamma, delta, chi, theta, tau, rho_z, \
        sigma_z, pi2, pi3, pi4, f1, f2, f3, nx, ny, nz] = params1
    tau2 = params2[6]
    (zbar, Zbar, NN, nx, ny, nz, logX, Sylv) = LINparams
    
    (initial, nobs, ts, generateLIN, args1, args2, params1, params2) = simargs
    
    # get time zero prediction
    # parameters for tau1 portion
    params3 = np.array([alpha, beta, gamma, delta, chi, theta, tau, rho_z, \
        0., pi2, pi3, pi4, f1, f2, f3, nx, ny, nz])
    # paramters for tau2 portion
    params4 = np.array([alpha, beta, gamma, delta, chi, theta, tau2, rho_z, \
        0., pi2, pi3, pi4, f1, f2, f3, nx, ny, nz])
    
    # find actual steady state for baseline
    # get list of arguments for predictions simulation
    predargs = (initial, nobs, nobs, generateLIN, args1, args2, params3, \
                params3)
    # simulate with zero shocks and see what k converges to in last period
    k2pred, k3pred, k4pred, l1pred, l2pred, l3pred, zpred, GDPpred, \
        Kpred, Lpred, wpred, rpred, T4pred, Bpred, c1pred, c2pred, c3pred, \
        c4pred, Cpred, Ipred, u1pred, u2pred, u3pred, u4pred, k2, \
        k3, k4, l1, l2, l3, z, GDP, K, L, w, r, T4, B, c1, c2, c3, c4, C, I, \
        u1, u2, u3, u4, RMsqEerr = polsim(predargs)
    
    # find actual (uncertainty) steady state values for baseline
    k2act = k2pred[nobs-1]
    k3act = k3pred[nobs-1]
    k4act = k4pred[nobs-1]
    l1act = l1pred[nobs-1]
    l2act = l2pred[nobs-1]
    l3act = l3pred[nobs-1]
    Kact  = Kpred[nobs-1]
    Lact  = Lpred[nobs-1]
    GDPact = GDPpred[nobs-1]
    wact  = wpred[nobs-1]
    ract  = rpred[nobs-1]
    T4act = T4pred[nobs-1]
    Bact  = Bpred[nobs-1]
    c1act = c1pred[nobs-1]
    c2act = c2pred[nobs-1]
    c3act = c3pred[nobs-1]
    c4act = c3pred[nobs-1]
    Cact  = Cpred[nobs-1]
    Iact  = Ipred[nobs-1]
    u1act = u1pred[nobs-1]
    u2act = u2pred[nobs-1]
    u3act = u3pred[nobs-1]
    u4act = u4pred[nobs-1]
    act = (k2act, k3act, k4act, l1act, l2act, l3act, GDPact, \
        Kact, Lact, wact, ract, T4act, Bact, c1act, c2act, c3act, \
        c4act, Cact, Iact, u1act, u2act, u3act, u4act)
    
    initial = (k2act, k3act, k4act, 0.)

    # get NEW list of arguments for predictions simulation
    predargs = (initial, nobs, ts, generateLIN, args1, args2, params3, params4)
    
    # find predicted series
    k2pred, k3pred, k4pred, l1pred, l2pred, l3pred, zpred, GDPpred, \
        Kpred, Lpred, wpred, rpred, T4pred, Bpred, c1pred, c2pred, c3pred, \
        c4pred, Cpred, Ipred, u1pred, u2pred, u3pred, u4pred, k2, \
        k3, k4, l1, l2, l3, z, GDP, K, L, w, r, T4, B, c1, c2, c3, c4, C, I, \
        u1, u2, u3, u4, RMsqEerr = polsim(predargs)
    
    # preallocate mc matrices
    k2mc = np.zeros((nsim, nobs+1))
    k3mc = np.zeros((nsim, nobs+1))
    k4mc = np.zeros((nsim, nobs+1))
    l1mc = np.zeros((nsim, nobs))
    l2mc = np.zeros((nsim, nobs))
    l3mc = np.zeros((nsim, nobs))
    zmc  = np.zeros((nsim, nobs))
    Kmc  = np.zeros((nsim, nobs))
    Lmc  = np.zeros((nsim, nobs))
    GDPmc = np.zeros((nsim, nobs))
    wmc  = np.zeros((nsim, nobs))
    rmc  = np.zeros((nsim, nobs))
    T4mc = np.zeros((nsim, nobs))
    Bmc  = np.zeros((nsim, nobs))
    c1mc = np.zeros((nsim, nobs))
    c2mc = np.zeros((nsim, nobs))
    c3mc = np.zeros((nsim, nobs))
    c4mc = np.zeros((nsim, nobs))
    Cmc  = np.zeros((nsim, nobs))
    Imc  = np.zeros((nsim, nobs)) 
    u1mc = np.zeros((nsim, nobs))
    u2mc = np.zeros((nsim, nobs))
    u3mc = np.zeros((nsim, nobs))
    u4mc = np.zeros((nsim, nobs))
    foremeanmc = np.zeros((nsim, 24)) 
    forevarmc = np.zeros((nsim, 24)) 
    zformeanmc = np.zeros((nsim, 24))
    zforvarmc = np.zeros((nsim, 24))
    RMsqEerrmc = np.zeros((nsim, nx + ny))
                                       
    # run remaining simulations                                
    for i in range(0, nsim):
        if np.fmod(i, repincr) == 0.:
            print('mc #:', i, 'of', nsim)
        k2hist, k3hist, k4hist, l1hist, l2hist, l3hist, zhist, GDPhist, \
            Khist, Lhist, whist, rhist, T4hist, Bhist, c1hist, c2hist, c3hist,\
            c4hist, Chist, Ihist, u1hist, u2hist, u3hist, u4hist, k2fhist, \
            k3fhist, k4fhist, l1fhist, l2fhist, l3fhist, zfhist, GDPfhist, \
            Kfhist, Lfhist, wfhist, rfhist, T4fhist, Bfhist, c1fhist, c2fhist,\
            c3fhist, c4fhist, Cfhist, Ifhist, u1fhist, u2fhist, u3fhist, \
            u4fhist, RMsqEerrhist = polsim(simargs)
            
        # replace 1-period ahead forecast with abs value of forecast error
        for t in range(1, nobs):
            k2fhist[t] = np.abs(k2fhist[t]/k2hist[t]-1)
            k3fhist[t] = np.abs(k3fhist[t]/k3hist[t]-1)
            k4fhist[t] = np.abs(k4fhist[t]/k4hist[t]-1)
            l1fhist[t] = np.abs(l1fhist[t]/l1hist[t]-1)
            l2fhist[t] = np.abs(l2fhist[t]/l2hist[t]-1)
            l3fhist[t] = np.abs(l3fhist[t]/l3hist[t]-1)
            zfhist[t]  = np.abs(zfhist[t] / zhist[t]-1)
            Kfhist[t]  = np.abs(Kfhist[t] / Khist[t]-1)
            Lfhist[t]  = np.abs(Lfhist[t] / Lhist[t]-1)
            GDPfhist[t] = np.abs(GDPfhist[t]/GDPhist[t]-1)
            wfhist[t]  = np.abs(wfhist[t] / whist[t]-1)
            rfhist[t]  = np.abs(rfhist[t] / rhist[t]-1)
            T4fhist[t] = np.abs(T4fhist[t]/T4hist[t]-1)
            Bfhist[t]  = np.abs(Bfhist[t] / Bhist[t]-1)
            c1fhist[t] = np.abs(c1fhist[t]/c1hist[t]-1)
            c2fhist[t] = np.abs(c2fhist[t]/c2hist[t]-1)
            c3fhist[t] = np.abs(c3fhist[t]/c3hist[t]-1)
            c4fhist[t] = np.abs(c4fhist[t]/c4hist[t]-1)
            Cfhist[t]  = np.abs(Cfhist[t] / Chist[t]-1)
            Ifhist[t]  = np.abs(Ifhist[t] / Ihist[t]-1)
            u1fhist[t] = np.abs(u1fhist[t]/u1hist[t]-1)
            u2fhist[t] = np.abs(u2fhist[t]/u2hist[t]-1)
            u3fhist[t] = np.abs(u3fhist[t]/u3hist[t]-1)
            u4fhist[t] = np.abs(u4fhist[t]/u4hist[t]-1)
            
        # calculate mean 1-period ahead forecast errors
        foremean = np.array([np.mean(k2fhist[1:nobs]),
                             np.mean(k3fhist[1:nobs]),
                             np.mean(k4fhist[1:nobs]),
                             np.mean(l1fhist[1:nobs]),
                             np.mean(l2fhist[1:nobs]),
                             np.mean(l3fhist[1:nobs]),
                             np.mean(zfhist[1:nobs]), 
                             np.mean(Kfhist[1:nobs]),
                             np.mean(Lfhist[1:nobs]),
                             np.mean(GDPfhist[1:nobs]),
                             np.mean(wfhist[1:nobs]), 
                             np.mean(rfhist[1:nobs]),
                             np.mean(T4fhist[1:nobs]), 
                             np.mean(Bfhist[1:nobs]),
                             np.mean(c1fhist[1:nobs]),
                             np.mean(c2fhist[1:nobs]),
                             np.mean(c3fhist[1:nobs]),
                             np.mean(c4fhist[1:nobs]),
                             np.mean(Cfhist[1:nobs]), 
                             np.mean(Ifhist[1:nobs]),
                             np.mean(u1fhist[1:nobs]),
                             np.mean(u2fhist[1:nobs]),
                             np.mean(u3fhist[1:nobs]),
                             np.mean(u4fhist[1:nobs])])  
        
        # calculate mean 1-period ahead forecast variances
        forevar = np.array([np.mean(k2fhist[1:nobs]**2),
                             np.mean(k3fhist[1:nobs]**2),
                             np.mean(k4fhist[1:nobs]**2),
                             np.mean(l1fhist[1:nobs]**2),
                             np.mean(l2fhist[1:nobs]**2),
                             np.mean(l3fhist[1:nobs]**2),
                             np.mean(zfhist[1:nobs]**2), 
                             np.mean(Kfhist[1:nobs]**2),
                             np.mean(Lfhist[1:nobs]**2),
                             np.mean(GDPfhist[1:nobs]**2),
                             np.mean(wfhist[1:nobs]**2), 
                             np.mean(rfhist[1:nobs]**2),
                             np.mean(T4fhist[1:nobs]**2), 
                             np.mean(Bfhist[1:nobs]**2),
                             np.mean(c1fhist[1:nobs]**2),
                             np.mean(c2fhist[1:nobs]**2),
                             np.mean(c3fhist[1:nobs]**2),
                             np.mean(c4fhist[1:nobs]**2),
                             np.mean(Cfhist[1:nobs]**2), 
                             np.mean(Ifhist[1:nobs]**2),
                             np.mean(u1fhist[1:nobs]**2),
                             np.mean(u2fhist[1:nobs]**2),
                             np.mean(u3fhist[1:nobs]**2),
                             np.mean(u4fhist[1:nobs]**2)])  
    
        # calculate mean period zero forecast errors
        zformean = np.array([np.mean(k2hist[1:nobs]/k2pred[1:nobs]-1),
                             np.mean(k3hist[1:nobs]/k3pred[1:nobs]-1),
                             np.mean(k4hist[1:nobs]/k4pred[1:nobs]-1),
                             np.mean(l1hist[1:nobs]/l1pred[1:nobs]-1),
                             np.mean(l2hist[1:nobs]/l2pred[1:nobs]-1),
                             np.mean(l3hist[1:nobs]/l3pred[1:nobs]-1),
                             np.mean(zhist[1:nobs] / zpred[1:nobs]-1), 
                             np.mean(Khist[1:nobs] / Kpred[1:nobs]-1),
                             np.mean(Lhist[1:nobs] / Lpred[1:nobs]-1),
                             np.mean(GDPhist[1:nobs]/GDPpred[1:nobs]-1),
                             np.mean(whist[1:nobs] / wpred[1:nobs]-1), 
                             np.mean(rhist[1:nobs] / rpred[1:nobs]-1),
                             np.mean(T4hist[1:nobs]/T4pred[1:nobs]-1),
                             np.mean(Bhist[1:nobs] / Bpred[1:nobs]-1),
                             np.mean(c1hist[1:nobs]/c1pred[1:nobs]-1),
                             np.mean(c2hist[1:nobs]/c2pred[1:nobs]-1),
                             np.mean(c3hist[1:nobs]/c3pred[1:nobs]-1),
                             np.mean(c4hist[1:nobs]/c4pred[1:nobs]-1),
                             np.mean(Chist[1:nobs] / Cpred[1:nobs]-1), 
                             np.mean(Ihist[1:nobs] / Ipred[1:nobs]-1),
                             np.mean(u1hist[1:nobs]/u1pred[1:nobs]-1),
                             np.mean(u2hist[1:nobs]/u2pred[1:nobs]-1),
                             np.mean(u3hist[1:nobs]/u3pred[1:nobs]-1),
                             np.mean(u4hist[1:nobs]/u4pred[1:nobs]-1)])  

        # calculate mean period zero forecast variances
        zforvar = np.array([np.mean((k2hist[1:nobs]/ k2pred[1:nobs]-1)**2),
                             np.mean((k3hist[1:nobs]/k3pred[1:nobs]-1)**2),
                             np.mean((k4hist[1:nobs]/k4pred[1:nobs]-1)**2),
                             np.mean((l1hist[1:nobs]/l1pred[1:nobs]-1)**2),
                             np.mean((l2hist[1:nobs]/l2pred[1:nobs]-1)**2),
                             np.mean((l3hist[1:nobs]/l3pred[1:nobs]-1)**2),
                             np.mean((zhist[1:nobs] / zpred[1:nobs]-1)**2), 
                             np.mean((Khist[1:nobs] / Kpred[1:nobs]-1)**2),
                             np.mean((Lhist[1:nobs] / Lpred[1:nobs]-1)**2),
                             np.mean((GDPhist[1:nobs]/GDPpred[1:nobs]-1)**2),
                             np.mean((whist[1:nobs] / wpred[1:nobs]-1)**2), 
                             np.mean((rhist[1:nobs] / rpred[1:nobs]-1)**2),
                             np.mean((T4hist[1:nobs]/T4pred[1:nobs]-1)**2),
                             np.mean((Bhist[1:nobs] / Bpred[1:nobs]-1)**2),
                             np.mean((c1hist[1:nobs]/c1pred[1:nobs]-1)**2),
                             np.mean((c2hist[1:nobs]/c2pred[1:nobs]-1)**2),
                             np.mean((c3hist[1:nobs]/c3pred[1:nobs]-1)**2),
                             np.mean((c4hist[1:nobs]/c4pred[1:nobs]-1)**2),
                             np.mean((Chist[1:nobs] / Cpred[1:nobs]-1)**2), 
                             np.mean((Ihist[1:nobs] / Ipred[1:nobs]-1)**2),
                             np.mean((u1hist[1:nobs]/u1pred[1:nobs]-1)**2),
                             np.mean((u2hist[1:nobs]/u2pred[1:nobs]-1)**2),
                             np.mean((u3hist[1:nobs]/u3pred[1:nobs]-1)**2),
                             np.mean((u4hist[1:nobs]/u4pred[1:nobs]-1)**2)])  
            
        # store results in Monte Carlo matrices
        k2mc[i,:] = k2hist
        k3mc[i,:] = k3hist
        k4mc[i,:] = k4hist
        l1mc[i,:] = l1hist
        l2mc[i,:] = l2hist
        l3mc[i,:] = l3hist
        zmc[i,:]  = zhist
        Kmc[i,:]  = Khist
        Lmc[i,:]  = Lhist
        GDPmc[i,:] = GDPhist
        wmc[i,:]  = whist
        rmc[i,:]  = rhist
        T4mc[i,:] = T4hist
        Bmc[i,:]  = Bhist
        c1mc[i,:] = c1hist
        c2mc[i,:] = c2hist
        c3mc[i,:] = c3hist
        c4mc[i,:] = c4hist
        Cmc[i,:]  = Chist
        Imc[i,:]  = Ihist
        u1mc[i,:] = u1hist
        u2mc[i,:] = u2hist
        u3mc[i,:] = u3hist
        u4mc[i,:] = u4hist
        foremeanmc[i,:] = foremean
        forevarmc[i,:] = forevar
        zformeanmc[i,:] = zformean
        zforvarmc[i,:] = zforvar
        RMsqEerrmc[i,:] = np.mean(RMsqEerrhist[1:nobs,:])
        
        mcdata = (k2mc, k3mc, k4mc, l1mc, l2mc, l3mc, zmc, Kmc, Lmc, GDPmc, \
            wmc, rmc, T4mc, Bmc, c1mc, c2mc, c3mc, c4mc, Cmc, Imc, u1mc, \
            u2mc, u3mc, u4mc, foremeanmc, forevarmc, zformeanmc, zforvarmc, \
            RMsqEerrmc)
        
        histdata = (k2hist, k3hist, k4hist, l1hist, l2hist, l3hist, zhist, \
            Khist, Lhist, GDPhist, whist, rhist, T4hist, Bhist, c1hist, \
            c2hist, c3hist, c4hist, Chist, Ihist, u1hist, u2hist, u3hist, \
            u4hist)
        
        preddata = (k2pred, k3pred, k4pred, l1pred, l2pred, l3pred, zpred, \
            Kpred, Lpred, GDPpred, wpred, rpred, T4pred, Bpred, c1pred, \
            c2pred, c3pred, c4pred, Cpred, Ipred, u1pred, u2pred, u3pred, \
            u4pred)
        
    return mcdata, histdata, preddata, act