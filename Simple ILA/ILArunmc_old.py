'''
Run Monte Carlos for Simple ILA Model
'''
import numpy as np

def runmc(funcname, args, nsim, nobs, repincr):
    '''
    This function returns all the results from a set of Monte Carlo simulations
    of the Simple ILA model.
    
    Inputs:
    funcname: name of the policy simulation function to be used.
        The function must be set up to take a single argument which is a list
    args: the list of arguments to be used by funcname
    nsim: the number of Monte Carlo simulations to run
    nobs: the number of observations in each simulation
    repincr:  the increment between MC reports (helps to see how fast the
        simulations run)
    
    Outputs:
    mcdata: a list of numpy arrays with simulations in the rows and
        observations in the columns
    histdata: a list of 1-dimensional numpy arrays for the final simulation  
    '''
    
    # preallocate mc matrices
    kmc = np.zeros((nsim, nobs+1))
    ellmc = np.zeros((nsim, nobs))
    zmc = np.zeros((nsim, nobs))
    Ymc = np.zeros((nsim, nobs))
    wmc = np.zeros((nsim, nobs))
    rmc = np.zeros((nsim, nobs))
    Tmc = np.zeros((nsim, nobs))
    cmc = np.zeros((nsim, nobs))
    imc = np.zeros((nsim, nobs))
    umc = np.zeros((nsim, nobs)) 
    foremeanmc = np.zeros((nsim, 10)) 
                                       
    # run remaining simulations                                
    for i in range(0, nsim):
        if np.fmod(i, repincr) == 0.:
            print('mc #:', i, 'of', nsim)
        khist, ellhist, zhist, Yhist, whist, rhist, Thist, chist, ihist, uhist, \
        kfhist, ellfhist, zfhist, Yfhist, wfhist, rfhist, Tfhist, cfhist, ifhist, \
        ufhist= funcname(args)
            
        # replace forecast with abs value of forecast error
        for t in range(1, nobs):
            kfhist[t] = np.abs(kfhist[t] - khist[t])
            ellfhist[t] = np.abs(ellfhist[t] - ellhist[t])
            zfhist[t] = np.abs(zfhist[t] - zhist[t])
            Yfhist[t] = np.abs(Yfhist[t] - Yhist[t])
            wfhist[t] = np.abs(wfhist[t] - whist[t])
            rfhist[t] = np.abs(rfhist[t] - rhist[t])
            Tfhist[t] = np.abs(Tfhist[t] - Thist[t])
            cfhist[t] = np.abs(cfhist[t] - chist[t])
            ifhist[t] = np.abs(ifhist[t] - ihist[t])
            ufhist[t] = np.abs(ufhist[t] - uhist[t])
            
        # caclulate mean forecast errors
        foremean = np.array([np.mean(kfhist[1:nobs]),
                             np.mean(ellfhist[1:nobs]),
                             np.mean(zfhist[1:nobs]), 
                             np.mean(Yfhist[1:nobs]),
                             np.mean(wfhist[1:nobs]), 
                             np.mean(rfhist[1:nobs]),
                             np.mean(Tfhist[1:nobs]), 
                             np.mean(cfhist[1:nobs]),
                             np.mean(ifhist[1:nobs]), 
                             np.mean(ufhist[1:nobs])])   
            
        # store results in Monte Carlo matrices
        kmc[i,:] = khist
        ellmc[i,:] = ellhist
        zmc[i,:] = zhist
        Ymc[i,:] = Yhist
        wmc[i,:] = whist
        rmc[i,:] = rhist
        Tmc[i,:] = Thist
        cmc[i,:] = chist
        imc[i,:] = ihist
        umc[i,:] = uhist
        foremeanmc[i,:] = foremean
        
        mcdata = (kmc, ellmc, zmc, Ymc, wmc, rmc, Tmc, cmc, imc, umc, \
                  foremeanmc)
        
        histdata = (khist, ellhist, zhist, Yhist, whist, rhist, Thist, chist, \
                    ihist, uhist)
        
    return mcdata, histdata