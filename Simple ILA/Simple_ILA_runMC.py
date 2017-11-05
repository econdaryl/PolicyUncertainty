'''
Run Monte Carlos for Simple ILA Model
'''
import numpy as np

def runMC(funcname, args, nsim, nobs):
    
    # run first simulation and store in Monte Carlo matrices
    kmc, ellmc, zmc, Ymc, wmc, rmc, Tmc, cmc, imc, umc, \
    kfmc, ellfmc, zfmc, Yfmc, wfmc, rfmc, Tfmc, cfmc, ifmc, ufmc \
        = funcname(args)
    
    # replace forecast with abs value of forecast error
    for t in range(1, nobs):
        kfmc[t] = np.abs(kfmc[t] - kmc[t])
        ellfmc[t] = np.abs(ellfmc[t] - ellmc[t])
        zfmc[t] = np.abs(zfmc[t] - zmc[t])
        Yfmc[t] = np.abs(Yfmc[t] - Ymc[t])
        wfmc[t] = np.abs(wfmc[t] - wmc[t])
        rfmc[t] = np.abs(rfmc[t] - rmc[t])
        Tfmc[t] = np.abs(Tfmc[t] - Tmc[t])
        cfmc[t] = np.abs(cfmc[t] - cmc[t])
        ifmc[t] = np.abs(ifmc[t] - imc[t])
        ufmc[t] = np.abs(ufmc[t] - umc[t])
    
    # caclulate mean forecast errors
    foremeanmc = np.array([np.mean(kfmc[1:nobs]),
                           np.mean(ellfmc[1:nobs]),
                           np.mean(zfmc[1:nobs]), 
                           np.mean(Yfmc[1:nobs]),
                           np.mean(wfmc[1:nobs]), 
                           np.mean(rfmc[1:nobs]),
                           np.mean(Tfmc[1:nobs]), 
                           np.mean(cfmc[1:nobs]),
                           np.mean(ifmc[1:nobs]), 
                           np.mean(ufmc[1:nobs])])                                 
                                       
    # run remaining simulations                                
    for i in range(1, nsim):
        print('mc #:', i, nsim)
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
            
        # stack results in Monte Carlo matrices
        kmc = np.vstack((kmc, khist))
        ellmc = np.vstack((ellmc, ellhist))
        zmc = np.vstack((zmc, zhist))
        Ymc = np.vstack((Ymc, Yhist))
        wmc = np.vstack((wmc, whist))
        rmc = np.vstack((rmc, rhist))
        Tmc = np.vstack((Tmc, Thist))
        cmc = np.vstack((cmc, chist))
        imc = np.vstack((imc, ihist))
        umc = np.vstack((umc, uhist))
        foremeanmc = np.vstack((foremeanmc, foremean))
        
        mcdata = (kmc, ellmc, zmc, Ymc, wmc, rmc, Tmc, cmc, imc, umc, \
                  foremeanmc)
        
        return mcdata