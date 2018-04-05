'''
Analysis of Monte Carlos for Simple ILA Model
'''
import numpy as np
from OLG3plots import OLGplots
    

def mcanalysis(mcdata, preddata, bardata, histdata, name, nsim):
    '''
    This function finds confidence bands for data from the Monte Carlo
    simulations.  It also plots predictions and with confidence bands, and 
    predictions versus the final simulation as an example.
    
    Inputs:
    -----------  
    mcdata: a list of numpy arrays with simulations in the rows and
        observations in the columns
    preddata: a list of 1-dimensional numpy arrays for the period zero
        predictions from the model
    bardata: a list of steady state values from the baseline
    histdata: a list of 1-dimensional numpy arrays for the final simulation 
    name: a string that is used when saving the plots and other files
    nsim: the number of Monte Carlo simulations that have been run
    
    Outputs:
    -----------  
    avgdata: list of 1-dimensional numpy arrays containing the average values 
        from the simulations for each time period
    uppdata: list of 1-dimensional numpy arrays containing the upper confidence
        bands from the simulations for each time period
    lowdata: list of 1-dimensional numpy arrays containing the lower confidence
        bands from the simulations for each time period
    '''    
    
    #unpack data
    (k2mc, k3mc, l1mc, l2mc, zmc, Kmc, Lmc, GDPmc, \
        wmc, rmc, T3mc, Bmc, c1mc, c2mc, c3mc, Cmc, Imc, u1mc, \
        u2mc, u3mc, foremeanmc, forevarmc, zformeanmc, zforvarmc, \
        RMsqEerrmc) = mcdata
    # calculate and report statistics and charts from Monte Carlos  
    (k2pred, k3pred, l1pred, l2pred, zpred, \
        Kpred, Lpred, GDPpred, wpred, rpred, T3pred, Bpred, c1pred, \
        c2pred, c3pred, Cpred, Ipred, u1pred, u2pred, u3pred) = preddata   
    (k2bar, k3bar, l1bar, l2bar, Kbar, \
        Lbar, GDPbar, wbar, rbar, T3bar, Bbar, c1bar, c2bar, c3bar, \
        Cbar, Ibar, u1bar, u2bar, u3bar) = bardata
    (k2hist, k3hist, l1hist, l2hist, zhist, \
        Khist, Lhist, GDPhist, whist, rhist, T3hist, Bhist, c1hist, \
        c2hist, c3hist, Chist, Ihist, u1hist, u2hist, u3hist) = histdata
          
    # now sort the Monte Carlo matrices over the rows
    k2mc = np.sort(k2mc, axis = 0)
    k3mc = np.sort(k3mc, axis = 0)
    l1mc = np.sort(l1mc, axis = 0)
    l2mc = np.sort(l2mc, axis = 0)
    zmc = np.sort(zmc, axis = 0)
    Kmc = np.sort(Kmc, axis = 0)
    Lmc = np.sort(Lmc, axis = 0)
    GDPmc = np.sort(GDPmc, axis = 0)
    wmc = np.sort(wmc, axis = 0)
    rmc = np.sort(rmc, axis = 0)
    T3mc = np.sort(T3mc, axis = 0)
    Bmc = np.sort(Bmc, axis = 0)
    c1mc = np.sort(c1mc, axis = 0)
    c2mc = np.sort(c2mc, axis = 0)
    c3mc = np.sort(c3mc, axis = 0)
    Cmc = np.sort(Cmc, axis = 0)
    Imc = np.sort(Imc, axis = 0)
    u1mc = np.sort(u1mc, axis = 0)
    u2mc = np.sort(u2mc, axis = 0)
    u3mc = np.sort(u3mc, axis = 0)
    foremeanmc = np.sort(foremeanmc, axis = 0)
    forevarmc = np.sort(forevarmc, axis = 0)
    zformeanmc = np.sort(zformeanmc, axis = 0)
    zforvarmc = np.sort(zforvarmc, axis = 0)
    
    # find the average values for each variable in each time period across 
    # Monte Carlos
    k2avg = np.mean(k2mc, axis = 0)
    k3avg = np.mean(k3mc, axis = 0)
    l1avg = np.mean(l1mc, axis = 0)
    l2avg = np.mean(l2mc, axis = 0)
    zavg = np.mean(zmc, axis = 0)
    Kavg = np.mean(Kmc, axis = 0)
    Lavg = np.mean(Lmc, axis = 0)
    GDPavg = np.mean(GDPmc, axis = 0)
    wavg = np.mean(wmc, axis = 0)
    ravg = np.mean(rmc, axis = 0)
    T3avg = np.mean(T3mc, axis = 0)
    Bavg = np.mean(Bmc, axis = 0)
    c1avg = np.mean(c1mc, axis = 0)
    c2avg = np.mean(c2mc, axis = 0)
    c3avg = np.mean(c3mc, axis = 0)
    Cavg = np.mean(Cmc, axis = 0)
    Iavg = np.mean(Imc, axis = 0)
    u1avg = np.mean(u1mc, axis = 0)
    u2avg = np.mean(u2mc, axis = 0)
    u3avg = np.mean(u3mc, axis = 0)
    foremeanavg = np.mean(np.abs(foremeanmc), axis = 0)
    forevaravg = np.mean(np.abs(forevarmc), axis = 0)
    zformeanavg = np.mean(np.abs(zformeanmc), axis = 0)
    zforvaravg = np.mean(np.abs(zforvarmc), axis = 0)
    RMsqEerravg = np.mean(np.abs(RMsqEerrmc), axis = 0)
    
    # find the rows for desired confidence bands
    conf = .1
    low = int(np.floor((conf/2)*nsim))
    high = nsim - low
    
    # find the upper and lower confidence bands for each variable
    k2upp = k2mc[high,:]
    k3upp = k3mc[high,:]
    l1upp = l1mc[high,:]
    l2upp = l2mc[high,:]
    zupp = zmc[high,:]
    Kupp = Kmc[high,:]
    Lupp = Lmc[high,:]
    GDPupp = GDPmc[high,:]
    wupp = wmc[high,:]
    rupp = rmc[high,:]
    T3upp = T3mc[high,:]
    Bupp = Bmc[high,:]
    c1upp = c1mc[high,:]
    c2upp = c2mc[high,:]
    c3upp = c3mc[high,:]
    Cupp = Cmc[high,:]
    Iupp = Imc[high,:]
    u1upp = u1mc[high,:]
    u2upp = u2mc[high,:]
    u3upp = u3mc[high,:]
    foremeanupp = foremeanmc[high,:]
    forevarupp = forevarmc[high,:]
    zformeanupp = zformeanmc[high,:]
    zforvarupp = zforvarmc[high,:]
    
    k2low = k2mc[low,:]
    k3low = k3mc[low,:]
    l1low = l1mc[low,:]
    l2low = l2mc[low,:]
    zlow = zmc[low,:]
    Klow = Kmc[low,:]
    Llow = Lmc[low,:]
    GDPlow = GDPmc[low,:]
    wlow = wmc[low,:]
    rlow = rmc[low,:]
    T3low = T3mc[low,:]
    Blow = Bmc[low,:]
    c1low = c1mc[low,:]
    c2low = c2mc[low,:]
    c3low = c3mc[low,:]
    Clow = Cmc[low,:]
    Ilow = Imc[low,:]
    u1low = u1mc[low,:]
    u2low = u2mc[low,:]
    u3low = u3mc[low,:]
    foremeanlow = foremeanmc[low,:]
    forevarlow = forevarmc[low,:]
    zformeanlow = zformeanmc[low,:]
    zforvarlow = zforvarmc[low,:]
    
    # create a list of time series to plot
    dataplot = np.array([k2pred/k2bar, k2upp/k2bar, k2low/k2bar, k2hist/k2bar,\
        k3pred/k3bar, k3upp/k3bar, k3low/k3bar, k3hist/k3bar, \
        l1pred/l1bar, l1upp/l1bar, l1low/l1bar, l1hist/l1bar, \
        l2pred/l2bar, l2upp/l2bar, l2low/l2bar, l2hist/l2bar, \
        zpred, zupp, zlow, zhist, \
        Kpred/Kbar, Kupp/Kbar, Klow/Kbar, Khist/Kbar, \
        Lpred/Lbar, Lupp/Lbar, Llow/Lbar, Lhist/Lbar, \
        GDPpred/GDPbar, GDPupp/GDPbar, GDPlow/GDPbar, GDPhist/GDPbar, \
        wpred/wbar, wupp/wbar, wlow/wbar, whist/wbar, \
        rpred/rbar, rupp/rbar, rlow/rbar, rhist/rbar, \
        T3pred/T3bar, T3upp/T3bar, T3low/T3bar, T3hist/T3bar, \
        Bpred/Bbar, Bupp/Bbar, Blow/Bbar, Bhist/Bbar, \
        c1pred/c1bar, c1upp/c1bar, c1low/c1bar, c1hist/c1bar, \
        c2pred/c2bar, c2upp/c2bar, c2low/c2bar, c2hist/c2bar, \
        c3pred/c3bar, c3upp/c3bar, c3low/c3bar, c3hist/c3bar, \
        Cpred/Cbar, Cupp/Cbar, Clow/Cbar, Chist/Cbar, \
        Ipred/Ibar, Iupp/Ibar, Ilow/Ibar, Ihist/Ibar,\
        u1pred/u1bar, u1upp/u1bar, u1low/u1bar, u1hist/u1bar, \
        u2pred/u2bar, u2upp/u2bar, u2low/u2bar, u2hist/u2bar, \
        u3pred/u3bar, u3upp/u3bar, u3low/u3bar, u3hist/u3bar])
    
    # plot using Simple ILA Model Plot.py
    OLGplots(dataplot, name)
    
    # create lists of data to return
    avgdata = (k2avg, k3avg, l1avg, l2avg, zavg, Kavg, Lavg, \
               GDPavg, wavg, ravg, T3avg, Bavg, c1avg, c2avg, c3avg, \
               Cavg, Iavg, u1avg, u2avg, u3avg, foremeanavg, \
               forevaravg, zformeanavg, zforvaravg, RMsqEerravg) 
    uppdata = (k2upp, k3upp, l1upp, l2upp, zupp, Kupp, Lupp, \
               GDPupp, wupp, rupp, T3upp, Bupp, c1upp, c2upp, c3upp, \
               Cupp, Iupp, u1upp, u2upp, u3upp, \
               foremeanupp, forevarupp, zformeanupp, zforvarupp) 
    lowdata = (k2low, k3low, l1low, l2low, zlow, Klow, Llow, \
               GDPlow, wlow, rlow, T3low, Blow, c1low, c2low, c3low, \
               Clow, Ilow, u1low, u2low, u3low, \
               foremeanlow, forevarlow, zformeanlow, zforvarlow) 
    
    return avgdata, uppdata, lowdata