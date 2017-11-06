#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 06:38:49 2017

@author: klp4
"""
import matplotlib.pyplot as plt

def ILAplots(dataplot, name):
    '''
    This function takes a list of time series from the ILA model generated u
    using either linearization or VFI.  It plots and saves a series of graphs
    of these over time.
    
    The list data must contain the following time series for each variable:
    x_pred - the predicted time path as of date zero
    x_upp - the upper confidence band
    x_low - the lower confidence band
    x_hist - a typical history
    
    The variables to be plotted are:
    k - capital stock
    ell - labor
    z - productivity
    Y - GDP
    w - wage
    r - rental
    T - tax revenue
    c - consumption
    i - investment
    u - within period utility
    '''
    
    # turn interactive plotting off
    # plt.ioff()
    
    # unpack data for plots
    [kpred, kupp, klow, khist,
        ellpred, ellupp, elllow, ellhist, \
        zpred, zupp, zlow, zhist, \
        Ypred, Yupp, Ylow, Yhist, \
        wpred, wupp, wlow, whist, \
        rpred, rupp, rlow, rhist, \
        Tpred, Tupp, Tlow, Thist, \
        cpred, cupp, clow, chist, \
        ipred, iupp, ilow, ihist, \
        upred, uupp, ulow, uhist] = dataplot
            
    # plot
    fig1 = plt.figure()
    plt.subplot(2,2,1)
    plt.plot(range(kpred.size), kpred, 'k-',
             range(kupp.size), kupp, 'k:',
             range(klow.size), klow, 'k:')
    plt.title('Capital')
    plt.xticks([])
    
    plt.subplot(2,2,2)
    plt.plot(range(ellpred.size), ellpred, 'k-',
             range(ellupp.size), ellupp, 'k:',
             range(elllow.size), elllow, 'k:')
    plt.title('Labor')
    plt.xticks([])
    
    plt.subplot(2,2,3)
    plt.plot(range(zpred.size), zpred, 'k-',
             range(zupp.size), zupp, 'k:',
             range(zlow.size), zlow, 'k:')
    plt.title('Technology')
    plt.xticks([])
    
    plt.subplot(2,2,4)
    plt.plot(range(Ypred.size), Ypred, 'k-',
             range(Yupp.size), Yupp, 'k:',
             range(Ylow.size), Ylow, 'k:')
    plt.title('GDP')
    plt.xticks([])
    
    # save high quality version to external file
    plt.savefig('ILALinfig1.eps', format='eps', dpi=2000)
    plt.show(fig1)
    plt.close(fig1)


    fig2 = plt.figure()
    plt.subplot(3,2,1)
    plt.plot(range(wpred.size), wpred, 'k-',
             range(wupp.size), wupp, 'k:',
             range(wlow.size), wlow, 'k:')
    plt.title('Wages')
    plt.xticks([])
    
    plt.subplot(3,2,2)
    plt.plot(range(rpred.size), rpred, 'k-',
             range(rupp.size), rupp, 'k:',
             range(rlow.size), rlow, 'k:')
    plt.title('Interest Rate')
    plt.xticks([])
    
    plt.subplot(3,2,3)
    plt.plot(range(Tpred.size), Tpred, 'k-',
             range(Tupp.size), Tupp, 'k:',
             range(Tlow.size), Tlow, 'k:')
    plt.title('Taxes')
    plt.xticks([])
    
    plt.subplot(3,2,4)
    plt.plot(range(cpred.size), cpred, 'k-',
             range(cupp.size), cupp, 'k:',
             range(clow.size), clow, 'k:')
    plt.title('Consumption')
    plt.xticks([])
    
    plt.subplot(3,2,5)
    plt.plot(range(ipred.size), ipred, 'k-',
             range(iupp.size), iupp, 'k:',
             range(ilow.size), ilow, 'k:')
    plt.title('Investment')
    plt.xticks([])
    
    plt.subplot(3,2,6)
    plt.plot(range(upred.size), upred, 'k-',
             range(uupp.size), uupp, 'k:',
             range(ulow.size), ulow, 'k:')
    plt.title('Utility')
    plt.xticks([])
    
    # save high quality version to external file
    plt.savefig('ILALinfig2.eps', format='eps', dpi=2000)
    plt.show(fig2)
    plt.close(fig2)
    
    
    fig3 = plt.figure()
    # plot
    plt.subplot(2,2,1)
    plt.plot(range(khist.size), khist, 'k-',
             range(kpred.size), kpred, 'r-')
    plt.title('Capital')
    plt.xticks([])
    
    plt.subplot(2,2,2)
    plt.plot(range(ellhist.size), ellhist, 'k-',
             range(ellpred.size), ellpred, 'r-')
    plt.title('Labor')
    plt.xticks([])
    
    plt.subplot(2,2,3)
    plt.plot(range(zhist.size), zhist, 'k-',
             range(zpred.size), zpred, 'r-')
    plt.title('Technology')
    plt.xticks([])
    
    plt.subplot(2,2,4)
    plt.plot(range(Yhist.size), Yhist, 'k-',
             range(Ypred.size), Ypred, 'r-')
    plt.title('GDP')
    plt.xticks([])
    
    # save high quality version to external file
    plt.savefig('ILALinfig3.eps', format='eps', dpi=2000)
    plt.show(fig3)
    plt.close(fig3)
    
    
    fig4 = plt.figure()
    plt.subplot(3,2,1)
    plt.plot(range(whist.size), whist, 'k-',
             range(wpred.size), wpred, 'r-')
    plt.title('Wages')
    plt.xticks([])
    
    plt.subplot(3,2,2)
    plt.plot(range(rhist.size), rhist, 'k-',
             range(rpred.size), rpred, 'r-')
    plt.title('Interest Rate')
    plt.xticks([])
    
    plt.subplot(3,2,3)
    plt.plot(range(Thist.size), Thist, 'k-',
             range(Tpred.size), Tpred, 'r-')
    plt.title('Taxes')
    plt.xticks([])
    
    plt.subplot(3,2,4)
    plt.plot(range(chist.size), chist, 'k-',
             range(cpred.size), cpred, 'r-')
    plt.title('Consumption')
    plt.xticks([])
    
    plt.subplot(3,2,5)
    plt.plot(range(ihist.size), ihist, 'k-',
             range(ipred.size), ipred, 'r-')
    plt.title('Investment')
    plt.xticks([])
    
    plt.subplot(3,2,6)
    plt.plot(range(uhist.size), uhist, 'k-',
             range(upred.size), upred, 'r-')
    plt.title('Utility')
    plt.xticks([])
    
    # save high quality version to external file
    plt.savefig('ILALinfig4.eps', format='eps', dpi=2000)
    plt.show(fig4)
    plt.close(fig4)