#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 06:38:49 2017

@author: klp4
"""
def ILAplots(data, name):
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
    
    
# plot
plt.subplot(2,2,1)
plt.plot(range(kpred.size), kpred/kbar, 'k-',
         range(kupp.size), kupp/kbar, 'k:',
         range(klow.size), klow/kbar, 'k:')
plt.title('k')

plt.subplot(2,2,2)
plt.plot(range(ellpred.size), ellpred/ellbar, 'k-',
         range(ellupp.size), ellupp/ellbar, 'k:',
         range(elllow.size), elllow/ellbar, 'k:')
plt.title('ell')

plt.subplot(2,2,3)
plt.plot(range(zpred.size), zpred, 'k-',
         range(zupp.size), zupp, 'k:',
         range(zlow.size), zlow, 'k:')
plt.title('z')

plt.subplot(2,2,4)
plt.plot(range(Ypred.size), Ypred/Ybar, 'k-',
         range(Yupp.size), Yupp/Ybar, 'k:',
         range(Ylow.size), Ylow/Ybar, 'k:')
plt.title('Y')

# save high quality version to external file
plt.savefig('ILALinfig1.eps', format='eps', dpi=2000)

plt.show()

plt.subplot(3,2,1)
plt.plot(range(wpred.size), wpred/wbar, 'k-',
         range(wupp.size), wupp/wbar, 'k:',
         range(wlow.size), wlow/wbar, 'k:')
plt.title('w')

plt.subplot(3,2,2)
plt.plot(range(rpred.size), rpred/rbar, 'k-',
         range(rupp.size), rupp/rbar, 'k:',
         range(rlow.size), rlow/rbar, 'k:')
plt.title('r')

plt.subplot(3,2,3)
plt.plot(range(Tpred.size), Tpred/Tbar, 'k-',
         range(Tupp.size), Tupp/Tbar, 'k:',
         range(Tlow.size), Tlow/Tbar, 'k:')
plt.title('T')

plt.subplot(3,2,4)
plt.plot(range(cpred.size), cpred/cbar, 'k-',
         range(cupp.size), cupp/cbar, 'k:',
         range(clow.size), clow/cbar, 'k:')
plt.title('c')

plt.subplot(3,2,5)
plt.plot(range(ipred.size), ipred/ibar, 'k-',
         range(iupp.size), iupp/ibar, 'k:',
         range(ilow.size), ilow/ibar, 'k:')
plt.title('iT')

plt.subplot(3,2,6)
plt.plot(range(upred.size), upred/ubar, 'k-',
         range(uupp.size), uupp/ubar, 'k:',
         range(ulow.size), ulow/ubar, 'k:')
plt.title('u')

# save high quality version to external file
plt.savefig('ILALinfig2.eps', format='eps', dpi=2000)

plt.show()

# plot
plt.subplot(2,2,1)
plt.plot(range(khist.size), khist/kbar, 'k-',
         range(kpred.size), kpred/kbar, 'r-')
plt.title('k')

plt.subplot(2,2,2)
plt.plot(range(ellhist.size), ellhist/ellbar, 'k-',
         range(ellpred.size), ellpred/ellbar, 'r-')
plt.title('ell')

plt.subplot(2,2,3)
plt.plot(range(zhist.size), zhist, 'k-',
         range(zpred.size), zpred, 'r-')
plt.title('z')

plt.subplot(2,2,4)
plt.plot(range(Yhist.size), Yhist/Ybar, 'k-',
         range(Ypred.size), Ypred/Ybar, 'r-')
plt.title('Y')

# save high quality version to external file
plt.savefig('ILALinfig3.eps', format='eps', dpi=2000)

plt.show()

plt.subplot(3,2,1)
plt.plot(range(whist.size), whist/wbar, 'k-',
         range(wpred.size), wpred/wbar, 'r-')
plt.title('w')

plt.subplot(3,2,2)
plt.plot(range(rhist.size), rhist/rbar, 'k-',
         range(rpred.size), rpred/rbar, 'r-')
plt.title('r')

plt.subplot(3,2,3)
plt.plot(range(Thist.size), Thist/Tbar, 'k-',
         range(Tpred.size), Tpred/Tbar, 'r-')
plt.title('T')

plt.subplot(3,2,4)
plt.plot(range(chist.size), chist/cbar, 'k-',
         range(cpred.size), cpred/cbar, 'r-')
plt.title('c')

plt.subplot(3,2,5)
plt.plot(range(ihist.size), ihist/ibar, 'k-',
         range(ipred.size), ipred/ibar, 'r-')
plt.title('iT')

plt.subplot(3,2,6)
plt.plot(range(uhist.size), uhist/ubar, 'k-',
         range(upred.size), upred/ubar, 'r-')
plt.title('u')

# save high quality version to external file
plt.savefig('ILALinfig4.eps', format='eps', dpi=2000)

plt.show()