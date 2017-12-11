#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 06:38:49 2017

@author: klp4
"""
import matplotlib.pyplot as plt

def OLGplots(dataplot, name):
    '''
    This function takes a list of time series from the ILA model and plots and
    saves a series of graphs of these over time.
    
    Inputs:
    -----------  
    dataplot: a list of series to plot
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
    name: a string appended to the begining of the saved plots to identify the
        model and solution method
    
    Outputs:
    -----------  
    no formal outputs, only plots displayed and saved
    
    '''
    
    # turn interactive plotting off
    # plt.ioff()
    
    # unpack data for plots
    [k2pred, k2upp, k2low, k2hist, \
        k3pred, k3upp, k3low, k3hist, \
        k4pred, k4upp, k4low, k4hist,
        l1pred, l1upp, l1low, l1hist, \
        l2pred, l2upp, l2low, l2hist, \
        l3pred, l3upp, l3low, l3hist, \
        zpred, zupp, zlow, zhist, \
        Kpred, Kupp, Klow, Khist, \
        Lpred, Lupp, Llow, Lhist, \
        GDPpred, GDPupp, GDPlow, GDPhist, \
        wpred, wupp, wlow, whist, \
        rpred, rupp, rlow, rhist, \
        T4pred, T4upp, T4low, T4hist, \
        Bpred, Bupp, Blow, Bhist, \
        c1pred, c1upp, c1low, c1hist, \
        c2pred, c2upp, c2low, c2hist, \
        c3pred, c3upp, c3low, c3hist, \
        c4pred, c4upp, c4low, c4hcst, \
        Cpred, Cupp, Clow, Chist, \
        Ipred, Iupp, Ilow, Ihist, \
        u1pred, u1upp, u1low, u1hist, \
        u2pred, u2upp, u2low, u2hist, \
        u3pred, u3upp, u3low, u3hist, \
        u4pred, u4upp, u4low, u4hcst] = dataplot
            
    # plot
    fig1 = plt.figure()
    plt.subplot(3,1,1)
    plt.plot(range(k2pred.size), k2pred, 'k-',
             range(k2upp.size), k2upp, 'k:',
             range(k2low.size), k2low, 'k:')
    plt.title('k2')
    plt.xticks([])
    
    plt.subplot(3,1,2)
    plt.plot(range(k3pred.size), k3pred, 'k-',
             range(k3upp.size), k3upp, 'k:',
             range(k3low.size), k3low, 'k:')
    plt.title('k3')
    plt.xticks([])
    
    plt.subplot(3,1,3)
    plt.plot(range(k4pred.size), k4pred, 'k-',
             range(k4upp.size), k4upp, 'k:',
             range(k4low.size), k4low, 'k:')
    plt.title('k4')

    # save high quality version to external file
    plt.savefig(name + 'fig1.pdf', format='pdf', dpi=2000)
    plt.show(fig1)
    plt.close(fig1)


    fig2 = plt.figure()
    plt.subplot(3,1,1)
    plt.plot(range(l1pred.size), l1pred, 'k-',
             range(l1upp.size), l1upp, 'k:',
             range(l1low.size), l1low, 'k:')
    plt.title('l1')
    plt.xticks([])
    
    plt.subplot(3,1,2)
    plt.plot(range(l2pred.size), l2pred, 'k-',
             range(l2upp.size), l2upp, 'k:',
             range(l2low.size), l2low, 'k:')
    plt.title('l2')
    plt.xticks([])
    
    plt.subplot(3,1,3)
    plt.plot(range(l3pred.size), l3pred, 'k-',
             range(l3upp.size), l3upp, 'k:',
             range(l3low.size), l3low, 'k:')
    plt.title('l3')
    
    # save high quality version to external file
    plt.savefig(name + 'fig2.pdf', format='pdf', dpi=2000)
    plt.show(fig2)
    plt.close(fig2)
    
    
    fig3 = plt.figure()
    plt.subplot(2,2,1)
    plt.plot(range(GDPpred.size), GDPpred, 'k-',
             range(GDPupp.size), GDPupp, 'k:',
             range(GDPlow.size), GDPlow, 'k:')
    plt.title('GDP')
    plt.xticks([])
    
    plt.subplot(2,2,2)
    plt.plot(range(Kpred.size), Kpred, 'k-',
             range(Kupp.size), Kupp, 'k:',
             range(Klow.size), Klow, 'k:')
    plt.title('K')
    
    plt.subplot(2,2,3)
    plt.plot(range(Lpred.size), Lpred, 'k-',
             range(Lupp.size), Lupp, 'k:',
             range(Llow.size), Llow, 'k:')
    plt.title('L')
    plt.xticks([])
    
    plt.subplot(2,2,4)
    plt.plot(range(T4pred.size), T4pred, 'k-',
             range(T4upp.size), T4upp, 'k:',
             range(T4low.size), T4low, 'k:')
    plt.title('T4')
    plt.xticks([])
    
    # save high quality version to external file
    plt.savefig(name + 'fig3.pdf', format='pdf', dpi=2000)
    plt.show(fig3)
    plt.close(fig3)