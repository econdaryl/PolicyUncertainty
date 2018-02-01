#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 06:38:49 2017

@author: klp4
"""
import matplotlib.pyplot as plt

def BMonesimplots(dataplot, name):
    '''
    This function takes a list of time series from the BM model and plots and
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
    [khist, Yhist, whist, rhist, Thist, chist, ihist, uhist] = dataplot
            
    # plot
    fig1 = plt.figure()
    plt.subplot(2,2,1)
    plt.plot(range(khist.size), khist, 'k-')
    plt.title('Capital')
    plt.xticks([])
    
    plt.subplot(2,2,3)
    plt.plot(range(Yhist.size), Yhist, 'k-')

    plt.title('GDP')
    plt.xticks([])
        
    plt.subplot(2,2,4)
    plt.plot(range(Thist.size), Thist, 'k-')
    plt.title('Taxes')
    plt.xticks([])
    
    # save high quality version to external file
    plt.savefig(name + 'fig1.pdf', format='pdf', dpi=2000)
    plt.show(fig1)
    plt.close(fig1)


    fig2 = plt.figure()
    plt.subplot(2,2,1)
    plt.plot(range(whist.size), whist, 'k-')
    plt.title('Wages')
    plt.xticks([])
    
    plt.subplot(2,2,2)
    plt.plot(range(rhist.size), rhist, 'k-')
    plt.title('Interest Rate')
    plt.xticks([])
    
    plt.subplot(2,2,3)
    plt.plot(range(chist.size), chist, 'k-')
    plt.title('Consumption')
    plt.xticks([])
    
    plt.subplot(2,2,4)
    plt.plot(range(ihist.size), ihist, 'k-')
    plt.title('Investment')
    plt.xticks([])
    
    # save high quality version to external file
    plt.savefig(name + 'fig2.pdf', format='pdf', dpi=2000)
    plt.show(fig2)
    plt.close(fig2)