#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  9 17:05:09 2018

@author: klp4
"""
import pickle as pkl

name = 'ILAsimLIN_KH'

# load steady state values and parameters
infile = open(name + '.pkl', 'rb')
timesim = pkl.load(infile)
alldata = pkl.load(infile)

infile.close()

(preddata, avgdata, uppdata, lowdata, foreperc, forevarc, zforperc, \
           zforvarc, RMsqEerravg, act) = alldata