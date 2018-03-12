#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  9 17:05:09 2018

@author: klp4
"""
import pickle as pkl

name = 'BMsolveVFI'

# load steady state values and parameters
infile = open(name + '.pkl', 'rb')
(coeffs1, coeffs2, timesolve) = pkl.load(infile)

infile.close()