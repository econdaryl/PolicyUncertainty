# -*- coding: utf-8 -*-
'''
Takes the series of simulations generated using GSSA, LIN, EX, and VFI methods
and puts them into one graph
'''

import numpy as np
import pickle as pkl
import timeit
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from ILArunmc import runmc
from ILAmcanalysis import mcanalysis

infile = open('ILAonesimGSSA.pkl', 'rb')
(khistGSSA, ellhistGSSA, YhistGSSA, whistGSSA, rhistGSSA, ThistGSSA, \
chistGSSA, ihistGSSA, uhistGSSA) = pkl.load(infile)
infile.close()

infile = open('ILAonesimLIN.pkl', 'rb')
(khistLIN, ellhistLIN, YhistLIN, whistLIN, rhistLIN, ThistLIN, \
chistLIN, ihistLIN, uhistLIN) = pkl.load(infile)
infile.close()

infile = open('ILAonesimEX.pkl', 'rb')
(khistEX, ellhistEX, YhistEX, whistEX, rhistEX, ThistEX, \
chistEX, ihistEX, uhistEX) = pkl.load(infile)
infile.close()

infile = open('ILAonesimVFI.pkl', 'rb')
(khistVFI, ellhistVFI, YhistVFI, whistVFI, rhistVFI, ThistVFI, \
chistVFI, ihistVFI, uhistVFI) = pkl.load(infile)
infile.close()
     
# plot
fig1 = plt.figure()
plt.subplot(2,2,1)
plt.plot(range(khistLIN.size), khistLIN, 'k-')
plt.title('Capital LIN')
plt.xticks([])

plt.subplot(2,2,3)
plt.plot(range(khistGSSA.size), khistGSSA, 'k-')
plt.title('Capital GSSA')
plt.xticks([])
    
plt.subplot(2,2,4)
plt.plot(range(khistVFI.size), khistVFI, 'k-')
plt.title('Capital VFI')
plt.xticks([])

# save high quality version to external file
plt.savefig('ILAonesimAll' + '_K.pdf', format='pdf', dpi=2000)
plt.show(fig1)
plt.close(fig1)

fig2 = plt.figure()
plt.subplot(2,2,1)
plt.plot(range(ellhistLIN.size), ellhistLIN, 'k-')
plt.title('Technology LIN')
plt.xticks([])

plt.subplot(2,2,3)
plt.plot(range(ellhistGSSA.size), ellhistGSSA, 'k-')
plt.title('Technology GSSA')
plt.xticks([])
    
plt.subplot(2,2,4)
plt.plot(range(ellhistVFI.size), ellhistVFI, 'k-')
plt.title('Technology VFI')
plt.xticks([])

# save high quality version to external file
plt.savefig('BMonesimAll' + '_L.pdf', format='pdf', dpi=2000)
plt.show(fig2)
plt.close(fig2)

fig3 = plt.figure()
plt.subplot(2,2,1)
plt.plot(range(YhistLIN.size), YhistLIN, 'k-')
plt.title('GDP LIN')
plt.xticks([])

plt.subplot(2,2,3)
plt.plot(range(YhistGSSA.size), YhistGSSA, 'k-')
plt.title('GDP GSSA')
plt.xticks([])
    
plt.subplot(2,2,4)
plt.plot(range(YhistVFI.size), YhistVFI, 'k-')
plt.title('GDP VFI')
plt.xticks([])

# save high quality version to external file
plt.savefig('BMonesimAll' + '_Y.pdf', format='pdf', dpi=2000)
plt.show(fig3)
plt.close(fig3)

fig4 = plt.figure()
plt.subplot(2,2,1)
plt.plot(range(whistLIN.size), whistLIN, 'k-')
plt.title('Wages LIN')
plt.xticks([])

plt.subplot(2,2,3)
plt.plot(range(whistGSSA.size), whistGSSA, 'k-')
plt.title('Wages GSSA')
plt.xticks([])
    
plt.subplot(2,2,4)
plt.plot(range(whistVFI.size), whistVFI, 'k-')
plt.title('Wages VFI')
plt.xticks([])

# save high quality version to external file
plt.savefig('BMonesimAll' + '_w.pdf', format='pdf', dpi=2000)
plt.show(fig4)
plt.close(fig4)

fig5 = plt.figure()
plt.subplot(2,2,1)
plt.plot(range(rhistLIN.size), rhistLIN, 'k-')
plt.title('Interest Rate LIN')
plt.xticks([])

plt.subplot(2,2,3)
plt.plot(range(rhistGSSA.size), rhistGSSA, 'k-')
plt.title('Interest Rate GSSA')
plt.xticks([])
    
plt.subplot(2,2,4)
plt.plot(range(rhistVFI.size), rhistVFI, 'k-')
plt.title('Interest Rate VFI')
plt.xticks([])

# save high quality version to external file
plt.savefig('BMonesimAll' + '_r.pdf', format='pdf', dpi=2000)
plt.show(fig5)
plt.close(fig5)

fig6 = plt.figure()
plt.subplot(2,2,1)
plt.plot(range(ThistLIN.size), ThistLIN, 'k-')
plt.title('Taxes LIN')
plt.xticks([])

plt.subplot(2,2,3)
plt.plot(range(ThistGSSA.size), ThistGSSA, 'k-')
plt.title('Taxes GSSA')
plt.xticks([])
    
plt.subplot(2,2,4)
plt.plot(range(ThistVFI.size), ThistVFI, 'k-')
plt.title('Taxes VFI')
plt.xticks([])

# save high quality version to external file
plt.savefig('BMonesimAll' + '_T.pdf', format='pdf', dpi=2000)
plt.show(fig6)
plt.close(fig6)

fig7 = plt.figure()
plt.subplot(2,2,1)
plt.plot(range(chistLIN.size), chistLIN, 'k-')
plt.title('Consumption LIN')
plt.xticks([])

plt.subplot(2,2,3)
plt.plot(range(chistGSSA.size), chistGSSA, 'k-')
plt.title('Consumption GSSA')
plt.xticks([])
    
plt.subplot(2,2,4)
plt.plot(range(chistVFI.size), chistVFI, 'k-')
plt.title('Consumption VFI')
plt.xticks([])

# save high quality version to external file
plt.savefig('BMonesimAll' + '_c.pdf', format='pdf', dpi=2000)
plt.show(fig7)
plt.close(fig7)

fig8 = plt.figure()
plt.subplot(2,2,1)
plt.plot(range(ihistLIN.size), ihistLIN, 'k-')
plt.title('Investment LIN')
plt.xticks([])

plt.subplot(2,2,3)
plt.plot(range(ihistGSSA.size), ihistGSSA, 'k-')
plt.title('Investment GSSA')
plt.xticks([])
    
plt.subplot(2,2,4)
plt.plot(range(ihistVFI.size), ihistVFI, 'k-')
plt.title('Investment VFI')
plt.xticks([])

# save high quality version to external file
plt.savefig('BMonesimAll' + '_i.pdf', format='pdf', dpi=2000)
plt.show(fig8)
plt.close(fig8)

fig9 = plt.figure()
plt.subplot(2,2,1)
plt.plot(range(uhistLIN.size), uhistLIN, 'k-')
plt.title('Utility LIN')
plt.xticks([])

plt.subplot(2,2,3)
plt.plot(range(uhistGSSA.size), uhistGSSA, 'k-')
plt.title('Utility GSSA')
plt.xticks([])
    
plt.subplot(2,2,4)
plt.plot(range(uhistVFI.size), uhistVFI, 'k-')
plt.title('Utility VFI')
plt.xticks([])

# save high quality version to external file
plt.savefig('BMonesimAll' + '_u.pdf', format='pdf', dpi=2000)
plt.show(fig9)
plt.close(fig9)