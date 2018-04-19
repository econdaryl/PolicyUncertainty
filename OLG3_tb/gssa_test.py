import numpy as np
from gssa import XYfunc, poly1
import matplotlib.pyplot as plt

def GSSA_test(params, kbar, ellbar, GSSAparams, old_coeffs):
    regtype = 'poly1' # functional form for X & Y functions 
    fittype = 'MVOLS'   # regression fitting method
    ccrit = 1.0E-8  # convergence criteria for XY change
    damp = 0.01  # damping paramter for fixed point algorithm
    
    [alpha, beta, gamma, delta, chi, theta, tau, rho, \
    sigma, pi2, pi3, f1, f2, nx, ny, nz] = params
    (T, nx, ny, nz, pord, old) = GSSAparams
    cnumb = int((pord+1)*(nx+nz) + .5*(nx+nz-1)*(nx+nz-2))
    cnumb2 = int(3*(nx+nz) + .5*(nx+nz-1)*(nx+nz-2))
    (kbar2, kbar3) = kbar
    (ellbar1, ellbar2) = ellbar
    nx = int(nx)
    ny = int(ny)
    nz = int(nz)
    XYparams = (pord, nx, ny, nz)

    Xstart = kbar
    
    #create history of Z's

    if regtype == 'poly1' and old == False:
        coeffs = np.array([[0.05*kbar2, 0.05*kbar3, 0.95*ellbar1, 0.95*ellbar2], \
                           [0.95, 0., 0., 0.], \
                           [0., 0.95, 0., 0.], \
                           [0., 0., 0.05*ellbar1, 0.05*ellbar2], \
                           [0., 0., 0., 0.], \
                           [0., 0., 0., 0.], \
                           [0., 0., 0., 0.], \
                           [0., 0., 0., 0.], \
                           [0., 0., 0., 0.], \
                           [0., 0., 0., 0.]])
    elif old == True:
        coeffs = old_coeffs
    
    if old == False & pord > 2:
        A = np.zeros((cnumb-cnumb2, nx+ny))
        coeffs = np.insert(coeffs, cnumb2-1, A, axis=0)
        
    dist = 1000.
    distold = 2.
    count = 0
    damp = .01
    XYold = np.ones((T-1, nx+ny))

    while dist > ccrit and count < 10000:
        Z = np.zeros((T,nz))
        for t in range(1,T):
            Z[t,:] = rho*Z[t-1] + np.random.randn(1)*sigma
        count = count + 1
        X = np.zeros((T+1, nx))
        Y = np.zeros((T, ny))
        Xin = np.zeros((T, nx+nz))
        A = np.exp(Z)
        x = np.zeros((T,(cnumb)))
        X[0, :], Y[0, :] = XYfunc(Xstart, Z[0], XYparams, coeffs)
        for t in range(1,T+1):
            X[t, :], Y[t-1, :] = XYfunc(X[t-1, :], Z[t-1, :], XYparams, coeffs)
            Xin[t-1,:] = np.concatenate((X[t-1], A[t-1]))
            x[t-1,:] = poly1(Xin[t-1,:], XYparams)
        X1 = X[0:T, :]
        #El1 = np.zeros((T-1, 1))
        #El2 = np.zeros((T-1, 1))
        #Ek2 = np.zeros((T-1, 1))
        #Ek3 = np.zeros((T-1, 1))
        #for t in range(0, T-1):
        #    inmat = np.concatenate((X[t+2, :], X[t+1, :], X[t, :], Y[t+1, :], Y[t, :], Z[t+1, :], Z[t, :]))
        #    El1[t], El2[t], Ek2[t], Ek3[t] = (Modeldyn(inmat, params) + 1)
        #    Ek2[t] = 1/Ek2[t]
        #    Ek3[t] = 1/Ek3[t]
        #Gam = np.hstack((Ek2, Ek3))
        #Lam = np.hstack((El1, El2))
        # plot time series
        if count % 1 == 0:
            timeperiods = np.asarray(range(0,T))
            plt.subplot(2,1,1)
            plt.plot(timeperiods, X1-kbar, label='X')
            #plt.axhline(y=kbar2, color='k')
            #plt.axhline(y=kbar3, color='r')
            plt.subplot(2,1,2)
            plt.plot(timeperiods, Y-ellbar, label='Y')
            #plt.axhline(y=ellbar1, color='k')
            #plt.axhline(y=ellbar2, color='r')
            plt.xlabel('time')
            plt.legend(loc=9, ncol=(nx+ny))
            plt.show()    
        
        #Generate Gamma and lambda series
        k2 = np.reshape(X[:, 0], (T+1, 1))
        k3 = np.reshape(X[:, 1], (T+1, 1))
        l1 = np.reshape(Y[:, 0], (T, 1))
        l2 = np.reshape(Y[:, 1], (T, 1))
        for t in range(0, T):
            if l1[t] > 0.9999:
                l1[t] = 0.9999
            elif l1[t] < 0.0001:
                l1[t] = 0.0001
            if l2[t] > 0.9999:
                l2[t] = 0.9999
            elif l2[t] < 0.0001:
                l2[t] = 0.0001  
        K = k2 + pi2*k3
        L = f1*l1 + pi2*f2*l2
        GDP = K[0:T]**alpha*(A*L)**(1-alpha)
        r = alpha*GDP / K[0:T]
        w = (1-alpha)*GDP / L
        T3 = tau*w*L
        B = (1+r-delta)*((1-pi2)*k2[0:T] + (1-pi3)*pi2*k3[0:T]) /(1+pi2+pi3)
        c1 = (1-tau)*(w*f1*l1) + B - k2[1:T+1]
        c2 = (1-tau)*(w*f2*l2) + B + (1+r-delta)*k2[0:T] - k3[1:T+1]
        c3 = (1+r-delta)*k3[0:T] + B + T3
        # T-by-1
        El1 = (c1[0:T-1]**(-gamma)*(1-tau)*w[0:T-1]*f1) / (chi*l1[0:T-1]**theta)
        El2 = (c2[0:T-1]**(-gamma)*(1-tau)*w[0:T-1]*f2) / (chi*l2[0:T-1]**theta)
        Ek2 = (beta*c2[1:T]**(-gamma)*(1 + r[1:T] - delta)) / (c1[0:T-1]**(-gamma))
        Ek3 = (beta*c3[1:T]**(-gamma)*(1 + r[1:T] - delta)) / (c2[0:T-1]**(-gamma))
        # T-1-by-1
        Gam = np.hstack((Ek2, Ek3))
        #print('Gam', Gam)
        Lam = np.hstack((El1, El2))

        # update values for X and Y
        temp1 = np.mean(Ek2)
        temp2 = np.mean(Ek3)
        temp3 = np.mean(El1)
        temp4 = np.mean(El2)
        Xnew = (Gam)*X[1:T, :]
        Ynew = (Lam)*Y[1:T, :]
        XY = np.append(Xnew, Ynew, axis = 1)
        x = x[0:T-1,:]
        
        if fittype == 'MVOLS':
            coeffsnew = MVOLS(XY, x)
        
        dist = np.mean(np.abs(1-XY/XYold))
        if count % 100 == 1:
            print('count ', count, 'distance', dist, \
              'Gam', temp1, temp2, 'Lam', temp3, temp4)
        
        if dist < distold:
            damp = damp*1.05
        else:
            damp = damp*.8
        
        if damp > 1.:
            damp = 1.
        elif damp < .001:
            damp = .001
        
        distold = 1.*dist
        '''
        # update coeffs
        XYold = XY*1.
        coeffs = (1-damp)*coeffs + damp*coeffsnew
        if count % 100 == 0:
            print('coeffs', coeffs)
        '''
    return coeffs