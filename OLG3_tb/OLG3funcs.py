'''
Modeldefs and ModelDyn functions for Simple OLG Model
'''
import numpy as np


def Modeldefs(Xp, X, Y, Z, params):
    '''
    This function takes vectors of endogenous and exogenous state variables
    along with a vector of 'jump' variables and returns explicitly defined
    values for consumption, gdp, wages, real interest rates, and transfers
    
    Inputs are:
        Xp: value of capital in next period
        X: value of capital this period
        Y: value of labor this period
        Z: value of productivity this period
        params: list of parameter values
    
    Outputs are:
        GDP: GDP
        w: wage rate
        r: rental rate on capital
        T: transfer payments
        c: consumption
        i: investment
        u: utiity
    '''
    # unpack params
    [alpha, beta, gamma, delta, chi, theta, tau, rho, sigma, pi2, pi3, \
        f1, f2, nx, ny, nz] = params
     
    nx = int(nx)
    ny = int(ny)
    nz = int(nz)
    
    # unpack input vectors
    [k2p, k3p] = Xp
    [k2, k3] = X
    [l1, l2] = Y
    
#    if l1 > 0.9999:
#        l1 = 0.9999
#    elif l1 < 0.0001:
#        l1 = 0.0001
#        
#    if l2 > 0.9999:
#        l2 = 0.9999
#    elif l2 < 0.0001:
#        l2 = 0.0001
#        
#    if l3 > 0.9999:
#        l3 = 0.9999
#    elif l3 < 0.0001:
#        l3 = 0.0001
    z = Z
    
    K = k2 + pi2*k3
    L = f1*l1 + pi2*f2*l2
   
    # find definintion values
    GDP = K**alpha*(np.exp(z)*L)**(1-alpha)
    w = (1-alpha)*GDP/L
    r = alpha*GDP/K
    T3 = tau*w*L
    B = (1+r-delta)*((1-pi2)*k2 + (1-pi3)*pi2*k3) /(1+pi2+pi3)
    c1 = (1-tau)*(w*f1*l1) + B - k2p
    c2 = (1-tau)*(w*f2*l2) + B + (1+r-delta)*k2 - k3p
    c3 = (1+r-delta)*k3 + B + T3
    C = c1 + pi2*c2 + pi3*c3
    I = GDP - C
    u1 = c1**(1-gamma)/(1-gamma) - chi*l1**(1+theta)/(1+theta)
    u2 = c2**(1-gamma)/(1-gamma) - chi*l2**(1+theta)/(1+theta)
    u3 = c3**(1-gamma)/(1-gamma)
    
    return K, L, GDP, w, r, T3, B, c1, c2, c3, C, I, u1, u2, u3


def Modeldyn(inmat, params):
    '''
    This function takes vectors of endogenous and exogenous state variables
    along with a vector of 'jump' variables and returns values from the
    characterizing Euler equations.
    
    Inputs are:
        theta: a vector containng (Xpp, Xp, X, Yp, Y, Zp, Z) where:
            Xpp: value of capital stocks in two periods
            Xp: value of capital stocks in next period
            X: value of capital stocks this period
            Yp: value of labors in next period
            Y: value of labors this period
            Zp: value of productivity in next period
            Z: value of productivity this period
        params: list of parameter values
    
    Output are:
        Euler: a vector of Euler equations written so that they are zero at the
            steady state values of X, Y & Z.  This is a 2x1 numpy array. 
    '''
    
    # unpack params
    [alpha, beta, gamma, delta, chi, theta, tau, rho, sigma, pi2, pi3, \
        f1, f2, nx, ny, nz] = params
    nx = int(nx)
    ny = int(ny)
    nz = int(nz)
    
    # unpack inmat
    Xpp = inmat[0:nx]
    Xp =  inmat[nx:2*nx]
    X =   inmat[2*nx:3*nx]
    Yp =  inmat[3*nx:3*nx+ny]
    Y =   inmat[3*nx+ny:3*nx+2*ny]
    Zp =  inmat[3*nx+2*ny]
    Z =   inmat[3*nx+2*ny+1]
    
    # unpack params

    # find definitions for now and next period
    [l1, l2] = Y
    
    if l1 > 0.9999:
        l1 = 0.9999
    elif l1 < 0.0001:
        l1 = 0.0001
        
    if l2 > 0.9999:
        l2 = 0.9999
    elif l2 < 0.0001:
        l2 = 0.0001
        
    K, L, GDP, w, r, T3, B, c1, c2, c3, C, I, u1, u2, u3 = \
        Modeldefs(Xp, X, Y, Z, params)
    Kp, Lp, GDPp, wp, rp, T3p, Bp, c1p, c2p, c3p, Cp, Ip, u1p, u2p, \
        u3p = Modeldefs(Xpp, Xp, Yp, Zp, params)
    
    # Evaluate Euler equations
    El1 = (c1**(-gamma)*(1-tau)*w*f1) / (chi*l1**theta) - 1
    El2 = (c2**(-gamma)*(1-tau)*w*f2) / (chi*l2**theta) - 1
    Ek2 = (c1**(-gamma)) / (beta*c2p**(-gamma)*(1 + rp - delta)) - 1
    Ek3 = (c2**(-gamma)) / (beta*c3p**(-gamma)*(1 + rp - delta)) - 1

    return np.array([El1, El2, Ek2, Ek3])