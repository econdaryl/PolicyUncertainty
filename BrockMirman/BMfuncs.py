'''
Modeldefs and ModelDyn functions for Simple ILA Model
'''
import numpy as np


def Modeldefs(Xp, X, Z, params):
    '''
    This function takes vectors of endogenous and exogenous state variables
    along with a vector of 'jump' variables and returns explicitly defined
    values for consumption, gdp, wages, real interest rates, and transfers
    
    Inputs are:
        Xp: value of capital in next period
        X: value of capital this period
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
    
    # unpack input vectors
    kp = Xp
    k = X
    z = Z
    
    # unpack params
    [alpha, beta, tau, rho, sigma] = params
    
    # find definintion values
    GDP = k**alpha*np.exp(z)
    w = (1-alpha)*GDP*(1-tau)
    r = alpha*GDP*(1-tau)/k
    T = tau*GDP
    c = (1-tau)*GDP - kp
    i = GDP - c
    u = np.log(c)
    return GDP, w, r, T, c, i, u


def Modeldyn(theta0, params):
    '''
    This function takes vectors of endogenous and exogenous state variables
    along with a vector of 'jump' variables and returns values from the
    characterizing Euler equations.
    
    Inputs are:
        theta: a vector containng (Xpp, Xp, X, Yp, Y, Zp, Z) where:
            Xpp: value of capital in two periods
            Xp: value of capital in next period
            X: value of capital this period
            Zp: value of productivity in next period
            Z: value of productivity this period
        params: list of parameter values
    
    Output are:
        Euler: a vector of Euler equations written so that they are zero at the
            steady state values of X, Y & Z.  This is a 2x1 numpy array. 
    '''
    
    # unpack theat0
    (Xpp, Xp, X, Zp, Z) = theta0
    
    # unpack params
    [alpha, beta, tau, rho, sigma] = params
    
    # find definitions for now and next period

    GDP, w, r, T, c, i, u = Modeldefs(Xp, X, Z, params)
    GDPp, wp, rp, Tp, cp, ip, up = Modeldefs(Xpp, Xp, Zp, params)
    
    # Evaluate Euler equations
    E1 = (c**(-1)) / (beta*cp**(-1)*rp) - 1
    
    return np.array([E1])