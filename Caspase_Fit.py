# -*- coding: utf-8 -*-
"""
Created on Mon Aug 15 10:39:55 2016

@author: Agus
"""

import numpy as np
import itertools

import matplotlib.pyplot as plt

from scipy.optimize import minimize
from scipy.signal import savgol_filter
from scipy.interpolate import splrep, splev
from scipy.integrate import ode

#Define normalizing function
def Normalize(vect):
    vect -= min(vect)
    vect /= max(vect)
    return vect

# Define Delta functions and sigmoid derivative
def D(m):
    return (max(m)-m)/2

def Deltam(m, interpolate=False, timepoints=10):
    der_sav = savgol_filter(m, 5, 2, deriv = 1, delta = timepoints)
    
    if interpolate:
        t = np.arange(0, timepoints*(len(m)), timepoints)
        f = splrep(t, der_sav, k=3, s=0)
        der_sav = splev(np.arange(0,max(t)), f, der=0)
    
    return der_sav

def Deltam_m(m, interpolate=False, timepoints=10):
    #m = Normalize(m)
    der_sav = savgol_filter(m, 5, 2, deriv = 1, delta = timepoints)
    der_sav_m = np.zeros(len(der_sav))
    der_sav_m.fill(np.nan)
    der_sav_m[m>0.01] = der_sav[m>0.01]/m[m>0.01]
    
    if interpolate:
        t = np.arange(0, timepoints*(len(m)), timepoints)
        f = splrep(t[np.isfinite(der_sav_m)], der_sav_m[np.isfinite(der_sav_m)], k=3, s=0)
        der_sav = splev(np.arange(0,max(t)), f, der=0)
    
    return der_sav

def Deltad(d, interpolate=False, timepoints=10):
    der_sav = -savgol_filter(d, 5, 2, deriv = 1, delta = timepoints)
    if interpolate:
        t = np.arange(0, timepoints*(len(d)), timepoints)
        f = splrep(t, der_sav, k=3, s=0)
        der_sav = splev(np.arange(0,max(t)), f, der=0)
    return der_sav

def Deltad_d(d, interpolate=False, timepoints=10):
    #d = Normalize(d)
    der_sav = -savgol_filter(d, 5, 2, deriv = 1, delta = timepoints)
    der_sav_d = np.ones(len(der_sav))
    der_sav_d.fill(np.nan)
    der_sav_d[d>0.01] = der_sav[d>0.01]/d[d>0.01]
    
    if interpolate:
        t = np.arange(0, timepoints*(len(d)), timepoints)
        f = splrep(t[np.isfinite(der_sav_d)], der_sav_d[np.isfinite(der_sav_d)], k=3, s=0)
        der_sav = splev(np.arange(0,max(t)), f, der=0)
    
    return der_sav

# Define fit with mass action simulation

def sigmoid(x, base, amplitude, rate, x0):
    y = base + amplitude / (1 + np.exp(-rate*(x-x0)))
    return y

def sigmoid_der1(x, base, amplitude, rate, x0):
    """
    first derivative of sigmoid function
    """
    return (amplitude*rate * np.exp(-rate*(x-x0))) / ((1 + np.exp(-rate*(x-x0)))**2)

def gompertz(t, base, amplitude, k, tc):
    return amplitude * np.exp(- np.exp (-k * (t- tc))) + base

def gompertz_der1(t, base, amplitude, k, tc):
    return k * amplitude * np.exp(- np.exp (k * (t- tc)) - tc * k + t * k)

def rhs(t, x, params):
    t = np.asarray(t)
    
    t_0, rate, k = params[0], params[1], params[2]
    
    try:
        xp = np.zeros([3, len(t)])
        
        xp[0, :] = sigmoid_der1(t, 0, 1, rate, t_0) # e
    
        xp[1, :] = - k * x[0] * x[1] # s
    
        xp[2, :] = 2 * k * x[0] * x[1] # p
        
    except:
        xp = np.zeros(3)
    
        xp[0] = sigmoid_der1(t, 0, 1, rate, t_0) # e
    
        xp[1] = - k * x[0] * x[1] # s
    
        xp[2] = 2 * k * x[0] * x[1] # p
    
    return xp

def rhs_g(t, x, params):
    t = np.asarray(t)
    
    t_0, rate, k = params[0], params[1], params[2]
    
    try:
        xp = np.zeros([3, len(t)])
        
        xp[0, :] = gompertz_der1(t, 0, 1, rate, t_0) # e
    
        xp[1, :] = - k * x[0] * x[1] # s
    
        xp[2, :] = 2 * k * x[0] * x[1] # p
        
    except:
        xp = np.zeros(3)
    
        xp[0] = gompertz_der1(t, 0, 1, rate, t_0) # e
    
        xp[1] = - k * x[0] * x[1] # s
    
        xp[2] = 2 * k * x[0] * x[1] # p
    
    return xp

def simulate(max_time, t_0, rate, k):
    simulation = ode(rhs).set_integrator('dopri5')
    
    x0 = [0, 0.5, 0]
    t0 = 0
    
    params = [t_0, rate/100, k/100]
    
    simulation.set_initial_value(x0, t0).set_f_params(params)
    dt = 1
    
    C = []
    S = []
    P = []
    
    while simulation.successful() and simulation.t < max_time:
        simulation.integrate(simulation.t+dt)
        c, s, p = simulation.y[0], simulation.y[1], simulation.y[2]
        C.append(c)
        S.append(s)
        P.append(p)
    
    C = np.asarray(C)
    S = np.asarray(S)
    P = np.asarray(P)
    
    return C, S, P

def fit_caspase(m, time_estimate, timepoints=10, Plot=False, fix_rate=None, fix_k=None):
    ms = Normalize(m)
    #ds = (1-ms)/2
    #deltams = savgol_filter(ms, 5, 2, deriv = 1, delta = timepoints)
    #deltams = Normalize(deltams)
    max_temp = len(m)*timepoints
    
    time_coarse = np.arange(0, len(m)*timepoints, timepoints)
    time_fine = np.arange(0, len(m)*timepoints)
    
    def chi_2(params):
        t_0, rate, k = params[0], params[1], params[2]
        if fix_rate is not None:
            rate = fix_rate
        if fix_k is not None:
            k = fix_k
        
        casp, sens, prod = simulate(max_temp, t_0, rate, k)
        
        out = 0.    
        for n, m in enumerate(ms):
            if m<0.4:
                out += ((prod[n*timepoints] - m) ** 2)*4
            else:
                out += (prod[n*timepoints] - m) ** 2
    
        return out
    
    Initial_parameters = np.ones(3)
    Initial_parameters[0] = time_estimate - 30 # t0 for caspase
    Initial_parameters[1] = 0.1 # caspase rate
    Initial_parameters[2] = 0.1 # k
    #Se podrian cambiar esos initial parameters
    
    bounds = [(0, 100)] * len(Initial_parameters)
    bounds[1:3] = [(0, 50)] * 2
    bounds[0] = (0, 300) #Probar cambiar a 900
    #bounds[1] = (0, 30)
    
    mini = minimize(chi_2, x0=Initial_parameters, bounds=bounds)#, options = option)
    
    t_0, rate, k = mini.x[0], mini.x[1], mini.x[2]
    casp, sens, prod = simulate(max_temp, t_0, rate, k)
    
    chi = chi_2([t_0, rate, k])
    #option = {'maxiter' : 5}
    Other_Initial_parameters = [10, 50]
    for Initial_parameters[1], Initial_parameters[2] in itertools.permutations(Other_Initial_parameters):
        mini = minimize(chi_2, x0=Initial_parameters, bounds=bounds)#, options = option)
        
        this_t_0, this_rate, this_k = mini.x[0], mini.x[1], mini.x[2]
        this_casp, this_sens, this_prod = simulate(max_temp, t_0, rate, k)
        _chi = chi_2([this_t_0, this_rate, this_k])
        
        if _chi<chi:
            t_0, rate, k = this_t_0, this_rate, this_k
            casp, sens, prod = this_casp, this_sens, this_prod
            chi = _chi
    
    if Plot:
        # Plot m
        plt.plot(time_coarse, ms, 'ob', label='m')
        plt.plot(time_fine, prod, 'b', label='m fit')
        plt.legend(loc=2)
        plt.xlabel('tiempo (minutos)')
        plt.ylabel('m prop')
        plt.show()
        
        # Plot d
        plt.plot(time_coarse, ds, 'ob', label='d')
        plt.plot(time_fine, sens, 'b', label='d fit')
        plt.legend(loc=2)
        plt.xlabel('tiempo (minutos)')
        plt.ylabel('d prop')
        plt.show()
        
        # Plot deltam
        plt.plot(time_coarse, ms, 'ob', label='m')
        plt.plot(time_fine, casp, 'r', label='caspase fit')
        plt.legend(loc=2)
        plt.xlabel('tiempo (minutos)')
        plt.ylabel('prop')
        plt.show()
    
    Sol = {'t0':t_0, 'rate':rate, 'k':k, 'casp':casp, 'sens':sens, 'prod':prod, 'chi':chi}
    return Sol