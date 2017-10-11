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
    """
    Returns a normalization of vect from 0 to 1.
    """
    this_vect = vect[:]
    this_vect -= min(this_vect)
    this_vect /= max(this_vect)
    return this_vect

# Define Delta functions and sigmoid derivative
def D(m):
    """
    Calculates dimer fraction from m fraction curve, using maximum m as unity.
    """
    return (max(m)-m)/2

def Deltam(m, interpolate=False, timepoints=10):
    """
    Calculates monomer fraction derivative using Savitzky Golay filter.
    
    Parameters
    ----------
    m : Array-like
        Monomer fraction curve.
    interpolate : boolean, optional
        if True, interpolation over the derivation is done. Defaults to False.
    timepoints : float
        Time span of m curve. Defaults to 10.
    
    Returns
    -------
    der_sav : Array-like
        Savitzky-Golay filtered and derived m curve, interpolated if interpolate is True.
    """
    der_sav = savgol_filter(m, 5, 2, deriv = 1, delta = timepoints)
    
    if interpolate:
        t = np.arange(0, timepoints*(len(m)), timepoints)
        f = splrep(t, der_sav, k=3, s=0)
        der_sav = splev(np.arange(0,max(t)), f, der=0)
    
    return der_sav

def Deltam_m(m, interpolate=False, timepoints=10):
    """
    Calculates monomer fraction derivative divided by monomer fraction at each timepoint
    using Savitzky Golay filter. If monomer fraction is smaller than 0.01, then nans are 
    returned for those values.
    
    Parameters
    ----------
    m : Array-like
        Monomer fraction curve.
    interpolate : boolean, optional
        if True, interpolation over the derivation is done. Defaults to False.
    timepoints : float
        Time span of m curve. Defaults to 10.
    
    Returns
    -------
    der_sav : Array-like
        Savitzky-Golay filtered and derived m curve, interpolated if interpolate is True.
    """
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
    """
    Calculates dimer fraction derivative using Savitzky Golay filter.
    
    Parameters
    ----------
    d : Array-like
        Dimer fraction curve.
    interpolate : boolean, optional
        if True, interpolation over the derivation is done. Defaults to False.
    timepoints : float
        Time span of d curve. Defaults to 10.
    
    Returns
    -------
    der_sav : Array-like
        Savitzky-Golay filtered and derived d curve, interpolated if interpolate is True.
    """
    der_sav = -savgol_filter(d, 5, 2, deriv = 1, delta = timepoints)
    if interpolate:
        t = np.arange(0, timepoints*(len(d)), timepoints)
        f = splrep(t, der_sav, k=3, s=0)
        der_sav = splev(np.arange(0,max(t)), f, der=0)
    return der_sav

def Deltad_d(d, interpolate=False, timepoints=10):
    """
    Calculates dimer fraction derivative divided by dimer fraction at each timepoint
    using Savitzky Golay filter. If dimer fraction is smaller than 0.01, then nans are 
    returned for those values.
    
    Parameters
    ----------
    d : Array-like
        Dimer fraction curve.
    interpolate : boolean, optional
        if True, interpolation over the derivation is done. Defaults to False.
    timepoints : float
        Time span of d curve. Defaults to 10.
    
    Returns
    -------
    der_sav : Array-like
        Savitzky-Golay filtered and derived d curve, interpolated if interpolate is True.
    """
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
    """
    Sigmoid function.
    """
    y = base + amplitude / (1 + np.exp(-rate*(x-x0)))
    return y

def sigmoid_der1(x, base, amplitude, rate, x0):
    """
    First derivative of sigmoid function.
    """
    return (amplitude*rate * np.exp(-rate*(x-x0))) / ((1 + np.exp(-rate*(x-x0)))**2)

def gompertz(t, base, amplitude, k, tc):
    """
    Gompertz function
    """
    return amplitude * np.exp(- np.exp (-k * (t- tc))) + base

def gompertz_der1(t, base, amplitude, k, tc):
    """
    First derivative of gompertz function.
    """
    return k * amplitude * np.exp(- np.exp (k * (t- tc)) - tc * k + t * k)

def rhs(t, x, params, function='sig'):
    """
    Differential equations representing the simple enzyme, substrate product system. 
    No substrate:product complex is considered. Enzyme source may be any function proposed
    or a sigmoid or gompertz function can be used.
    
    Parameters
    ----------
    t : Array-like or float
        Time.
    x : Array-like
        Values for enzyme, substrate and product in actual time.
    params : Array-like, list
        Parameters to be passed to enzyme source function.
    function : string or function, optional
        'sig' : (Default) sigmoid function for enzyme source.
        'gom' : gompertz function for enzyme source.
        func : function to be used as enzyme source.
    
    Returns
    -------
    xp : Array-like
        Array containing the values for [enzyme, substrate, product] for the following
        time(s):
    """
    t = np.asarray(t)
    
    t_0, rate, k = params[0], params[1], params[2]
    
    if function=='sig':
        enzyme_source = sigmoid_der1(t, 0, 1, rate, t_0)
    elif function=='gom':
        enzyme_source = gompertz_der1(t, 0, 1, rate, t_0)
    else:
        enzyme_source = function
    
    try:
        xp = np.zeros([3, len(t)])
        
        xp[0, :] = enzyme_source # e
    
        xp[1, :] = - k * x[0] * x[1] # s
    
        xp[2, :] = 2 * k * x[0] * x[1] # p
        
    except:
        xp = np.zeros(3)
    
        xp[0] = enzyme_source # e
    
        xp[1] = - k * x[0] * x[1] # s
    
        xp[2] = 2 * k * x[0] * x[1] # p
    
    return xp

def simulate(t, t_0, rate, k, func='sig'):
    """
    Simulates the enzyme, substrate and product system with the given parameters. max_time 
    is the maximum time of simulation, t_0 the mid point of enzyme apparition, rate is the rate at which enzyme 
    appears in, k is the cleaving constant of the enzyme, and func is the function to be 
    used for enzyme apparition.
    
    Parameters
    ----------
    t : list, array-like
        time vector to be used for simulation. Special consideration for length
        and dt must be taken.
    t_0 : float
        Parameter for enzyme apparition function representing half apparition.
    rate : float
        Rate parameter for enzyme apparition function.
    k : float
        Cleaving constant for enzyme.
    function : string or function, optional
        'sig' : (Default) sigmoid function for enzyme source.
        'gom' : gompertz function for enzyme source.
        func : function to be used as enzyme source.
    
    Returns
    -------
    C : numpy.array
        Array of the active effective enzyme along the simulation.
    S : numpy.array
        Array of the effective substrate along the simulation.
    P : numpy.array
        Array of the effective product along the simulation.
    """
    simulation = ode(rhs).set_integrator('dopri5')
    
    x0 = [0, 0.5, 0]
    t0 = t[0]
    
    params = [t_0, rate/100, k/100]
    
    simulation.set_initial_value(x0, t0).set_f_params(params, func)
    
    C = [x0[0]]
    S = [x0[1]]
    P = [x0[2]]
    
    for this_t in t[1:]:
        if simulation.successful():
            simulation.integrate(this_t)
            c, s, p = simulation.y[0], simulation.y[1], simulation.y[2]
            C.append(c)
            S.append(s)
            P.append(p)
        else:
            print('simulation aborted')
            break
    
    C = np.asarray(C)
    S = np.asarray(S)
    P = np.asarray(P)
    
    return C, S, P


def fit_caspase(m, time_estimate, function='sig', timepoints=10, Plot=False, fix_rate=None, fix_k=None):
    """
    Fits the monomer fraction curve with a simulation for the enzyme, substrate and product 
    system.
    
    Parameters
    ----------
    m : Array-like
        Monomer fraction curve to be fitted.
    time_estimate : float
        estimation of hal effective active enzyme apparition used to facilitate simulation fit.
    function : string or function, optional
        'sig' : (Default) sigmoid function for enzyme source.
        'gom' : gompertz function for enzyme source.
        func : function to be used as enzyme source.
    timepoints : float, optional
        Timespan of the curve. Defaults to 10.
    Plot : boolean, optional
        If True, the simulation is plotted alongside the data m curve. Default is False.
    fix_rate : float, optional
        Fixes the rate to the specified value. Default is None so the parameter is 
        free to be fitted.
    fix_k : float, optional
        Fixes the cleaving constant of the enzyme to the specified value. 
        Default is None so the parameter is free to be fitted.
    
    Returns
    -------
    Sol : Dictionary
        returns a dictionary with the results of the fit.
        't0' : half active effective enzyme apparition parameter
        'rate' : rate parameter for effective enzyme apparition function
        'k' : cleaving constant for enzyme
        'casp' : numpy.array of active effective caspase simulation
        'sens': numpy.array of effective substrate simulation
        'prod' : numpy.array of effective product simulation
        'chi' : chi square value of fit.
    """
    
    ms = Normalize(m)
    #ds = (1-ms)/2
    #deltams = savgol_filter(ms, 5, 2, deriv = 1, delta = timepoints)
    #deltams = Normalize(deltams)
    max_temp = len(m)*timepoints
    
    time_coarse = np.arange(0, max_temp, timepoints)
    time_fine = np.arange(0, max_temp)
    
    def chi_2(params):
        t_0, rate, k = params[0], params[1], params[2]
        if fix_rate is not None:
            rate = fix_rate
        if fix_k is not None:
            k = fix_k
        
        casp, sens, prod = simulate(time_fine, t_0, rate, k, func=function)
        
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
    casp, sens, prod = simulate(time_fine, t_0, rate, k)
    
    chi = chi_2([t_0, rate, k])
    #option = {'maxiter' : 5}
    Other_Initial_parameters = [10, 50]
    for Initial_parameters[1], Initial_parameters[2] in itertools.permutations(Other_Initial_parameters):
        mini = minimize(chi_2, x0=Initial_parameters, bounds=bounds)#, options = option)
        
        this_t_0, this_rate, this_k = mini.x[0], mini.x[1], mini.x[2]
        this_casp, this_sens, this_prod = simulate(time_fine, t_0, rate, k)
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
        """
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
        """
    
    Sol = {'t0':t_0, 'rate':rate, 'k':k, 'casp':casp, 'sens':sens, 'prod':prod, 'chi':chi}
    return Sol