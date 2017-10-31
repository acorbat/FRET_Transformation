# -*- coding: utf-8 -*-
"""
Created on Mon Aug 15 09:36:56 2016

@author: Agus
"""

import operator
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from scipy.optimize import minimize, curve_fit
from scipy.signal import savgol_filter as savgol
from scipy.interpolate import splrep, splev
from scipy.misc import derivative

from fret_transformation import anisotropy_functions as af
from fret_transformation import caspase_fit as cf

# Define region selector Functions

def sigmoid_region(Curve_x0, Curve_rate, Curve, timepoints=10, minimal = 0.01, n_min = 5, n_max = 15):
    """
    Gets the sigmoid curve section and its index according to sigmoid fit parameters.

    Parameters
    ----------
    Curve_x0 : value
        x0 sigmoid fit parameter.
    Curve_rate : value
        rate of sigmoid.
    Curve : list
        list of values from where curve section must be taken.
    timepoints : value
        temporal difference between points in the curve. Default is 10.
    minimal : value
        value at which normalized sigmoid curve is considered constant.
        Smaller values mean larger sections. Default is 0.01.
    n_min : int
        Minimum length of curve section. Default is 5.
    n_max : int
        Maximum length of curve section. Default is 15.

    Returns
    -------
    Curve_Section : list
        Section of the curve that corresponds to the sigmoid region.
    index : int
        index of the first element of the curve section in curve.
    """
    length = len(Curve)
    time = np.arange(0, timepoints*length, timepoints)
    Value = min(time, key=lambda x:abs(x-Curve_x0))
    index = np.where(time==Value)
    start_time = -np.log(1/minimal - 1)/Curve_rate
    end_time = -np.log(1/(1-minimal) - 1)/Curve_rate
    start_timepoints = int(abs(start_time/timepoints))
    end_timepoints = int(abs(end_time/timepoints))
    if start_timepoints < n_min:
        start_timepoints = n_min
    if end_timepoints < n_min:
        end_timepoints = n_min
    if start_timepoints > n_max:
        start_timepoints = n_max
    if end_timepoints > n_max:
        end_timepoints = n_max
    index = index[0] - start_timepoints
    if index[0]<0:
        index[0] = 0
    index = int(index[0])
    Curve_Section = Curve[index:index+start_timepoints+end_timepoints]
    return Curve_Section, index


def post_region(Curve_x0, Curve_rate, Curve, timepoints=10, minimal = 0.01, n_min = 5, n_max = 15):
    """
    Based on sigmoid region, gets the 10 points following the curve section.

    Parameters
    ----------
    Curve_x0 : value
        x0 sigmoid fit parameter.
    Curve_rate : value
        rate of sigmoid.
    Curve : list
        list of values from where curve section must be taken.
    timepoints : value
        temporal difference between points in the curve. Default is 10.
    minimal : value
        value at which normalized sigmoid curve is considered constant.
        Smaller values mean larger sections. Default is 0.01.
    n_min : int
        Minimum length of curve section. Default is 5.
    n_max : int
        Maximum length of curve section. Default is 15.

    Returns
    -------
    Curve_Section : list
        10 first elements following curve section if possible.
    """
    # This function could actually call sigmoid region and decide from there
    length = len(Curve)
    time = np.arange(0, timepoints*length, timepoints)
    Value = min(time, key=lambda x:abs(x-Curve_x0))
    index = np.where(time==Value)
    end_time = -np.log(1/(1-minimal) - 1)/Curve_rate
    end_timepoints = int(abs(end_time/timepoints))
    if end_timepoints < n_min:
        end_timepoints = n_min
    if end_timepoints > n_max:
        end_timepoints = n_max
    index = index[0] + end_timepoints
    index = int(index[0])
    Curve_Section = Curve[index:index+10]
    return Curve_Section

def pre_region(Curve_x0, Curve_rate, Curve, timepoints=10, minimal = 0.01, n_min = 5, n_max = 15):
    """
    Based on sigmoid region, gets the 10 points following the curve section.

    Parameters
    ----------
    Curve_x0 : value
        x0 sigmoid fit parameter.
    Curve_rate : value
        rate of sigmoid.
    Curve : list
        list of values from where curve section must be taken.
    timepoints : value
        temporal difference between points in the curve. Default is 10.
    minimal : value
        value at which normalized sigmoid curve is considered constant.
        Smaller values mean larger sections. Default is 0.01.
    n_min : int
        Minimum length of curve section. Default is 5.
    n_max : int
        Maximum length of curve section. Default is 15.

    Returns
    -------
    Curve_Section : list
        10 first elements preceding curve section if possible.
    """
    length = len(Curve)
    time = np.arange(0, timepoints*length, timepoints)
    Value = min(time, key=lambda x:abs(x-Curve_x0))
    index = np.where(time==Value)
    start_time = -np.log(1/minimal - 1)/Curve_rate
    start_timepoints = round(abs(start_time/timepoints))
    if start_timepoints < n_min:
        start_timepoints = n_min
    if start_timepoints < n_max:
        start_timepoints = n_max
    index = index[0] - start_timepoints
    if index[0]<0:
        index[0] = 0
    index = int(index[0])
    Curve_Section = Curve[:index+1]
    try:
        Curve_Section = Curve_Section[-10:]
    except:
        Curve_Section = Curve_Section

    return Curve_Section

#Define normalizing function
def Normalize(vect):
    """
    Returns a normalization of vect from 0 to 1.
    """
    this_vect = vect[:]
    this_vect -= min(this_vect)
    this_vect /= max(this_vect)
    return this_vect

#%% Define function to check if a curve is a possible apoptosis curve

def is_apoptosis(df, fluo, maximum_cond, minimum_cond, timepoints=10):
    """
    (Deprecated) Checks some parameters to detect curves that are not apoptotic.
    """
    if df.finitepoints.values<=20:
        return False
    if np.isnan(df['r_'+fluo].values[0]).all():
        return False

    start_time = [ind for ind, r in enumerate(df['r_'+fluo].values[0]) if not np.isnan(r)][0]
    start_time *=timepoints

    base, amplitude, rate, x0 = df[fluo+'_base'], df[fluo+'_amplitude'], df[fluo+'_rate'], df[fluo+'_x0']
    start_percent = (cf.sigmoid(start_time, base, amplitude, rate, x0)-base)/(amplitude)

    if start_percent.values>=0.17:
        return False

    conds = ((operator.lt, maximum_cond), (operator.gt, minimum_cond))
    for op, this_dict in conds:
        for name in this_dict:
            if not op(df[name].values, this_dict[name]):
                return False
    else:
        return True

#%% Define function to correct inadequate sigmoid fit

def ReFit(df, index, fluorophore, timepoints=10, Plot=True, p0=None):
    """
    Takes dataframe and re fits with a sigmoid the specified index and fluorophore
    returning the new set of popts.

    Parameters
    ----------
    df : Pandas DataFrame
        Dataframe of the complete experiment.
    index : index
        Index of the row that is to be re-analyzed.
    fluorophore : string
        Fluorophore name of the curve that is to be refitted.
    timepoints : float, optional
        Timespan of the curve. Defaults to 10.
    Plot : boolean, optional
        If True, the old and new fit is plotted alongside the curve. Default is False.
    p0 : list of floats, optional
        Initial paramters for the non linear fit. Defaults to None.

    Returns
    -------
    new_popt : list of floats
        [base, amplitude, rate, x0] of the new sigmoid fit.
    """
    x = np.arange(0, 90*timepoints, timepoints)
    y = df['r_'+fluorophore][index]

    new_popt, _, _, _ = nanfit(cf.sigmoid, y, x, p0=p0)

    if Plot:
        base, amplitude, rate, x0 = df[fluorophore+'_base'][index], df[fluorophore+'_amplitude'][index], df[fluorophore+'_rate'][index], df[fluorophore+'_x0'][index]
        fitted = cf.sigmoid(x, base, amplitude, rate, x0)

        base, amplitude, rate, x0 = new_popt
        new_fitted = cf.sigmoid(x, base, amplitude, rate, x0)

        plt.plot(x, y, 's', label='data')
        plt.plot(x, fitted, '--b', label='previous')
        plt.plot(x, new_fitted, '--r', label='new')
        plt.legend(loc=2)
        plt.show()

    return new_popt

def windowFit(func, y, x=None, timepoints=10, windowsize=30, windowstep=10):
    """
    Performs succesive fits with func in the windowed curve y, with window size windowsize
    and window step windowstep. Returns a list with the popts for each window.

    Take into consideration that windowsize will set a maximum for the rates that can
    be fitted adequately. windowstep should be smaller thatn windowsize in order to avoid
    the possibility that half a sigmoid is in one window and the other half in another.

    Parameters
    ----------
    func : function
        Function used to fit.
    y : Array-like, list
        Curve to be fitted.
    x : Array-like, list, optional
        x curve for y data. Default is None. Makes timepoints unnecessary if given.
    timepoints : float, optional
        Time span of the y curve. Default is 10.
    windowsize : int, optional
        Amount of points to be taken into the window for fitting. Default is 30
    windowstep : int, optional
        Step for the traslation of each window generated.

    Returns
    -------
    popts : list of list of floats
        List containing the popts returned for each windowed fit.
    """
    if x is None:
        x = np.arange(y.shape[0])
        x = x * timepoints

    popts = []
    start_ind = 0
    end_ind = 0 + windowsize
    while end_ind<=y.shape[0]:
        windowed_y = y[start_ind:end_ind]
        windowed_x = x[start_ind:end_ind]

        chi2 = float('+inf')
        if np.sum(np.isfinite(windowed_y)) >= windowsize // 2:
            for rate in (.1, .17, .1, .7, 1):
                try:
                    _popt, _, _, (_ ,res) = nanfit(func, windowed_y, xdata=windowed_x, timepoints=timepoints, p0=[0.2, 0.2, rate, x[start_ind + windowsize // 2]], returnfit=True)
                    _chi2 = np.nansum(res * res)
                    if _chi2 < chi2:
                        popt = np.copy(_popt)
                        chi2 = np.copy(_chi2)
                except RuntimeError:
                    if 'popt' not in locals():
                        popt = [np.nan] * 4

            popts.append((popt))

        else:
            popts.append([np.nan] * 4)

        start_ind += windowstep
        end_ind += windowstep

    return popts


def nanfit(func, ydata, xdata=None, timepoints=10, returnfit=False, p0=None):
    """
    Fits function func to ydata ignoring nans.

    Parameters
    ----------
    func : function
        Function used to fit.
    ydata : Array-like, list
        Curve to be fitted
    xdata : Array-like, list
        x curve for the y curve. Defaults to None. If None, timepoints is used to generate
        an equispaced vector.
    timepoints : float
        Time span of the y curve. Defaults to 10. Ignored if xdata is given.
    returnfit : boolean, optional
        If True, returns fitline and resline
    p0 : list of floats, optional
        Initial parameters to be used in non linear fit.

    Returns
    -------
    popt : Array-like
        Parameters returned from non linear fit.
    pcov : Array-like
        Covariance matrix from fit.
    fitline : tuple of curves, optional
        If returnfit, returns a tuple of xdata and fit result.
    resline : tuple of curves, optional
        If returnfit, returns a tuple of xdata and fit result residues.
    """
    if xdata is None:
        xdata = np.arange(ydata.shape[0])
        xdata = xdata * timepoints

    mask = np.where(np.isfinite(ydata))

    if mask ==  np.array([]):
        popt = None
        pcov = None
    else:
        if p0 is not None:
            popt, pcov = curve_fit(func, xdata[mask], ydata[mask], p0=p0)
        else:
            popt, pcov = curve_fit(func, xdata[mask], ydata[mask])

    if returnfit and popt is not None:
        fitline = (xdata, func(xdata, *popt))
        resline = (xdata, ydata - fitline[1])
    else:
        fitline = None
        resline = None

    return popt, pcov, fitline, resline


#%% Define fitting functions

# Define fitting function for crossed intensities
def Fit_Global(df, Am1, Am2, Ad1, Ad2, fluo, timepoints=10, length=90, minimal = 0.001, n_min=5, n_max=15, Plot = False):
    """
    (Dprecated: see Fit_Global_r)
    """
    # Concatenate curves
    pars = []
    pers = []
    lengths = [0]

    for i in df.index:
        parallel, start_time = sigmoid_region(df[fluo+'_x0'][i], df[fluo+'_rate'][i], df[fluo+'_I_parallel'][i], timepoints, minimal, n_min, n_max)
        perpendicular, start_time = sigmoid_region(df[fluo+'_x0'][i], df[fluo+'_rate'][i], df[fluo+'_I_perpendicular'][i], timepoints, minimal, n_min, n_max)
        pars.append(parallel)
        pers.append(perpendicular)
        lengths.append(len(parallel))
    m_starts = np.cumsum(lengths)

    # Define Chi Square Function to fit
    def chi_2_global(parameters):
        alfa, beta, b, ft = parameters[0], parameters[1], parameters[2], parameters[3]
        Am = Am1 * alfa + Am2 * (1-alfa)
        Ad = Ad1 * beta + Ad2 * (1-beta)
        ms = parameters[4:]

        out = 0.
        for n, (par, per, m_start) in enumerate(zip(pars, pers, m_starts)):
            m = ms[m_start:m_start + len(par)]
            out += sum((par - af.Int_par(m, Am, Ad, b, ft)) ** 2)
            out += sum((per - af.Int_per(m, Am, Ad, b, ft)) ** 2)

        return out

    # Define Chi Square Function to fit
    def chi_2_global_real(parameters):
        alfa, beta, b, ft = parameters[0], parameters[1], parameters[2], parameters[3]
        Am = Am1 * alfa + Am2 * (1-alfa)
        Ad = Ad1 * beta + Ad2 * (1-beta)
        ms = parameters[4:]

        out = 0.
        for n, (par, per, m_start) in enumerate(zip(pars, pers, m_starts)):
            m = ms[m_start:m_start + len(par)]
            out += sum(((par - af.Int_par(m, Am, Ad, b, ft)) ** 2)/af.Int_par(m, Am, Ad, b, ft))
            out += sum(((per - af.Int_per(m, Am, Ad, b, ft)) ** 2)/af.Int_per(m, Am, Ad, b, ft))

        return out

    # Define minimization initial parameters and bounds
    # Initial parameters
    Initial_parameters = np.zeros(4+m_starts[-1])
    Initial_parameters[0] = 0.5 # Alfa
    Initial_parameters[1] = 0.5 # Beta
    Initial_parameters[2] = 0.5 # b
    Initial_parameters[3] = 0 # ft


    for n, e in enumerate(m_starts):
        ind = 4+e-1
        Initial_parameters[ind] = 1

    # Bounds
    bounds = [(0, 1)] * len(Initial_parameters)
    bounds[3] = (0, np.inf)

    mini = minimize(chi_2_global, x0=Initial_parameters, bounds=bounds)
    mini = minimize(chi_2_global_real, x0=mini.x, bounds=bounds)

    alfa, beta, b, ft = mini.x[0], mini.x[1], mini.x[2], mini.x[3]
    Am = Am1 * alfa + Am2 * (1-alfa)
    Ad = Ad1 * beta + Ad2 * (1-beta)
    ms = mini.x[4:]

    if Plot:
        # Plot Fit

        print(b)

        for n, (par, per, m_start) in enumerate(zip(pars, pers, m_starts)):
            m_fit = ms[m_start:m_start + len(par)]
            # Plot Crossed Intensities
            plt.plot(par, 'sb')
            plt.plot(per, 'sr')
            plt.plot(af.Int_par(m_fit, Am, Ad, b, ft), 'b')
            plt.plot(af.Int_per(m_fit, Am, Ad, b, ft), 'r')
            plt.xlabel('tiempo (minutos)')
            plt.ylabel('Intensidad (u.a.)')
            plt.show()

            # Plot proportion
            plt.plot(m_fit, 'b')
            plt.xlabel('tiempo (minutos)')
            plt.ylabel('Proporción de Monómero')
            plt.show()

            # Plot Anisotropy
            plt.plot([0, len(m_fit)], [Am, Am], 'r--')
            plt.plot([0, len(m_fit)], [Ad, Ad], 'r--')
            plt.plot(af.Anisotropy_FromFit(m_fit, Am, Ad, b), 'r')
            plt.plot(af.Anisotropy_FromInt(par, per), 'sr')
            #plt.plot(Anisotropy)
            plt.xlabel('tiempo (minutos)')
            plt.ylabel('Anisotropía')
            plt.show()

    ms = np.concatenate((np.zeros(start_time), ms, np.ones(length-len(ms)-start_time)))
    if mini.success:
        Sol = {'Am': Am, 'Ad': Ad, 'b': b, 'f': ft, 'm': ms}
    else:
        Sol = {'Am': np.nan, 'Ad': np.nan, 'b': np.nan, 'f': [np.nan]*len(ms), 'm': [np.nan]*len(ms)}

    return Sol


# Define fitting function for normalized crossed intensities using experimental b

def Fit_Global_r(df, Am1, Am2, Ad1, Ad2, b_min, b_max, fluo, timepoints=10, minimal = 0.001, n_min=5, n_max=15, Plot = False):
    """
    Optimizes the transformation of anisotropy curves to monomer fraction curves,
    fitting for the best monomer and dimer anisotropies and brightness relation.

    Parameters
    ----------
    df : Pandas DataFrame
        DataFrame of the whole experiment with the anisotropy curves of each fluorophore.
    Am1 : float
        Minimum possible value for monomer anisotropy.
    Am2 : float
        Maximum possible value for monomer anisotropy.
    Ad1 : float
        Minimum possible value for dimer anisotropy.
    Ad2 : float
        Maximum possible value for dimer anisotropy.
    b_min : float
        Minimum possible value for brightness relation.
    b_max : float
        Maximum possible value for brightness relation.
    fluo : string
        fluorophore to optimize.
    timepoints : float, optional
        Time span of curves. Defaults to 10.
    minimal : float, optional
        Parameter passed to sigmoid region function and represents the value where
        sigmoid is considered constant. Default is 0.001.
    n_min : int, optional
        Parameter passed to sigmoid region to select minimum amount of points for a
        sigmoid region. Default is 5.
    n_max : int, optional
        Parameter passed to sigmoid region to select maximum amount of points for a
        sigmoid region. Default is 15.
    Plot : boolean, optional
        If True, plots graphs showing results of the transformation.

    Returns
    -------
    Sol : Dictionary
        Dictionary containing {'Am': monomer anisotropy, 'Ad': dimer anisotropy,
        'b': brightness relation, 'm': monomer fraction curve} for the row passed
        in DataFrame or makes a fit for every row in DataFrame (didn't work).
    """
    # Concatenate curves
    pars = []
    pers = []
    lengths = [0]

    for i in df.index:
        parallel, start_time = sigmoid_region(df[fluo+'_x0'][i], df[fluo+'_rate'][i], df[fluo+'_I_parallel_n'][i], timepoints=timepoints, minimal=minimal, n_min=n_min, n_max=n_max)
        perpendicular, start_time = sigmoid_region(df[fluo+'_x0'][i], df[fluo+'_rate'][i], df[fluo+'_I_perpendicular_n'][i], timepoints=timepoints, minimal=minimal, n_min=n_min, n_max=n_max)
        pars.append(parallel)
        pers.append(perpendicular)
        lengths.append(len(parallel))
    m_starts = np.cumsum(lengths)

    # Define Chi Square Function to fit
    def chi_2_global(parameters):
        alfa, beta, b = parameters[0], parameters[1], parameters[2]
        Am = Am1 * alfa + Am2 * (1-alfa)
        Ad = Ad1 * beta + Ad2 * (1-beta)
        ms = parameters[3:]

        out = 0.
        for n, (par, per, m_start) in enumerate(zip(pars, pers, m_starts)):
            m = Normalize(ms[m_start:m_start + len(par)])
            out += sum((par - af.Int_par_r(m, Am, Ad, b)) ** 2)
            out += sum((per - af.Int_per_r(m, Am, Ad, b)) ** 2)

        return out

    # Define Real Chi Square Function to fit
    def chi_2_global_real(parameters):
        alfa, beta, b = parameters[0], parameters[1], parameters[2]
        Am = Am1 * alfa + Am2 * (1-alfa)
        Ad = Ad1 * beta + Ad2 * (1-beta)
        ms = parameters[3:]

        out = 0.
        for n, (par, per, m_start) in enumerate(zip(pars, pers, m_starts)):
            m = Normalize(ms[m_start:m_start + len(par)])
            out += sum(((par - af.Int_par_r(m, Am, Ad, b)) ** 2)/af.Int_par_r(m, Am, Ad, b))
            out += sum(((per - af.Int_per_r(m, Am, Ad, b)) ** 2)/af.Int_per_r(m, Am, Ad, b))

        return out

    # Define minimization initial parameters and bounds
    # Initial parameters
    Initial_parameters = np.zeros(3+m_starts[-1])
    Initial_parameters[0] = 0.5 # Alfa
    Initial_parameters[1] = 0.5 # Beta
    Initial_parameters[2] = 0.5 # b


    for n, e in enumerate(m_starts):
        ind = 3+e-1
        Initial_parameters[ind] = 1

    # Bounds
    bounds = [(0, 1)] * len(Initial_parameters)
    bounds[2] = (b_min, b_max)

    mini = minimize(chi_2_global, x0=Initial_parameters, bounds=bounds)
    mini = minimize(chi_2_global_real, x0=mini.x, bounds=bounds)

    alfa, beta, b = mini.x[0], mini.x[1], mini.x[2]
    Am = Am1 * alfa + Am2 * (1-alfa)
    Ad = Ad1 * beta + Ad2 * (1-beta)
    ms = mini.x[3:]

    if Plot:

        print(b)

        for n, (par, per, m_start) in enumerate(zip(pars, pers, m_starts)):
            m_fit = ms[m_start:m_start + len(par)]
            # Plot Crossed Intensities
            plt.plot(par, 'sb')
            plt.plot(per, 'sr')
            plt.plot(af.Int_par_r(m_fit, Am, Ad, b), 'b')
            plt.plot(af.Int_per_r(m_fit, Am, Ad, b), 'r')
            plt.xlabel('tiempo (minutos)')
            plt.ylabel('Intensidad (u.a.)')
            plt.show()

            # Plot proportion
            plt.plot(m_fit, 'b')
            plt.xlabel('tiempo (minutos)')
            plt.ylabel('Proporción de Monómero')
            plt.show()

            # Plot Anisotropy
            plt.plot([0, len(m_fit)], [Am, Am], 'r--')
            plt.plot([0, len(m_fit)], [Ad, Ad], 'r--')
            plt.plot(af.Anisotropy_FromFit(m_fit, Am, Ad, b), 'r')
            plt.plot(af.Anisotropy_FromInt(par, per), 'sr')
            #plt.plot(Anisotropy)
            plt.xlabel('tiempo (minutos)')
            plt.ylabel('Anisotropía')
            plt.show()

    length = len(parallel)
    ms = np.concatenate((np.zeros(start_time), ms, np.ones(length-len(ms)-start_time)))
    Sol = {'Am': Am, 'Ad': Ad, 'b': b, 'm': ms}

    return Sol

#%% caspase from anisotropy derivation


def replace_nan(curve):
    """Replaces values in curve with a linear interpolation and discards nans
    at the beginning and end of the list."""
    # First we interpolate linearly in the middle points
    curve = pd.Series(curve)
    curve = curve.interpolate(method="linear")

    # Then we find nearest finite number to fill beggining and end
    curve = curve.values
    ind = np.where(~np.isnan(curve))[0]
    first, last = ind[0], ind[-1]
    curve[:first] = curve[first]
    curve[last + 1:] = curve[last]
    return curve


def get_max_ind(compl):
    """Finds index of max without considering the beginning and ending of
    vector"""
    compl = compl.copy()
    ind_max = np.where(compl==np.max(compl))[0]
    ini = 0
    while (any(ind_max==0) or any((ind_max+1)==len(compl))) and len(compl)>1:
        if any(ind_max==0):
            ini +=1
            compl = np.delete(compl, 0)
        elif any((ind_max+1)==len(compl)):
            compl = np.delete(compl, -1)
        ind_max = np.where(compl==np.max(compl))[0]

    ind_max = np.where(compl==np.max(compl))[0][0]
    return ind_max + ini


def find_complex(df, col_to_der='r_from_i', order=5, timepoints=10, Plot=False):
    """
    Takes the whole dataframe and applies finite differences to data in order
    to find the derivative of the sigmoid region anisotropy data of filtered
    curves.

    This function first selects the sigmoid region of the data, does a linear
    interpolation over missing data, replacing with nearest at end and
    beginning of curves. Finite differences is used to find the first
    derivative, and filter noise. Spline interpolation is used to
    find maximum at the derived curve. Maximum cannot be found at the
    beginning and ending of curves (this is usually caused by interpolation) so
    these values are discarded.

    Parameters
    ----------
    df : Pandas DataFrame
        DataFrame containing the curves for each fluorophores, the best fit
        values, which are used to filter the curves as well.
    col_to_der : string, optional
        suffix of column to derive. Default is "r_from_i".
    order : optional, odd int
        Number of points to be used in the finite differences. Must be odd.
        Default is 5.
    fluorophores : list of strings
    Plot : boolean, optional
        True if plots are to be showed. Default is False.

    Returns
    -------
    df : Pandas DataFrame
        Updated Pandas DataFrame
    """
    fluorophores = [col for col in df.columns if col_to_der in col]
    fluorophores = [col.split('_')[0] for col in fluorophores]
    
    ders = {}
    maxs = {}
    for fluo in fluorophores:
        ders[fluo] = []
        maxs[fluo] = []

    for i in df.index:
        for fluo in fluorophores:
            if np.isfinite(df[fluo+'_rate'][i]):

                r = df['_'.join([fluo, col_to_der])][i]
                r = replace_nan(r)
                time = np.arange(0, len(r)*timepoints, timepoints)

                if all(np.isfinite(r)) and len(r)>1:
                    plot = True

                    def this_vect(t):
                        ind = t//timepoints
                        ind = np.clip(ind, 0, len(r)-1)
                        return r[ind]

                    r_der = derivative(this_vect, time, dx=timepoints, order=order)

                    t = np.arange(0, len(r)*timepoints)
                    f = splrep(time, r_der, k=3, s=0)
                    der_interp = splev(t, f, der=0)

                    x0 = df[fluo+'_x0'][i]
                    rate = df[fluo+'_rate'][i]
                    r_reg, ind = sigmoid_region(x0, rate, r, minimal=0.01, timepoints=timepoints)
                    this_time = time[ind:ind+len(r_reg)]

                    max_act = get_max_ind(der_interp[ind*timepoints:(ind+len(r_reg))*timepoints]) + ind*timepoints

                    if Plot:
                        fig, axs = plt.subplots(2,1, sharex=True, figsize=(10,12))
                        axs[0].plot(this_time, r_reg, 'o'+Colors[fluo])
                        axs[0].plot(time, r, Colors[fluo]+'--', alpha=0.5)
                        axs[0].set_ylabel('fraction')

                        axs[1].plot(time, r_der)
                        axs[1].plot(t, der_interp, Colors[fluo])
                        axs[1].set_ylabel('complex')
                        axs[1].set_xlabel('time (min.)')

                        plt.suptitle('obj:'+str(i)+' exp:'+df.Content_YFP[i]+' max:'+str(max_act))
                        last_maxs = [fluo+' '+str(maxs[fluo][-1]) for fluo in fluorophores]
                        note = '\n'.join(last_maxs)
                        plt.show()
                        print(note)

                    ders[fluo].append(r_der)
                    maxs[fluo].append(max_act)

                else:
                    ders[fluo].append([np.nan])
                    maxs[fluo].append(np.nan)

            else:
                ders[fluo].append([np.nan])
                maxs[fluo].append(np.nan)

    for fluo in fluorophores:
        df[fluo+'_r_complex'] = ders[fluo]
        df[fluo+'_max_activity'] = maxs[fluo]

    return df
