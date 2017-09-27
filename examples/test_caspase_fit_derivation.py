# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 14:37:08 2017

@author: Agus
"""

# import packages to be used
import os
import pathlib
import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from scipy.optimize import curve_fit
from scipy.signal import savgol_filter as savgol
from scipy.interpolate import splrep, splev
from scipy.misc import derivative
from scipy.stats import ttest_rel

# import not registered modules
#this_dir = pathlib.Path(r'C:\Users\Agus\Documents\Laboratorio\Imaging three sensors\Analysis')
this_dir = pathlib.Path(r'C:\Users\Agus\Documents\Laboratorio\Imaging three sensors\Deriving')
#this_dir = pathlib.Path(r'D:\Agus\Imaging three sensors\Deriving')
os.chdir(r'C:\Users\Agus\Documents\Laboratorio\Imaging three sensors\FRET_Transformation')
#os.chdir(r'D:\Agus\Imaging three sensors\FRET_Transformation')
import anisotropy_functions as af
import transformation as tf
import caspase_fit as cf
os.chdir(str(this_dir))

#%% load data

#df = pd.read_pickle('2017-01-17_OneCasp_mCit Cas8 mCit_b_100.pandas')
fluorophores = ['YFP', 'mKate', 'TFP']
timepoints = 10
Colors = {'YFP':'y', 'mKate':'r', 'TFP':'g'}

#%% Define function to find maximum complex peaks


def myderivative(vec, dx=1.0, n=1, order=3):
    """
    Find the n-th derivative of a function at a point.
    Given a function, use a central difference formula with spacing `dx` to
    compute the `n`-th derivative at `x0`.
    Parameters
    ----------
    vec : function
        Input function.
    x0 : float
        The point at which `n`-th derivative is found.
    dx : float, optional
        Spacing.
    n : int, optional
        Order of the derivative. Default is 1.
    order : int, optional
        Number of points to use, must be odd.
    Notes
    -----
    Decreasing the step size too small can result in round-off error.
    Examples
    --------
    >>> from scipy.misc import derivative
    >>> def f(x):
    ...     return x**3 + x**2
    >>> derivative(f, 1.0, dx=1e-6)
    4.9999999999217337
    """
    if order < n + 1:
        raise ValueError("'order' (the number of points used to compute the derivative), "
                         "must be at least the derivative order 'n' + 1.")
    if order % 2 == 0:
        raise ValueError("'order' (the number of points used to compute the derivative) "
                         "must be odd.")
    # pre-computed for n=1 and 2 and low-order for speed.
    array = np.array
    if n == 1:
        if order == 3:
            weights = array([-1,0,1])/2.0
        elif order == 5:
            weights = array([1,-8,0,8,-1])/12.0
        elif order == 7:
            weights = array([-1,9,-45,0,45,-9,1])/60.0
        elif order == 9:
            weights = array([3,-32,168,-672,0,672,-168,32,-3])/840.0
        else:
            raise ValueError
    elif n == 2:
        if order == 3:
            weights = array([1,-2.0,1])
        elif order == 5:
            weights = array([-1,16,-30,16,-1])/12.0
        elif order == 7:
            weights = array([2,-27,270,-490,270,-27,2])/180.0
        elif order == 9:
            weights = array([-9,128,-1008,8064,-14350,8064,-1008,128,-9])/5040.0
        else:
            raise ValueError
    else:
        raise ValueError
    
    ho = order >> 1
    out = np.nan * vec
    for ndx in range(ho, len(vec)-2*ho):
        out[ndx] = np.sum(weights * vec[(ndx-ho):(ndx+ho+1)])
    return out / dx



def replace_nan(curve):
    """Replaces values in curve with a linear interpolation and discards nans 
    at the beginning and end of the list."""
    curve = pd.Series(curve)
    curve = curve.interpolate(method="linear")
    
    curve = curve.values
    curve = curve[np.isfinite(curve)]
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


def find_complex(df, pp, window=13, poly=6):
    """
    Takes the whole dataframe and applies a Savitzky-Golay filter to smooth 
    data and find the derivative of the sigmoid region anisotropy data 
    of filtered curves.
    
    This function first selects the sigmoid region of the data, does a linear 
    interpolation over missing data, replacing with nearest at end and 
    beginning of curves. Savitzky Golay filter is used to smooth the data 
    curve and then find the first derivative. Spline interpolation is used to 
    find maximum at the derivative curve. Maximum cannot be found at the
    beginning and ending of curves (this is usually caused by interpolation) so
    these values are discarded.
    
    Parameters
    ----------
    df : Pandas DataFrame
        DataFrame containing the curves for each fluorophores, the best fit 
        values, which are used to filter the curves as well.
    pp : matplotlib backend PdfPages Object
        Used to save the resulting figures with the maximum values found.
    window : optional, odd int
        data points to be used for Savitzky Golay filter. Default is 13
    poly : optional, int smaller than window
        polinomial order to be used in Savitsky Golay filter. Has to be smaller
        than window. Default is 6
    
    Returns
    -------
    df : Pandas DataFrame
        Updated Pandas DataFrame
    Also updates (not generates) the pdf with the results.
    """
    time = np.arange(0, 50*timepoints, timepoints)
    fils = {}
    ders = {}
    maxs = {}
    for fluo in fluorophores:
        fils[fluo] = []
        ders[fluo] = []
        maxs[fluo] = []
        
    for i in df.index:
        fig, axs = plt.subplots(2,1, sharex=True, figsize=(10,12))
        plot = False
        for fluo in fluorophores:
            if np.isfinite(df[fluo+'_rate'][i]):
                
                r = df['r_'+fluo][i]
                x0 = df[fluo+'_x0'][i]
                rate = df[fluo+'_rate'][i]
                
                r_reg, ind = tf.sigmoid_region(x0, rate, r, minimal=0.00001)
                r_reg = replace_nan(r_reg)
                this_time = time[ind:ind+len(r_reg)]
                
                if all(np.isfinite(r_reg)) and len(r_reg)>window:
                    plot = True
                    r_fil = savgol(r_reg, window, poly)
                    r_der = savgol(r_reg, window, poly, deriv=1, delta=timepoints)
                    
                    t = np.arange(ind*timepoints, (ind+len(r_reg))*timepoints)
                    f = splrep(this_time, r_der, k=3, s=0)
                    der_interp = splev(t, f, der=0)
                    
                    max_act = get_max_ind(der_interp)+ind*timepoints
                    
                    axs[0].plot(this_time, r_reg, 'o'+Colors[fluo])
                    axs[0].plot(this_time, r_fil, Colors[fluo])
                    axs[0].set_ylabel('fraction')
                    
                    axs[1].plot(this_time, r_der, Colors[fluo])
                    axs[1].plot(t, der_interp)
                    axs[1].set_ylabel('complex')
                    axs[1].set_xlabel('time (min.)')
                    
                    fils[fluo].append(r_fil)
                    ders[fluo].append(r_der)
                    maxs[fluo].append(max_act)
                    
                else:
                    fils[fluo].append([np.nan])
                    ders[fluo].append([np.nan])
                    maxs[fluo].append(np.nan)
                    
            else:
                fils[fluo].append([np.nan])
                ders[fluo].append([np.nan])
                maxs[fluo].append(np.nan)
        if plot:
            plt.suptitle('obj:'+str(i)+' exp:'+df.Content_YFP[i]+' max:'+str(max_act))
            last_maxs = [fluo+' '+str(maxs[fluo][-1]) for fluo in fluorophores]
            note = '\n'.join(last_maxs)
            pp.attach_note(note, positionRect=[100,100,100,100])
            pp.savefig()
        plt.close()
        
    for fluo in fluorophores:
        df[fluo+'_r_filtered'] = fils[fluo]
        df[fluo+'_r_complex'] = ders[fluo]
        df[fluo+'_max_activity'] = maxs[fluo]
        
    return df

windows = [3, 9, 13]
polys = [3, 4, 6]
res_path = pathlib.Path(r'D:\Agus\Imaging three sensors\Deriving\results')
#res_path = pathlib.Path(r'C:\Users\Agus\Documents\Laboratorio\Imaging three sensors\Deriving\results')

for win, pol in itertools.product(windows, polys):
    if win>pol:
        df = pd.read_pickle('2017-09-15_OneCaspFiltered.pandas')
        
        filename = 'OneCasp_derivations_win%02d_pol%02d.pdf' % (win, pol)
        pandasname = 'OneCasp_derivations_win%02d_pol%02d.pandas' % (win, pol)
        pdf_path = res_path.joinpath(filename)
        pandas_path = res_path.joinpath(pandasname)
        with PdfPages(str(pdf_path)) as pp:
            df = find_complex(df, pp, window=win, poly=pol)
            df.to_pickle(str(pandas_path))


#%% Try finite differences for derivation

def find_complex(df, pp, order=5):
    """
    Takes the whole dataframe and applies finite differences to data in order 
    to find the derivative of the sigmoid region anisotropy data of filtered 
    curves.
    
    This function first selects the sigmoid region of the data, does a linear 
    interpolation over missing data, replacing with nearest at end and 
    beginning of curves. Finite differencesis is used to find data the first 
    derivative, and filter noise. Spline interpolation is used to 
    find maximum at the derived curve. Maximum cannot be found at the
    beginning and ending of curves (this is usually caused by interpolation) so
    these values are discarded.
    
    Parameters
    ----------
    df : Pandas DataFrame
        DataFrame containing the curves for each fluorophores, the best fit 
        values, which are used to filter the curves as well.
    pp : matplotlib backend PdfPages Object
        Used to save the resulting figures with the maximum values found.
    order : optional, odd int
        Number of points to be used in the finite differences. Must be odd.
        Default is 5.
    
    Returns
    -------
    df : Pandas DataFrame
        Updated Pandas DataFrame
    Also updates (not generates) the pdf with the results.
    """
    time = np.arange(0, 50*timepoints, timepoints)
    ders = {}
    maxs = {}
    for fluo in fluorophores:
        ders[fluo] = []
        maxs[fluo] = []
        
    for i in df.index:
        fig, axs = plt.subplots(2,1, sharex=True, figsize=(10,12))
        plot = False
        for fluo in fluorophores:
            if np.isfinite(df[fluo+'_rate'][i]):
                
                r = df['r_'+fluo][i]
                x0 = df[fluo+'_x0'][i]
                rate = df[fluo+'_rate'][i]
                
                r_reg, ind = tf.sigmoid_region(x0, rate, r, minimal=0.00001)
                r_reg = r
                ind = 0
                r_reg = replace_nan(r_reg)
                this_time = time[ind:ind+len(r_reg)]
                
                if all(np.isfinite(r_reg)) and len(r_reg)>1:
                    plot = True
                    
                    def this_vect(t):
                        ind = t//timepoints
                        ind = np.clip(ind, 0, len(r_reg)-1)
                        return r_reg[ind]
                    
                    this_inds = np.arange(0, len(r_reg)*timepoints, timepoints)
                    #r_der = derivative(this_vect, this_inds, dx=timepoints, order=order)
                    r_der = myderivative(r_reg, timepoints, n=1, order=order)
                    print(r_der)
                    t = np.arange(ind*timepoints, (ind+len(r_reg))*timepoints)
                    f = splrep(this_time, r_der, k=3, s=0)
                    der_interp = splev(t, f, der=0)
                    
                    try:
                        max_act = get_max_ind(der_interp)+ind*timepoints
                    except Exception as e:
                        print(e)
                        max_act = 0
                    
                    axs[0].plot(this_time, r_reg, 'o'+Colors[fluo])
                    axs[0].set_ylabel('fraction')
                    
                    axs[1].plot(this_time, r_der)
                    axs[1].plot(t, der_interp)
                    axs[1].set_ylabel('complex')
                    axs[1].set_xlabel('time (min.)')
                    
                    ders[fluo].append(r_der)
                    maxs[fluo].append(max_act)
                    
                else:
                    ders[fluo].append([np.nan])
                    maxs[fluo].append(np.nan)
                    
            else:
                ders[fluo].append([np.nan])
                maxs[fluo].append(np.nan)
        if plot:
            plt.suptitle('obj:'+str(i)+' exp:'+df.Content_YFP[i]+' max:'+str(max_act))
            last_maxs = [fluo+' '+str(maxs[fluo][-1]) for fluo in fluorophores]
            note = '\n'.join(last_maxs)
            pp.attach_note(note, positionRect=[100,100,100,100])
            pp.savefig()
        plt.close()
        
    for fluo in fluorophores:
        df[fluo+'_r_complex'] = ders[fluo]
        df[fluo+'_max_activity'] = maxs[fluo]
        
    return df

#%%
for this_order in [3, 5]:
    df = pd.read_pickle('2017-09-15_OneCaspFiltered.pandas')
    
    filename = 'OneCasp_derivations_order%02d.pdf' % (this_order)
    pandasname = 'OneCasp_derivations_order%02d.pandas' % (this_order)
    pdf_path = res_path.joinpath(filename)
    pandas_path = res_path.joinpath(pandasname)
    with PdfPages(str(pdf_path)) as pp:
        df = find_complex(df, pp, order=this_order)
        df.to_pickle(str(pandas_path))


#%% Methods to evaluate time differences

def add_differences(df):
    Differences_tags = ['YFP_to_TFP', 'YFP_to_mKate', 'TFP_to_mKate']
    
    Differences = {}
    Differences['YFP_to_TFP'] = df.TFP_max_activity.values-df.YFP_max_activity.values
    Differences['YFP_to_mKate'] = df.mKate_max_activity.values-df.YFP_max_activity.values
    Differences['TFP_to_mKate'] = df.mKate_max_activity.values-df.TFP_max_activity.values
    
    for Differences_tag in Differences_tags:
        df[Differences_tag] = Differences[Differences_tag]
    
    return df


def nanhist(data, title='histogram', bins=10):
    plt.hist(data[np.isfinite(data)], bins=bins)
    plt.xlabel('time difference (mins.)')
    plt.ylabel('frequency')
    plt.title(title)


this_order = 5
pandasname = 'OneCasp_derivations_order%02d.pandas' % (this_order)
pandas_path = res_path.joinpath(pandasname)
df = pd.read_pickle(str(pandas_path))

df = add_differences(df)

#%% Plot polar histogram (not useful as values should be close to zero, but worth taking a look at)

p = df.YFP_to_TFP.values + 1j* df.YFP_to_mKate.values
p = p[np.isfinite(p)]
pn = p / np.abs(p)
for i, this_pn in enumerate(pn):
    if np.isnan(this_pn):
        pn[i] = 0

frecs = np.histogram(np.angle(pn), bins = 20)

N = len(frecs[0])

theta = frecs[1][:-1] # np.linspace(0.0, 2 * np.pi, N, endpoint=False)
radii = frecs[0]
width = (2*np.pi) / N

#matplotlib.rcParams.update({'font.size': 18})

plt.figure(figsize=(10,10))
ax = plt.subplot(111, polar=True)
bars = ax.bar(theta, radii, width=width)

# Use custom colors and opacity
for r, bar in zip(radii, bars):
    bar.set_facecolor(plt.cm.jet(r / 10.))
    bar.set_alpha(0.8)

plt.show()

plt.polar(p, ls='', marker='o')
plt.show()

#%% Scatter plots of temporality, pearson should give an idea of how well determined time is

for fluo1, fluo2 in itertools.combinations(fluorophores, 2):
    x = df[fluo1+'_max_activity'].values
    y = df[fluo2+'_max_activity'].values
    mask = [True if np.isfinite(this_x) and np.isfinite(this_y) else False for this_x, this_y in zip(x,y)]
    x = x[mask]
    y = y[mask]
    plt.scatter(x, y)
    plt.title(str(np.corrcoef(x, y)))
    plt.show()


#%% 

Differences_tags = ['YFP_to_TFP', 'YFP_to_mKate', 'TFP_to_mKate']

def aprox_percent(Curve, p=0.1):
    return np.where(Curve>=p)[0][0]


def aprox_r_percent(time, r, time_estimate, p=0.1, Plot=False):
    popt, _ = curve_fit(cf.sigmoid, time, r, p0=[np.nanmin(r), np.nanmax(r), 0.5, time_estimate])
    this_curve = cf.sigmoid(np.arange(time[0], time[-1]), *popt)
    this_curve = cf.Normalize(this_curve)
    
    if Plot:
        plt.plot(time, r, 'or')
        plt.plot(time, cf.sigmoid(time, *popt), 'b')
        plt.show()
        print(popt)
    return aprox_percent(this_curve, p=p)


def v_(desc):
    return df[desc].values[np.isfinite(df[desc].values)]


def find_r_percent(df, p=0.5):
    time = np.arange(0, 50*timepoints, timepoints)
    maxs = {}
    for fluo in fluorophores:
        maxs[fluo] = []
        
    for i in df.index:
        for fluo in fluorophores:
            if np.isfinite(df[fluo+'_rate'][i]):
                
                r = df['r_'+fluo][i]
                x0 = df[fluo+'_x0'][i]
                rate = df[fluo+'_rate'][i]
                
                r_reg, ind = tf.sigmoid_region(x0, rate, r, minimal=0.00001)
                r_reg = replace_nan(r_reg)
                this_time = time[ind:ind+len(r_reg)]
                
                if all(np.isfinite(r_reg)) and len(r_reg)>1:
                    max_act = aprox_r_percent(this_time, r_reg, x0, p=p, Plot=True)
                    max_act += ind*timepoints
                    maxs[fluo].append(max_act)
                    
                else:
                    maxs[fluo].append(np.nan)
                    
            else:
                maxs[fluo].append(np.nan)
    
    for fluo in fluorophores:
        df[fluo+'_r_'+str(int(p*100))] = maxs[fluo]
        
    return df

def add_differences_r(df):
    Differences_tags = ['r_YFP_to_TFP', 'r_YFP_to_mKate', 'r_TFP_to_mKate']
    
    Differences = {}
    Differences['r_YFP_to_TFP'] = df.TFP_r_50.values-df.YFP_r_50.values
    Differences['r_YFP_to_mKate'] = df.mKate_r_50.values-df.YFP_r_50.values
    Differences['r_TFP_to_mKate'] = df.mKate_r_50.values-df.TFP_r_50.values
    
    for Differences_tag in Differences_tags:
        df[Differences_tag] = Differences[Differences_tag]
    
    return df

for Differences_tag in Differences_tags:
    r_Differences_tag = 'r_'+Differences_tag
    nanhist(df[Differences_tag].values, bins=20)
    nanhist(df[r_Differences_tag].values, bins=20)
    plt.show()
    statistic, p_value = ttest_rel(np.abs(v_(r_Differences_tag)), np.abs(v_(Differences_tag)))
    print('p-value of '+Differences_tag+' is '+str(p_value))
    if p_value>0.1:
        print('Cannot say that means are different')
    else:
        print('means are different with at least 90% confidence')
        this_mean = np.mean(np.abs(v_(r_Differences_tag))- np.abs(v_(Differences_tag)))
        if this_mean>0:
            print('Caspase fitting method is better for '+str(this_mean)+' minutes')
        else:
            print('Anisotropy fitting method is better for '+str(-this_mean)+' minutes')