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

from scipy.signal import savgol_filter as savgol
from scipy.interpolate import splrep, splev
from scipy.misc import derivative

# import not registered modules
#this_dir = pathlib.Path(r'C:\Users\Agus\Documents\Laboratorio\Imaging three sensors\Analysis')
this_dir = pathlib.Path(r'D:\Agus\Imaging three sensors\Deriving')
#os.chdir(r'C:\Users\Agus\Documents\Laboratorio\Imaging three sensors\FRET_Transformation')
os.chdir(r'D:\Agus\Imaging three sensors\FRET_Transformation')
import transformation as tf
import caspase_fit as cf
os.chdir(str(this_dir))

#%% load data

#df = pd.read_pickle('2017-01-17_OneCasp_mCit Cas8 mCit_b_100.pandas')
fluorophores = ['YFP', 'mKate', 'TFP']
timepoints = 10
Colors = {'YFP':'y', 'mKate':'r', 'TFP':'g'}

#%% Define function to find maximum complex peaks

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
                    
                    axs[1].plot(this_time, r_der)
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
                r_reg = replace_nan(r_reg)
                this_time = time[ind:ind+len(r_reg)]
                
                if all(np.isfinite(r_reg)) and len(r_reg)>1:
                    plot = True
                    
                    def this_vect(t):
                        ind = t//timepoints
                        ind = np.clip(ind, 0, len(r_reg)-1)
                        return r_reg[ind]
                    
                    this_inds = np.arange(0, len(r_reg)*timepoints, timepoints)
                    r_der = derivative(this_vect, this_inds, dx=timepoints, order=order)
                    
                    t = np.arange(ind*timepoints, (ind+len(r_reg))*timepoints)
                    f = splrep(this_time, r_der, k=3, s=0)
                    der_interp = splev(t, f, der=0)
                    
                    max_act = get_max_ind(der_interp)+ind*timepoints
                    
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


for this_order in [3, 5]:
    df = pd.read_pickle('2017-09-15_OneCaspFiltered.pandas')
    
    filename = 'OneCasp_derivations_order%02d.pdf' % (this_order)
    pandasname = 'OneCasp_derivations_order%02d.pandas' % (this_order)
    pdf_path = res_path.joinpath(filename)
    pandas_path = res_path.joinpath(pandasname)
    with PdfPages(str(pdf_path)) as pp:
        df = find_complex(df, pp, order=this_order)
        df.to_pickle(str(pandas_path))