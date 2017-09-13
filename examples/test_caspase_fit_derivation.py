# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 14:37:08 2017

@author: Agus
"""

# import packages to be used
import os
import pathlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from scipy.signal import savgol_filter as savgol
from scipy.interpolate import splrep, splev

# import not registered modules
this_dir = pathlib.Path(r'C:\Users\Agus\Documents\Laboratorio\Imaging three sensors\Analysis')
#os.chdir(r'C:\Users\Admin\Documents\Agus\Imaging three sensors\FRET_Transformation')
os.chdir(r'C:\Users\Agus\Documents\Laboratorio\Imaging three sensors\FRET_Transformation')
import transformation as tf
import caspase_fit as cf
os.chdir(str(this_dir))

#%% load data

df = pd.read_pickle('OneCaspFiltered.pandas')
#df = pd.read_pickle('2017-01-17_OneCasp_mCit Cas8 mCit_b_100.pandas')
fluorophores = ['YFP', 'mKate', 'TFP']
timepoints = 10
#%% Define function to find maximum complex peaks

def replace_nan(curve):
    curve = pd.Series(curve)
    curve.interpolate(method="linear")
    
    curve = curve.values
    curve = curve[np.isfinite(curve)]
    return curve

def find_complex(df, pp, window=13, poly=6):
    time = np.arange(0, 50*timepoints, timepoints)
    for fluo in fluorophores:
        fils = []
        ders = []
        maxs = []
        for i in df.index:
            fig, axs = plt.subplots(2,1, sharex=True, figsize=(10,12))
            if np.isfinite(df[fluo+'_rate'][i]):
                
                r = df['r_'+fluo][i]
                x0 = df[fluo+'_x0'][i]
                rate = df[fluo+'_rate'][i]
                
                r_reg, ind = tf.sigmoid_region(x0, rate, r, minimal=0.00001)
                r_reg = replace_nan(r_reg)
                this_time = time[ind:ind+len(r_reg)]
                if all(np.isfinite(r_reg)):
                    r_fil = savgol(r_reg, window, poly)
                    r_der = savgol(r_reg, window, poly, deriv=1, delta=timepoints)
                    
                    t = np.arange(ind*timepoints, (ind+len(r_reg))*timepoints)
                    f = splrep(this_time, r_der, k=3, s=0)
                    der_interp = splev(t, f, der=0)
                    
                    max_act = np.where(der_interp==np.max(der_interp))[0]+ind*timepoints
                    
                    axs[0].plot(this_time, r_reg)
                    axs[0].plot(this_time, r_fil)
                    axs[0].set_ylabel('fraction')
                    
                    axs[1].plot(this_time, r_der)
                    axs[1].plot(t, der_interp)
                    axs[1].set_ylabel('complex')
                    axs[1].set_xlabel('time (min.)')
                    
                    plt.suptitle('fluo:'+fluo+' obj:'+str(i)+' exp:'+df.Content_YFP[i]+' max:'+str(max_act))
                    pp.savefig()
                    
                    
                    fils.append(r_fil)
                    ders.append(r_der)
                    maxs.append(max_act)
                    
                else:
                    print(i)
                    fils.append([np.nan])
                    ders.append([np.nan])
                    maxs.append(np.nan)
                
            else:
                fils.append([np.nan])
                ders.append([np.nan])
                maxs.append(np.nan)
            plt.close()
            
        df[fluo+'_r_filtered'] = fils
        df[fluo+'_r_complex'] = ders
        df[fluo+'_max_activity'] = maxs
        
    return df

with PdfPages(r'C:\Users\Agus\Documents\Laboratorio\Imaging three sensors\Deriving\results\OneCasp_derivations.pdf') as pp:
    df = find_complex(df, pp)