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
    curve = pd.Series(curve)
    curve.interpolate(method="linear")
    
    curve = curve.values
    curve = curve[np.isfinite(curve)]
    return curve


def get_max_ind(compl):
    #TODO: need corrections because it's not working
    compl = compl[:]
    ind_max = np.where(compl==np.max(compl))[0]
    while len(compl)==(ind_max+1) and len(compl)>1:
        compl = compl[:-1]
        ind_max = np.where(compl==np.max(compl))[0]
    return ind_max

    
def find_complex(df, pp, window=13, poly=6):
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
        df = pd.read_pickle('OneCaspFiltered.pandas')
        
        filename = 'OneCasp_derivations_win%02d_pol%02d.pdf' % (win, pol)
        pandasname = 'OneCasp_derivations_win%02d_pol%02d.pandas' % (win, pol)
        pdf_path = res_path.joinpath(filename)
        pandas_path = res_path.joinpath(pandasname)
        with PdfPages(str(pdf_path)) as pp:
            df = find_complex(df, pp, window=win, poly=pol)
            df.to_pickle(str(pandas_path))