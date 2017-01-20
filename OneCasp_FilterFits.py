# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 11:13:21 2017

@author: Admin
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import Anisotropy_Functions as af
import Transformation as tf
import Caspase_Fit as cf

#%% Define useful variables

timepoints = 10 #min
fluorophores = ['TFP','mKate','YFP'] #[Cas3,Cas8,Cas9]
Colors = {'TFP' : 'g', 'YFP' : 'y', 'mKate' : 'r'}
sigmoid_parameters = ['base', 'amplitude','rate', 'x0']
fit_parameters = ['Am', 'Ad', 'b', 'm', 't0', 'rate', 'k', 'casp', 'sens', 'prod']

time_coarse = np.arange(0, 50*timepoints, timepoints)
time_fine = np.arange(0, 50*timepoints)

#%% Import dataframe

data = pd.read_pickle('2017-01-17_OneCasp_mCit Cas3 mCit + STS_b_100.pandas')
raw_data = pd.read_pickle('OneCaspFiltered.pandas')

#%% Plot all curves and fits

def plot_results(i):
    for fluo in fluorophores:
        if np.isfinite(data[fluo+'_t0'][i]):
            plt.scatter(time_coarse, cf.Normalize(raw_data['r_'+fluo][i]), c=Colors[fluo])
            plt.plot(time_fine, data[fluo+'_casp'][i][:500], Colors[fluo])
            plt.plot(time_fine, data[fluo+'_prod'][i][:500], Colors[fluo]+'--')
    plt.title(raw_data.Content_YFP[i]+' '+str(i))
    plt.xlabel('Time (min.)')
    plt.ylabel('Fraction')
    plt.show()
    print(i)

def ask_question(question='?'):    
    c= 0
    while c<=3:
        response = input(question)
        
        if response=='y':
            return True
        elif response=='n':
            return False
        else:
            print('answer y or n')
            c+=1
    raise ValueError

#%% reFilter data

for i in data.index:
    if any(np.isfinite([data[fluo+'_t0'][i] for fluo in fluorophores])):
        for fluo in fluorophores:
            plt.scatter(time_coarse, raw_data['r_'+fluo][i], c=Colors[fluo])
        plt.show()
        
        plot_results(i)
        
        good_fit = ask_question(question='was this a good fit?')
        
        if not good_fit:
            for fluo in fluorophores:
                for parameter in fit_parameters:
                    data[fluo+'_'+parameter][i] = np.nan

# 162 and 340 is incomplete, 166 has problems with YFP, check 175 and 365 and 396 and 432 and 441 and 945 and 989, 217 and 343 wrong YFP, 253 and 262 and 266 and 277 and 291 different pairs, 263 can be rescued, 341 wrong TFP

#%% Copy missing useful columns and save dataframe

for fluo in fluorophores:
    data['Content_'+fluo] = raw_data['Content_'+fluo]
    data['r_'+fluo] = raw_data['r_'+fluo]

data.to_pickle('OneCaspFitted.pandas')