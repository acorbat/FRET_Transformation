# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 11:01:50 2017

@author: Admin
"""

import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import Anisotropy_Functions as af
import Transformation as tf
import Caspase_Fit as cf

#from multiprocessing import Pool


#%% import data

data = pd.read_pickle('ExpOneCasp.pandas')

#%% Define constants to be used

timepoints = 10 #min
fluorophores = ['TFP','mKate','YFP'] #[Cas3,Cas8,Cas9]
Colors = {'TFP' : 'g', 'YFP' : 'y', 'mKate' : 'r'}
sigmoid_parameters = ['base', 'amplitude','rate', 'x0']

time_coarse = np.arange(0, 50*timepoints, timepoints)
time_fine = np.arange(0, 50*timepoints)

#%% Add anisotropy and fluorescence

for fluo in fluorophores:
    rs = []
    fs = []
    I_par_ns = []
    I_per_ns = []
    
    for i in data.index:
        I_par = data['par_'+fluo][i]
        I_per = data['per_'+fluo][i]
        r = af.Anisotropy_FromInt(I_par, I_per)
        f = af.Fluos_FromInt(I_par, I_per)
        I_par_n = I_par/f
        I_per_n = I_per/f
        
        rs.append(r)
        fs.append(f)
        I_par_ns.append(I_par_n)
        I_per_ns.append(I_per_n)
        
    data['r_'+fluo] = rs
    data['f_'+fluo] = fs
    data['I_par_n_'+fluo] = I_par_ns
    data['I_per_n_'+fluo] = I_per_ns

#%% First windowed sigmoid fit to estimate parameters

for fluo in fluorophores:
    this_popts = []
    for i in data.index:
        try:
            #this_popt, _, _, _ = tf.windowFit(cf.sigmoid, data['r_'+fluo][i])
            this_popt = tf.windowFit(cf.sigmoid, data['r_'+fluo][i])
        except:
            this_popt = [np.nan]*4
        
        this_popts.append(this_popt)
        
    data['first_popts_'+fluo] = this_popts

#%% Ask which window fit corresponds

for fluo in fluorophores:
    for i in data.index:
        popts = data['first_popts_'+fluo][i]
        if not isinstance(popts[0], float):
            for popt in popts:
                this_answer = []
                for _popt in popts:
                    plt.plot(time_fine, cf.sigmoid(time_fine, *_popt),'--')
                plt.plot(time_coarse, data['r_'+fluo][i])
                plt.plot(time_fine, cf.sigmoid(time_fine, *popt))
                
                plt.show()
                
                answer = input('is this best fit?')
                if answer
        else:
            plt.plot(time_coarse, data['r_'+fluo][i])
            plt.plot(time_fine, cf.sigmoid(time_fine, *popts))
            plt.show()
            
            answer = input('is this best fit?')