# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 15:49:25 2017

@author: Admin
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

os.chdir(r'D:\Agus\Imaging three sensors\FRET_Transformation')
import Anisotropy_Functions as af
import Transformation as tf
import Caspase_Fit as cf

#%% import data

data = pd.read_pickle(r'D:\Agus\Imaging three sensors\aniso_para_agustin\20131212_pos30\pos30_noErode_df.pandas')


#%% Define constants to be used

timepoints = 10 #min
fluorophores = ['TFP','mKate','YFP'] #[Cas3,Cas8,Cas9]
Colors = {'TFP' : 'g', 'YFP' : 'y', 'mKate' : 'r'}
sigmoid_parameters = ['base', 'amplitude','rate', 'x0']

time_coarse = np.arange(0, 90*timepoints, timepoints)
time_fine = np.arange(0, 90*timepoints)


#%% calculate r from mean par and per

def r_from_i_to_df(df):
    for fluo in fluorophores:
        this_rs = []
        for i in df.index:
            this_r = af.Anisotropy_FromInt(df[fluo+'_par_mean'][i], df[fluo+'_per_mean'][i])
            this_rs.append(this_r)
        df[fluo+'_r_from_i'] = this_rs
    return df


#%% Plot results

def plot_all_curves(df):
    for i in df.index:
        if not all([all(np.isnan(df[fluo+'_r_mean'][i])) for fluo in fluorophores]):
            for fluo in fluorophores:
                plt.plot(time_coarse, df[fluo+'_r_mean'][i], Colors[fluo], label='mean r '+fluo)
                plt.plot(time_coarse, df[fluo+'_r_from_i'][i], Colors[fluo]+'--', label='mean I '+fluo)
                fig = plt.gcf()
                fig.set_size_inches(7, 5)
            plt.title(df['object'][i])
            plt.legend(loc=4)
            plt.show()
            print(i)


def plot_curves_and_areas(df):
    for i in df.index:
        if not all([all(np.isnan(df[fluo+'_r_mean'][i])) for fluo in fluorophores]):
            fig, axs = plt.subplots(2, 1, figsize=(8, 10), sharex=True)
            for fluo in fluorophores:
                axs[0].plot(time_coarse, df[fluo+'_r_mean'][i], Colors[fluo], label='mean r '+fluo)
                axs[0].plot(time_coarse, df[fluo+'_r_from_i'][i], Colors[fluo]+'--', label='mean I '+fluo)
                axs[0].legend(loc=4)
                
                axs[1].plot(time_coarse, df[fluo+'_par_area'][i]-df[fluo+'_par_nanpixs'][i], Colors[fluo], label='area '+fluo)
                axs[1].legend(loc=3)
            plt.suptitle(df['object'][i])
            plt.show()
            print(i)


def plot_oldVSnew(new_df, old_df):
    for i in old_df.index:
        if not all([all(np.isnan(old_df[fluo+'_r_mean'][i])) for fluo in fluorophores]):
            fig, axs = plt.subplots(2, 1, figsize=(8, 10), sharex=True)
            for fluo in fluorophores:
                axs[0].plot(time_coarse, old_df[fluo+'_r_mean'][i], Colors[fluo], label='old mean r '+fluo)
                axs[0].plot(time_coarse, old_df[fluo+'_r_from_i'][i], Colors[fluo]+'--', label='old mean I '+fluo)
                axs[0].plot(time_coarse, new_df[fluo+'_r_from_i'][i], Colors[fluo]+'-.', label='new mean I '+fluo)
                axs[0].legend(loc=4)
                
                axs[1].plot(time_coarse, new_df[fluo+'_par_area'][i], Colors[fluo], label='area '+fluo)
                axs[1].legend(loc=3)
            plt.suptitle(new_df['object'][i])
            plt.show()
            print(i)


#%% Useful Functions

def apoptotic_popts(base, amplitude, rate, x0):
    if base>0.1 and base<0.5 and amplitude>0.001 and amplitude<0.5 and rate>0 and x0>0:
        return True
    else:
        return False

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

#%% Prepare first fit and filter

def general_fit(df, y_col='r_from_i'):
    for fluo in fluorophores:
        this_popts = []
        for i in df.index:
            try:
                #this_popt, _, _, _ = tf.windowFit(cf.sigmoid, df['r_'+fluo][i])
                this_popt = tf.windowFit(cf.sigmoid, df[fluo+'_'+y_col][i])
            except:
                this_popt = [np.nan]*4
            
            this_popts.append(this_popt)
            
        df[fluo+'_first_popts'] = this_popts
    return df


def first_filter(df):
    # All fluorophores need to be plotted to understand better what to filter
    for fluo in fluorophores:
        ok_1 = []
        for i in df.index:
            popts = df[fluo+'_first_popts'][i]
            if any([apoptotic_popts(*popt) for popt in popts]):
                plt.plot(time_coarse, df[fluo+'_r_from_i'][i])
                for popt in popts:
                    plt.plot(time_fine, cf.sigmoid(time_fine, *popt))
                for new_fluo in fluorophores:
                    plt.plot(time_coarse, df[fluo+'_r_from_i'][i], '--'+Colors[fluo])
                plt.title(fluo+' '+str(i))
                plt.show()
                
                answer = ask_question(question='is this an apoptotic curve?')
                ok_1.append(answer)
            else:
                ok_1.append(False)
            
        df[fluo+'_ok_1'] = ok_1
    return df

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
    best_popts = []
    for i in data.index:
        these_popts = []
        popts = data['first_popts_'+fluo][i]
        if not isinstance(popts[0], float):
            c=0
            for popt in popts:
                c+=1
                if apoptotic_popts(*popt):
                    for _popt in popts:
                        plt.plot(time_fine, cf.sigmoid(time_fine, *_popt),'--')
                    plt.plot(time_coarse, data['r_'+fluo][i])
                    plt.plot(time_fine, cf.sigmoid(time_fine, *popt))
                    
                    plt.title(fluo+' '+str(i)+' '+str(c))
                    plt.show()
                    
                    answer = ask_question(question='is this the best popt?')
                    these_popts.append(answer)
                else:
                    these_popts.append(False)
            best_popts.append(these_popts)
        else:
            if apoptotic_popts(*popts):
                plt.plot(time_coarse, data['r_'+fluo][i])
                plt.plot(time_fine, cf.sigmoid(time_fine, *popts))
                plt.show()
                
                answer = ask_question(question='is this a good popt?')
                
                these_popts.append(answer)
            else:
                these_popts.append(False)
    
    data['best_popts_'+fluo] = best_popts

#%% Add 164 TFP, 0 mKate

#%% Double 226 TFP, 376 TFP, 262 mKate, 684 mKate, 253 YFP, 291 YFP

#%% Save best parameters

for fluo in fluorophores:
    bases = []
    amplitudes = []
    rates = []
    x0s = []
    for i in data.index:
        these_popts = data['first_popts_'+fluo][i]
        these_best = data['best_popts_'+fluo][i]
        if any(these_best):
            for is_best, popt in zip(these_best, these_popts):
                if is_best:
                    bases.append(popt[0])
                    amplitudes.append(popt[1])
                    rates.append(popt[2])
                    x0s.append(popt[3])
                    break
        else:
            bases.append(np.nan)
            amplitudes.append(np.nan)
            rates.append(np.nan)
            x0s.append(np.nan)
    
    data[fluo+'_base'] = bases
    data[fluo+'_amplitude'] = amplitudes
    data[fluo+'_rate'] = rates
    data[fluo+'_x0'] = x0s

#%% Save file to pandas

data.to_pickle('OneCaspFiltered.pandas')