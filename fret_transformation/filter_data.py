# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 15:49:25 2017

@author: Admin
"""
import numpy as np
import matplotlib.pyplot as plt

import anisotropy_functions as af
import transformation as tf
import caspase_fit as cf


#%% Define constants to be used

timepoints = 10 #min
fluorophores = ['TFP','mKate','YFP'] #[Cas3,Cas8,Cas9]
Colors = {'TFP' : 'g', 'YFP' : 'y', 'mKate' : 'r'}

time_coarse = np.arange(0, 90*timepoints, timepoints)
time_fine = np.arange(0, 90*timepoints)


#%% calculate r from mean par and per

def r_from_i_to_df(df):
    """
    Calculates anistropy from par_mean and per_mean values for each fluorophore.
    """
    for fluo in fluorophores:
        this_rs = []
        for i in df.index:
            this_r = af.Anisotropy_FromInt(df[fluo+'_par_mean'][i], df[fluo+'_per_mean'][i])
            this_rs.append(this_r)
        df[fluo+'_r_from_i'] = this_rs
    return df


#%% question function and obvious filter

def apoptotic_popts(base, amplitude, rate, x0):
    """
    Quick checks that parameters of sigmoid are within possible range.
    
    0.1<base<0.5
    0.001<amplitude<0.5
    0<rate
    0<x0
    """
    if base>0.1 and base<0.5 and amplitude>0.003 and amplitude<0.5 and rate>0 and x0>0:
        return True
    else:
        return False

def ask_question(question='?'):
    """
    Asks yes/no question to user. If answered incorrectly 3 times, it raises ValueError.
    """
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
    """
    Applies sigmoid window fit to all fluorophores y_col curves and saves it to dataframe first_popts.
    """
    for fluo in fluorophores:
        this_popts = []
        for i in df.index:
            try:
                #this_popt, _, _, _ = tf.windowFit(cf.sigmoid, df['r_'+fluo][i])
                this_popt = tf.windowFit(cf.sigmoid, df[fluo+'_'+y_col][i])
            except:
                this_popt = [np.nan]*4
                this_popt = [this_popt]
            
            this_popts.append(this_popt)
            
        df[fluo+'_first_popts'] = this_popts
    return df


def first_filter(df, col_to_filter='r_from_i'):
    """
    Asks whether a curve is apoptotic or not taking into consideration if there is any plausible apoptic popt in first_popts
    while showing a plot of all fluorophores with all its sigmoid fits. col_to_filter is the curve plotted.
    """
    for fluo in fluorophores:
        ok_1 = []
        for i in df.index:
            popts = df[fluo+'_first_popts'][i]
            if any([apoptotic_popts(*popt) for popt in popts]):
                plt.plot(time_coarse, df[fluo+'_'+col_to_filter][i])
                for popt in popts:
                    plt.plot(time_fine, cf.sigmoid(time_fine, *popt))
                for new_fluo in fluorophores:
                    plt.plot(time_coarse, df[new_fluo+'_'+col_to_filter][i], '--'+Colors[new_fluo])
                plt.title(fluo+' '+str(i))
                plt.ylim((0.22, 0.35))
                plt.show()
                
                answer = ask_question(question='is this an apoptotic curve?')
                ok_1.append(answer)
            else:
                ok_1.append(False)
            
        df[fluo+'_ok_1'] = ok_1
    return df


def second_filter(df, col_to_filter='r_from_i'):
    """
    Sweeps through the accepted as apoptotic curves asking which of the plausible 
    popts is the best.
    """
    for fluo in fluorophores:
        best_popts = []
        for i in df.index:
            these_popts = []
            popts = df[fluo+'_first_popts'][i]
            if df[fluo+'_ok_1'][i]:
                if not isinstance(popts[0], float):
                    c=0
                    for popt in popts:
                        c+=1
                        if apoptotic_popts(*popt):
                            for _popt in popts:
                                plt.plot(time_fine, cf.sigmoid(time_fine, *_popt),'--')
                            plt.plot(time_coarse, df[fluo+'_'+col_to_filter][i])
                            plt.plot(time_fine, cf.sigmoid(time_fine, *popt))
                            
                            plt.title(fluo+' '+str(i)+' '+str(c))
                            plt.ylim((0.2, 0.35))
                            plt.show()
                            
                            answer = ask_question(question='is this the best popt?')
                            these_popts.append(answer)
                        else:
                            these_popts.append(False)
                    
                else:
                    if apoptotic_popts(*popts):
                        plt.plot(time_coarse, df[fluo+'_'+col_to_filter][i])
                        plt.plot(time_fine, cf.sigmoid(time_fine, *popts))
                        plt.ylim((0.2, 0.35))
                        plt.show()
                        
                        answer = ask_question(question='is this a good popt?')
                        
                        these_popts.append(answer)
                    else:
                        these_popts.append(False)
            else:
                these_popts.append(False)
                
            best_popts.append(these_popts)
        df[fluo+'_best_popts'] = best_popts
        
    return df


def set_popts(df):
    """
    sets the last best popt chosen as the parameters of the sigmoid fit.
    """
    for fluo in fluorophores:
        bases = []
        amps = []
        rates = []
        x0s = []
        
        for i in df.index:
            popts = df[fluo+'_first_popts'][i]
            best_popt = df[fluo+'_best_popts'][i]
            if any(best_popt):
                for popt, best in zip(popts, best_popt):
                    if best:
                        base, amplitude, rate, x0 = popt
            else:
                base, amplitude, rate, x0 = [np.nan]*4
            
            bases.append(base)
            amps.append(amplitude)
            rates.append(rate)
            x0s.append(x0)
        
        df[fluo+'_amplitude'] = amps
        df[fluo+'_base'] = bases
        df[fluo+'_rate'] = rates
        df[fluo+'_x0'] = x0s
    return df


def add_pre_post(df, col, colname, timepoints=10):
    """
    Uses sigmoid parameters to estimate mean pre and post values of col curve.
    colname is the suffix of the new column where results are saved.
    """
    for fluo in fluorophores:
        posts = []
        pres = []
        for i in df.index:
            if np.isfinite(df[fluo+'_base'][i]):
                post = tf.post_region(df[fluo+'_x0'][i], df[fluo+'_rate'][i], df[fluo+'_'+col][i], timepoints)
                pre  = tf.pre_region(df[fluo+'_x0'][i], df[fluo+'_rate'][i], df[fluo+'_'+col][i], timepoints)
                mean_post = np.nanmean(post)
                mean_pre  = np.nanmean(pre)
                posts.append(mean_post)
                pres.append(mean_pre)
            else:
                posts.append(np.nan)
                pres.append(np.nan)
        
        df[fluo+'_'+colname+'_pre'] = pres
        df[fluo+'_'+colname+'_pos'] = posts
    return df