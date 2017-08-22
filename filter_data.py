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

noErode_df = pd.read_pickle(r'D:\Agus\Imaging three sensors\aniso_para_agustin\20131212_pos30\filtered_newnoErode_df.pandas')
old_noErode_df = pd.read_pickle(r'D:\Agus\Imaging three sensors\aniso_para_agustin\20131212_pos30\filtered_noErode_df.pandas')

Erode_df = pd.read_pickle(r'D:\Agus\Imaging three sensors\aniso_para_agustin\20131212_pos30\filtered_newErode_5_df.pandas')
old_Erode_df = pd.read_pickle(r'D:\Agus\Imaging three sensors\aniso_para_agustin\20131212_pos30\filtered_Erode_5_df.pandas')


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


def plot_all_curves_withFit(df):
    for i in df.index:
        if any([df[fluo+'_ok_1'][i] for fluo in fluorophores]):
            for fluo in fluorophores:
                plt.plot(time_coarse, df[fluo+'_r_mean'][i], Colors[fluo], label='mean r '+fluo)
                plt.plot(time_coarse, df[fluo+'_r_from_i'][i], Colors[fluo]+'--', label='mean I '+fluo)
                popt = [df[fluo+'_base'][i], df[fluo+'_amplitude'][i], df[fluo+'_rate'][i], df[fluo+'_x0'][i]]
                plt.plot(time_fine, cf.sigmoid(time_fine, *popt), 'x'+Colors[fluo], label=fluo+' fit')
                fig = plt.gcf()
                fig.set_size_inches(7, 5)
            plt.title(df['object'][i])
            plt.legend(loc=4)
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


def first_filter(df, col_to_filter='r_from_i'):
    # All fluorophores need to be plotted to understand better what to filter
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


def add_pre_post(df, col, colname):
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


#%% generate pdf with scatter of pre vs pos

from matplotlib.backends.backend_pdf import PdfPages

pp = PdfPages(r'D:\Agus\Imaging three sensors\aniso_para_agustin\20131212_pos30\preVSpos.pdf')

for fluo in fluorophores:
    plt.scatter(old_noErode_df[fluo+'_r_mean_pos'].values, old_noErode_df[fluo+'_r_mean_pre'].values, color=Colors[fluo])
    plt.title(fluo+' no erode old mean r')
    plt.xlabel('pos Anisotropy')
    plt.ylabel('pre Anisotropy')
    pp.savefig()
    plt.show()
    
    plt.scatter(old_noErode_df[fluo+'_r_from_i_pos'].values, old_noErode_df[fluo+'_r_from_i_pre'].values, color=Colors[fluo])
    plt.title(fluo+' no erode old r from mean i')
    plt.xlabel('pos Anisotropy')
    plt.ylabel('pre Anisotropy')
    pp.savefig()
    plt.show()
    
    plt.scatter(old_Erode_df[fluo+'_r_mean_pos'].values, old_Erode_df[fluo+'_r_mean_pre'].values, color=Colors[fluo])
    plt.title(fluo+' erode 5 old mean r')
    plt.xlabel('pos Anisotropy')
    plt.ylabel('pre Anisotropy')
    pp.savefig()
    plt.show()
    
    plt.scatter(old_Erode_df[fluo+'_r_from_i_pos'].values, old_Erode_df[fluo+'_r_from_i_pre'].values, color=Colors[fluo])
    plt.title(fluo+' erode 5 old r from mean i')
    plt.xlabel('pos Anisotropy')
    plt.ylabel('pre Anisotropy')
    pp.savefig()
    plt.show()
    
    plt.scatter(noErode_df[fluo+'_r_from_i_pos'].values, noErode_df[fluo+'_r_from_i_pre'].values, color=Colors[fluo])
    plt.title(fluo+' no erode new r from mean i')
    plt.xlabel('pos Anisotropy')
    plt.ylabel('pre Anisotropy')
    pp.savefig()
    plt.show()
    
    plt.scatter(Erode_df[fluo+'_r_from_i_pos'].values, Erode_df[fluo+'_r_from_i_pre'].values, color=Colors[fluo])
    plt.title(fluo+' erode 5 new r from mean i')
    plt.xlabel('pos Anisotropy')
    plt.ylabel('pre Anisotropy')
    pp.savefig()
    plt.show()
    
    plt.scatter(old_noErode_df[fluo+'_r_mean_pos'].values, old_noErode_df[fluo+'_r_mean_pre'].values, color=Colors[fluo], label=fluo+' no erode old mean r')
    plt.scatter(old_noErode_df[fluo+'_r_from_i_pos'].values, old_noErode_df[fluo+'_r_from_i_pre'].values, color=Colors[fluo], marker='x', label=fluo+' no erode old r from mean i')
    plt.title(fluo+' no erode old mean r vs no erode old r from mean i')
    plt.xlabel('pos Anisotropy')
    plt.ylabel('pre Anisotropy')
    pp.savefig()
    plt.show()
    
    plt.scatter(old_Erode_df[fluo+'_r_mean_pos'].values, old_Erode_df[fluo+'_r_mean_pre'].values, color=Colors[fluo], label=fluo+' erode 5 old mean r')
    plt.scatter(old_Erode_df[fluo+'_r_from_i_pos'].values, old_Erode_df[fluo+'_r_from_i_pre'].values, color=Colors[fluo], marker='x', label=fluo+' erode 5 old r from mean i')
    plt.title(fluo+' erode 5 old mean r vs erode 5 old r from mean i')
    plt.xlabel('pos Anisotropy')
    plt.ylabel('pre Anisotropy')
    pp.savefig()
    plt.show()
    
    plt.scatter(old_noErode_df[fluo+'_r_from_i_pos'].values, old_noErode_df[fluo+'_r_from_i_pre'].values, color=Colors[fluo], label=fluo+' no erode old r from i')
    plt.scatter(noErode_df[fluo+'_r_from_i_pos'].values, noErode_df[fluo+'_r_from_i_pre'].values, color=Colors[fluo], marker='x', label=fluo+' no erode new r from mean i')
    plt.title(fluo+' no erode old r from i vs no erode new r from mean i')
    plt.xlabel('pos Anisotropy')
    plt.ylabel('pre Anisotropy')
    pp.savefig()
    plt.show()
    
    plt.scatter(old_Erode_df[fluo+'_r_from_i_pos'].values, old_Erode_df[fluo+'_r_from_i_pre'].values, color=Colors[fluo], label=fluo+' erode 5 old r from i')
    plt.scatter(Erode_df[fluo+'_r_from_i_pos'].values, Erode_df[fluo+'_r_from_i_pre'].values, color=Colors[fluo], marker='x', label=fluo+' erode 5 new r from mean i')
    plt.title(fluo+' erode 5 old r from i vs erode 5 new r from mean i')
    plt.xlabel('pos Anisotropy')
    plt.ylabel('pre Anisotropy')
    pp.savefig()
    plt.show()

pp.close()