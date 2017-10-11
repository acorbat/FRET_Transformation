# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 18:28:05 2017

@author: Admin
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

os.chdir(r'D:\Agus\Imaging three sensors\FRET_Transformation')
import Caspase_Fit as cf
#%% Define constants to be used

timepoints = 10 #min
fluorophores = ['TFP','mKate','YFP'] #[Cas3,Cas8,Cas9]
Colors = {'TFP' : 'g', 'YFP' : 'y', 'mKate' : 'r'}
sigmoid_parameters = ['base', 'amplitude','rate', 'x0']

time_coarse = np.arange(0, 90*timepoints, timepoints)
time_fine = np.arange(0, 90*timepoints)


#%% Plot results

def plot_all_curves(df):
    """
    Plots anisotropy from mean anisotropy and from mean intensity of all fluorophores for all objects.
    """
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
    """
    Plots anisotropy from mean anisotropy and from mean intensity of all fluorophores for all objects,
    as well as the area of the cell.
    """
    for i in df.index:
        if not all([all(np.isnan(df[fluo+'_r_mean'][i])) for fluo in fluorophores]):
            fig, axs = plt.subplots(2, 1, figsize=(8, 10), sharex=True)
            for fluo in fluorophores:
                axs[0].plot(time_coarse, df[fluo+'_r_mean'][i], Colors[fluo], label='mean r '+fluo)
                axs[0].plot(time_coarse, df[fluo+'_r_from_i'][i], Colors[fluo]+'--', label='mean I '+fluo)
                axs[0].legend(loc=4)
                
                axs[1].plot(time_coarse, df[fluo+'_par_area'][i], Colors[fluo], label='area '+fluo)
                axs[1].legend(loc=3)
            plt.suptitle(df['object'][i])
            plt.show()
            print(i)


def plot_oldVSnew(new_df, old_df):
    """
    Plots anisotropy from mean anisotropy and from mean intensities of the old method from old_df,
    superposed to anistropy from intensities of the new method. Subplot of area is added.
    """
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
    """
    Plots anisotropy from mean anisotropy and from mean intensities of the old method from old_df,
    superposed to sigmoid fit chosen.
    """
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


#%% generate pdf with scatter of pre vs pos

def pdf_preVSpos(pdfname):
    pp = PdfPages(pdfname)
    
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


#%% save plot old vs new to pdf

def plot_oldVSnew_pdf(pdfname, new_df, old_df):
    pp = PdfPages(pdfname)
    for i in old_df.index:
        if not all([all(np.isnan(old_df[fluo+'_r_mean'][i])) for fluo in fluorophores]):
            fig, axs = plt.subplots(2, 1, figsize=(8, 10), sharex=True)
            for fluo in fluorophores:
                axs[0].plot(time_coarse, old_df[fluo+'_r_mean'][i], Colors[fluo], label='old mean r '+fluo)
                axs[0].plot(time_coarse, old_df[fluo+'_r_from_i'][i], Colors[fluo]+'--', label='old mean I '+fluo)
                axs[0].plot(time_coarse, new_df[fluo+'_r_from_i'][i], Colors[fluo]+'-.', label='new mean I '+fluo)
                axs[0].legend(loc=4)
                axs[0].set_ylabel('Anisotropy')
                
                axs[1].plot(time_coarse, new_df[fluo+'_par_area'][i], Colors[fluo], label='area '+fluo)
                axs[1].legend(loc=3)
                axs[1].set_xlabel('time (min)')
                axs[1].set_ylabel('Area (pxs)')
            plt.suptitle(new_df['object'][i])
            pp.savefig()
            plt.show()
            print(i)
    pp.close()

#%% import data

noErode_df = pd.read_pickle(r'D:\Agus\Imaging three sensors\aniso_para_agustin\20131212_pos30\filtered_newnoErode_df.pandas')
old_noErode_df = pd.read_pickle(r'D:\Agus\Imaging three sensors\aniso_para_agustin\20131212_pos30\filtered_noErode_df.pandas')

Erode_df = pd.read_pickle(r'D:\Agus\Imaging three sensors\aniso_para_agustin\20131212_pos30\filtered_newErode_5_df.pandas')
old_Erode_df = pd.read_pickle(r'D:\Agus\Imaging three sensors\aniso_para_agustin\20131212_pos30\filtered_Erode_5_df.pandas')


pdf_name = r'D:\Agus\Imaging three sensors\aniso_para_agustin\20131212_pos30\preVSpos.pdf'

pdf_preVSpos(pdf_name)

pdf_name = r'D:\Agus\Imaging three sensors\aniso_para_agustin\20131212_pos30\oldVSnew_noErode.pdf'

plot_oldVSnew_pdf(pdf_name, noErode_df, old_noErode_df)

pdf_name = r'D:\Agus\Imaging three sensors\aniso_para_agustin\20131212_pos30\oldVSnew_Erode.pdf'

plot_oldVSnew_pdf(pdf_name, Erode_df, old_Erode_df)