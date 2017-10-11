# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 16:57:12 2017

@author: Agus
"""

# import packages to be used
import os
import pathlib
import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import lmfit
from matplotlib.backends.backend_pdf import PdfPages

# import not registered modules
this_dir = pathlib.Path(r'C:\Users\Agus\Documents\Laboratorio\Imaging three sensors\Modelling')
#os.chdir(r'C:\Users\Admin\Documents\Agus\Imaging three sensors\FRET_Transformation')
os.chdir(r'C:\Users\Agus\Documents\Laboratorio\Imaging three sensors\FRET_Transformation')
import caspase_fit as cf
os.chdir(str(this_dir))

#%% Generate DataFrame

# Generate dataframe from the local files
def Generate_DataFrame():
    filenames = [File for File in os.listdir() if File.startswith('SweepAllcc')]
    
    Data = []
    for filename in filenames:
        this_cc = filename.split('.')[0][-2]
        this_cas = filename.split('.')[0][-1]
        data = np.loadtxt(filename)
        this_time = data[:,0]
        this_sensor = data[:,1]
        this_caspase = data[:,2]
        
        if this_cas=='1':
            this_Data = {'cas'+this_cas+'_time' : this_time, 
                         'cas'+this_cas+'_sens' : this_sensor,
                         'cas'+this_cas+'_casp' : this_caspase}
        else:
            this_Data['cas'+this_cas+'_time'] = this_time
            this_Data['cas'+this_cas+'_sens'] = this_sensor
            this_Data['cas'+this_cas+'_casp'] = this_caspase
        
        if this_cas=='3':
            Data.append(this_Data)
    
    Data = pd.DataFrame(Data)
    return Data

df = Generate_DataFrame()

#%% Plot loaded caspases

Caspases = {'1':'Casp-3', '2':'Casp-8', '3':'Casp-9'}
Colors = {'1':'g', '2':'r', '3':'y'}
Concentrations = {'0':'0', '1':'10', '2':'100', '3':'1000', '4':'10000'}

result_path = this_dir.joinpath('results')
pdf_path = result_path.joinpath('parameter_fit_gompertz.pdf')
pp = PdfPages(str(pdf_path))

def Plot_Caspases(df):
    for cc in df.index:
        plt.figure(figsize=(7,5))
        for cas in range(1, 4):
            plt.plot(df['cas'+str(cas)+'_time'][cc], cf.Normalize(df['cas'+str(cas)+'_casp'][cc]), Colors[str(cas)], label=Caspases[str(cas)])
            plt.plot(df['cas'+str(cas)+'_time'][cc], cf.Normalize(df['cas'+str(cas)+'_sens'][cc]), Colors[str(cas)]+'--', label=Caspases[str(cas)])
        plt.xlabel('time (min.)')
        plt.ylabel('Active Caspase')
        plt.title(Concentrations[str(cc)])
        plt.legend(loc=2)
        plt.xlim((50, 300))
        pp.savefig()
        plt.close()

Plot_Caspases(df)

#%% Generate models and fits
model = lmfit.Model(cf.simulate)

def residual_model(params, t, data_prod, data_casp):
    
    casp, subs, prod = model.eval(params, t=t, func='gom')
    res = [np.abs(this_prod - this_data_prod) + np.abs(this_casp - this_data_casp) if this_data_prod>0.9 
           else (np.abs(this_prod - this_data_prod) + np.abs(this_casp - this_data_casp))/0.25 
           for (this_prod, this_casp, this_data_prod, this_data_casp) in zip(prod, casp, data_prod, data_casp)]
    return res

def residual(params, t, data_prod):
    
    casp, subs, prod = model.eval(params, t=t, func='gom')
    res = [this_prod - this_data_prod if this_data_prod>0.9 
           else (this_prod - this_data_prod)/0.25 
           for (this_prod, this_data_prod) in zip(prod, data_prod)]
    return res

def fit(res_func, time_estimate, time, data_sens, data_casp=None):
    params = lmfit.Parameters()
    params.add('t_0', value=time_estimate, min=1, max=1000)
    params.add('k', value=0.1, min=1e-5, max=900)
    params.add('rate', value=0.1, min=1e-5, max=900)
    
    if data_casp is not None:
        mini = lmfit.Minimizer(residual_model, params, fcn_args=(time, data_sens, data_casp))
    else:
        mini = lmfit.Minimizer(residual, params, fcn_args=(time, data_sens))
    
    best = mini.minimize()
    
    Other_Initial_parameters = [10, 50]
    for param in itertools.permutations(Other_Initial_parameters):
        params['k'].value = param[0]
        params['rate'].value = param[1]
        
        this_best = mini.minimize()
        
        if this_best.chisqr<best.chisqr:
            best = this_best
        
    return best
    

#%% First do all three caspase fit with caspase and product

casp_params = {}    
for casp in Caspases.keys():
    time = df['cas'+casp+'_time'].values[0]
    data_sens = df['cas'+casp+'_sens'].values[0]
    data_casp = df['cas'+casp+'_casp'].values[0]
    
    best = fit(residual_model, 100, time, data_sens, data_casp=data_casp)
    casp_params[Caspases[casp]] = best.params
    
    plt.plot(time, data_sens, label='sens')
    plt.plot(time, model.eval(best.params, t=time, func='gom')[2], label='fit sens')
    
    plt.plot(time, data_casp, label='casp')
    plt.plot(time, model.eval(best.params, t=time, func='gom')[0], label='fit casp')
    
    plt.xlabel('time (min.)')
    plt.ylabel('active fraction')
    plt.xlim((50, 300))
    plt.legend(loc=4)
    plt.title(Caspases[casp])
    
    pp.attach_note(lmfit.printfuncs.fit_report(best), positionRect=[100, 100, 100, 100])
    pp.savefig()
    plt.close()

#%% close pdf

pp.close()