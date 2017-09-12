# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 16:57:12 2017

@author: Agus
"""

# import packages to be used
import os
import pathlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import lmfit
from matplotlib.backends.backend_pdf import PdfPages

# import not registered modules
this_dir = pathlib.Path(os.getcwd())
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
pdf_path = result_path.joinpath('parameter_fit.pdf')
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

params = lmfit.Parameters()
params.add('t_0', value=100, min=1, max=1000)
params.add('k', value=1, min=1e-5, max=900)
params.add('rate', value=1, min=1e-5, max=900)

plt.figure(figsize=(7,5))
pp.attach_note(lmfit.printfuncs.fit_report(params), positionRect=[450, 450, 300, 300])
pp.savefig()

#%% close pdf

pp.close()