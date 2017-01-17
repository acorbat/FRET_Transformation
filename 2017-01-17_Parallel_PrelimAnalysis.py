# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 11:25:31 2017

@author: Agus
"""
import time

import numpy as np
import pandas as pd

import Anisotropy_Functions as af
import Transformation as tf
import Caspase_Fit as cf

from multiprocessing import Pool


#%% import data
data = pd.read_pickle('OneCaspFiltered.pandas')

#%% define dictionaries of variables to be swept
def preprocess():
    timepoints = 10 #min
    fluorophores = ['TFP','mKate','YFP'] #[Cas3,Cas8,Cas9]
    Colors = {'TFP' : 'g', 'YFP' : 'y', 'mKate' : 'r'}
    sigmoid_parameters = ['base', 'amplitude','rate', 'x0']
    
    time_coarse = np.arange(0, 90*timepoints, timepoints)
    time_fine = np.arange(0, 90*timepoints)
    
    experiments = {}
    for fluo in fluorophores:
        experiments[fluo] = set()
        for i in data.index:
            experiments[fluo].add(data['Content_'+fluo][i])
    
    #%% Generate fluorophore-wise indexes
    indexes = {}
    
    for fluo in fluorophores:
        indexes[fluo] = [i for i in data.index if np.isfinite(data[fluo+'_amplitude'][i]).any()]
    
    #%% Calculate and save monomer and dimer estimated values
    
    for fluo in fluorophores:
        A_M = []
        A_D = []
        A_Diff = []
        I_M = []    
        I_D = []
        b_exp = []
        I_Diff = []
        for i in data.index:
            if i in indexes[fluo]:
                # Estimate anisotropy values for monomer and dimer states
                Aniso_post = tf.post_region(data[fluo+'_x0'][i], data[fluo+'_rate'][i], data['r_'+fluo][i], timepoints)
                Aniso_pre  = tf.pre_region(data[fluo+'_x0'][i], data[fluo+'_rate'][i], data['r_'+fluo][i], timepoints)
                mean_Aniso_post = np.nanmean(Aniso_post)
                mean_Aniso_pre  = np.nanmean(Aniso_pre)
                A_M.append(mean_Aniso_post)
                A_D.append(mean_Aniso_pre)
                A_Diff.append(mean_Aniso_post-mean_Aniso_pre)
                
                # Estimate fluorescence intensity values for monomer and dimer states
                I_post = tf.post_region(data[fluo+'_x0'][i], data[fluo+'_rate'][i], data['f_'+fluo][i], timepoints)
                I_pre  = tf.pre_region(data[fluo+'_x0'][i], data[fluo+'_rate'][i], data['f_'+fluo][i], timepoints)
                mean_I_post = np.nanmean(I_post)
                mean_I_pre  = np.nanmean(I_pre)
                I_M.append(mean_I_post)
                I_D.append(mean_I_pre)
                b_exp.append(mean_I_pre/mean_I_post)
                I_Diff.append(mean_I_post-mean_I_pre)
                
            else:
                A_M.append(np.nan)
                A_D.append(np.nan)
                A_Diff.append(np.nan)
                I_M.append(np.nan)
                I_D.append(np.nan)
                b_exp.append(np.nan)
                I_Diff.append(np.nan)
                
    
        data[fluo+'_Monomer'] = A_M
        data[fluo+'_Dimer'] = A_D
        data[fluo+'_Aniso_Difference'] = A_Diff
        data[fluo+'_I_Monomer'] = I_M
        data[fluo+'_I_Dimer'] = I_D
        data[fluo+'_b_exp'] = b_exp
        data[fluo+'_I_Difference'] = I_Diff
    
    #%% Estimate experimental anisotropies
    sigmas = 2
    
    Mean_Anisotropies = {}
    Std_Anisotropies = {}
    Experimental_Anisotropies = {}
    Experimental_bs = {}
    
    for fluo in fluorophores:
        mean_aniso_M = np.nanmean(data[fluo+'_Monomer'].values)
        mean_aniso_D = np.nanmean(data[fluo+'_Dimer'].values)
        std_aniso_M = np.nanstd(data[fluo+'_Monomer'].values)
        std_aniso_D = np.nanstd(data[fluo+'_Dimer'].values)
        
        Mean_Anisotropies[fluo+'_Monomer'] = mean_aniso_M
        Mean_Anisotropies[fluo+'_Dimer'] = mean_aniso_D
        
        Std_Anisotropies[fluo+'_Monomer'] = std_aniso_M
        Std_Anisotropies[fluo+'_Dimer'] = std_aniso_D
        
        Experimental_Anisotropies[fluo+'_m_max'] = mean_aniso_M + sigmas * std_aniso_M
        Experimental_Anisotropies[fluo+'_m_min'] = mean_aniso_M - sigmas * std_aniso_M
        
        Experimental_Anisotropies[fluo+'_d_max'] = mean_aniso_D + sigmas * std_aniso_D
        Experimental_Anisotropies[fluo+'_d_min'] = mean_aniso_D - sigmas * std_aniso_D
        
        mean_b = np.nanmean(data[fluo+'_b_exp'].values)
        median_b = np.nanmedian(data[fluo+'_b_exp'].values)
        std_b = np.nanstd(data[fluo+'_b_exp'].values)
        
        Experimental_bs[fluo+'_b_min'] = median_b - sigmas * std_b
        Experimental_bs[fluo+'_b_max'] = median_b + sigmas * std_b
        
    return Experimental_Anisotropies, fluorophores, indexes, timepoints, experiments
    
#%% Transform each experiment and save transformation separating each experiment and using different b
    
Experimental_Anisotropies, fluorophores, indexes, timepoints, experiments = preprocess()
 
def f(args):
    fluo, i, b = args
    Am1, Am2, Ad1, Ad2 = Experimental_Anisotropies[fluo+'_m_max'], Experimental_Anisotropies[fluo+'_m_min'], Experimental_Anisotropies[fluo+'_d_max'], Experimental_Anisotropies[fluo+'_d_min']
    b_min, b_max = b, b
    
    Sol = tf.Fit_Global_r(data[i:i+1], Am1, Am2, Ad1, Ad2, b_min, b_max, fluo, minimal=0.001, n_max=25, Plot=False)
    
    this_Am, this_Ad, this_b, this_m = Sol['Am'], Sol['Ad'], Sol['b'], Sol['m']
    
    start_timepoints = np.where(this_m>0)[0][0]
    end_timepoints = np.where(this_m<1)[0][-1]
    
    start_timepoints -= 5
    end_timepoints += 5
    
    if start_timepoints<0:
        start_timepoints = 0
    if end_timepoints>89:
        end_timepoints = 89
    
    Sol = cf.fit_caspase(this_m[start_timepoints:end_timepoints], data[fluo+'_x0'][i]-start_timepoints*timepoints)
    
    this_t_0, this_rate, this_k = Sol['t0'], Sol['rate'], Sol['k']
    this_casp, this_sens, this_prod = Sol['casp'], Sol['sens'], Sol['prod']
    
    this_t_0 += timepoints * start_timepoints
    this_casp = np.concatenate((np.zeros(start_timepoints*timepoints), this_casp, np.ones((90-end_timepoints)*timepoints)))
    this_sens = np.concatenate((np.zeros(start_timepoints*timepoints), this_sens, np.ones((90-end_timepoints)*timepoints)))
    this_prod = np.concatenate((np.zeros(start_timepoints*timepoints), this_prod, np.ones((90-end_timepoints)*timepoints)))         

    return (fluo, i), (this_Am, this_Ad, this_b, this_m,
            this_t_0, this_rate, this_k, this_casp, this_sens, this_prod)

                        
def myiter(experiment, fluo, ndxs, b):
    for ndx in ndxs:
        if ndx in indexes[fluo] and experiment==data.Content_YFP[ndx]:
            yield fluo, ndx, b

if __name__ == '__main__':

    
    output = dict()
    bs = np.arange(0.8, 1.15, 0.1)
    
    start = time.time()
    print(start)

    with Pool(7) as p:
        
        for b in bs:
            for experiment in experiments['YFP']:
                for fluo in fluorophores:        
                    for key, output_value in p.imap_unordered(f, myiter(experiment, fluo, data.index, b)):
                        output[key] = output_value
            
    
    print(time.time() - start)
    PREFIX = '2017-01-17'
    for b in bs:
        for experiment in experiments['YFP']:
            Analyzed = pd.DataFrame(index=data.index)
            for fluo in fluorophores:            
                vals = {}
                vals[fluo+'_Am'] = []
                vals[fluo+'_Ad'] = []
                vals[fluo+'_b'] = []
                vals[fluo+'_m'] = []
                
                vals[fluo+'_t0'] = []
                vals[fluo+'_rate'] = []
                vals[fluo+'_k'] = []
                vals[fluo+'_casp'] = []
                vals[fluo+'_sens'] = []
                vals[fluo+'_prod'] = []
                
                for i in data.index:
                    c = output.get((fluo, i), [np.nan] * 10)
                    vals[fluo+'_Am'].append(c[0])
                    vals[fluo+'_Ad'].append(c[1])
                    vals[fluo+'_b'].append(c[2])
                    vals[fluo+'_m'].append(c[3])
                    
                    vals[fluo+'_t0'].append(c[4])
                    vals[fluo+'_rate'].append(c[5])
                    vals[fluo+'_k'].append(c[6])
                    vals[fluo+'_casp'].append(c[7])
                    vals[fluo+'_sens'].append(c[8])
                    vals[fluo+'_prod'].append(c[9])
                
                Analyzed[fluo+'_Am'] = vals[fluo+'_Am']
                Analyzed[fluo+'_Ad'] = vals[fluo+'_Ad']
                Analyzed[fluo+'_b'] = vals[fluo+'_b']
                Analyzed[fluo+'_m'] = vals[fluo+'_m']
                
                Analyzed[fluo+'_t0'] = vals[fluo+'_t0']
                Analyzed[fluo+'_rate'] = vals[fluo+'_rate']
                Analyzed[fluo+'_k'] = vals[fluo+'_k']
                Analyzed[fluo+'_casp'] = vals[fluo+'_casp']
                Analyzed[fluo+'_sens'] = vals[fluo+'_sens']
                Analyzed[fluo+'_prod'] = vals[fluo+'_prod']
                    
            Analyzed.to_pickle(PREFIX + '_OneCasp_'+experiment+'_b_'+str(int(b*100))+'.pandas')