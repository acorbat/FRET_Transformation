# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 10:09:18 2017

@author: Admin
"""

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

import Anisotropy_Functions as af
import Transformation as tf
import Caspase_Fit as cf

from scipy.stats import ttest_rel

#%% Load data

data = pd.read_pickle('OneCasp_Fitted.pandas')

#%% Define useful variables

timepoints = 10 #min
fluorophores = ['TFP','mKate','YFP'] #[Cas3,Cas8,Cas9]
Colors = {'TFP' : 'g', 'YFP' : 'y', 'mKate' : 'r'}
sigmoid_parameters = ['base', 'amplitude','rate', 'x0']
fit_parameters = ['Am', 'Ad', 'b', 'm', 't0', 'rate', 'k', 'casp', 'sens', 'prod']

time_coarse = np.arange(0, 90*timepoints, timepoints)
time_fine = np.arange(0, 90*timepoints)

#%% Define useful functions

def aprox_percent(Curve, p=0.1):
    return np.where(Curve>=p)[0][0]

def r_fromFit(fluo, i):
    Am = data[fluo+'_Am'][i]
    Ad = data[fluo+'_Ad'][i]
    b = data[fluo+'_b'][i]
    m = data[fluo+'_m'][i]
    return af.Anisotropy_FromFit(m, Am, Ad, b)

def aprox_r_percent(m, p=0.1):
    popt, _ = curve_fit(cf.sigmoid, time_coarse, m, p0=[0, 1, 1, np.where(m>0.5)[0][0]])
    this_curve = cf.sigmoid(time_fine, *popt)
    return aprox_percent(this_curve, p=p)

def plot_results(i):
    for fluo in fluorophores:
        if np.isfinite(data[fluo+'_t0'][i]):
            plt.scatter(time_coarse[:50], cf.Normalize(data['r_'+fluo][i]), c=Colors[fluo])
            plt.plot(time_fine, data[fluo+'_casp'][i], Colors[fluo])
            plt.plot(time_fine, data[fluo+'_prod'][i], Colors[fluo]+'--')
    plt.title(data.Content_YFP[i]+' '+str(i))
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

def v_(desc):
    return data[desc].values[np.isfinite(data[desc].values)]

#%% reFilter data

for i in data.index:
    if any(np.isfinite([data[fluo+'_t0'][i] for fluo in fluorophores])):
        for fluo in fluorophores:
            plt.scatter(time_coarse[:50], data['r_'+fluo][i], c=Colors[fluo])
        plt.show()
        
        plot_results(i)
        
        good_fit = ask_question(question='was this a good fit?')
        
        if not good_fit:
            for fluo in fluorophores:
                for parameter in fit_parameters:
                    data[fluo+'_'+parameter][i] = np.nan

#%% Find 10 percent activation and add columns

for fluo in fluorophores:
    t_10s = []
    for i in data.index:
        if np.isfinite(data[fluo+'_t0'][i]):
            this_t_10 = aprox_percent(data[fluo+'_casp'][i])
            t_10s.append(this_t_10)
        else:
            t_10s.append(np.nan)
        
    data[fluo+'_t_10'] = t_10s

#%% Find 10 percent activation in anisotropy and add columns

for fluo in fluorophores:
    r_t_10s = []
    for i in data.index:
        if np.isfinite(data[fluo+'_t0'][i]):
            r_curve = cf.Normalize(r_fromFit(fluo, i))
            this_r_t_10 = aprox_r_percent(r_curve)
            r_t_10s.append(this_r_t_10)
        else:
            r_t_10s.append(np.nan)
        
    data[fluo+'_r_t_10'] = r_t_10s

#%% Calculate differences in caspase timing

Differences_tags = ['YFP_to_TFP', 'YFP_to_mKate', 'TFP_to_mKate']

Differences = {}
Differences['YFP_to_TFP'] = data.TFP_t_10.values-data.YFP_t_10.values
Differences['YFP_to_mKate'] = data.mKate_t_10.values-data.YFP_t_10.values
Differences['TFP_to_mKate'] = data.mKate_t_10.values-data.TFP_t_10.values

for Differences_tag in Differences_tags:
    data[Differences_tag] = Differences[Differences_tag]

#%% Calculate differences in anisotropy timing

r_Differences_tags = ['r_YFP_to_TFP', 'r_YFP_to_mKate', 'r_TFP_to_mKate']

Differences = {}
Differences['r_YFP_to_TFP'] = data.TFP_r_t_10.values-data.YFP_r_t_10.values
Differences['r_YFP_to_mKate'] = data.mKate_r_t_10.values-data.YFP_r_t_10.values
Differences['r_TFP_to_mKate'] = data.mKate_r_t_10.values-data.TFP_r_t_10.values

for r_Differences_tag in r_Differences_tags:
    data[r_Differences_tag] = Differences[r_Differences_tag]


#%% Plot histograms and circular histogram of differences

for Differences_tag, r_Differences_tag in zip(Differences_tags, r_Differences_tags):
    this_difference = data[Differences_tag].values[np.isfinite(data[Differences_tag].values)]
    this_r_difference = data[r_Differences_tag].values[np.isfinite(data[r_Differences_tag].values)]
    mean = np.mean(this_difference)
    std = np.std(this_difference)
    r_mean = np.mean(this_r_difference)
    r_std = np.std(this_r_difference)
    plt.hist(this_difference, bins=20, alpha=0.5)
    plt.hist(this_r_difference, bins=20, color='r', alpha=0.5)
    plt.title(Differences_tag)
    plt.show()
    print('mean: {} +/- {}'.format(mean, std))
    print('r mean: {} +/- {}'.format(r_mean, r_std))

p = data.YFP_to_TFP.values + 1j* data.YFP_to_mKate.values
p = p[np.isfinite(p)]
pn = p / np.abs(p)
for i, this_pn in enumerate(pn):
    if np.isnan(this_pn):
        pn[i] = 0

frecs = np.histogram(np.angle(pn), bins = 20)

N = len(frecs[0])

theta = frecs[1][:-1] # np.linspace(0.0, 2 * np.pi, N, endpoint=False)
radii = frecs[0]
width = (2*np.pi) / N

#matplotlib.rcParams.update({'font.size': 18})

plt.figure(figsize=(10,10))
ax = plt.subplot(111, polar=True)
bars = ax.bar(theta, radii, width=width)

# Use custom colors and opacity
for r, bar in zip(radii, bars):
    bar.set_facecolor(plt.cm.jet(r / 10.))
    bar.set_alpha(0.8)

plt.show()

plt.polar(p, ls='', marker='o')
plt.show()

#%% Compare mean differences

for Differences_tag in Differences_tags:
    r_Differences_tag = 'r_'+Differences_tag
    statistic, p_value = ttest_rel(np.abs(v_(r_Differences_tag)), np.abs(v_(Differences_tag)))
    print('p-value of '+Differences_tag+' is '+str(p_value))
    if p_value>0.1:
        print('Cannot say that means are different')
    else:
        print('means are different with at least 90% confidence')
        this_mean = np.mean(np.abs(v_(r_Differences_tag))- np.abs(v_(Differences_tag)))
        if this_mean>0:
            print('Caspase fitting method is better for '+str(this_mean)+' minutes')
        else:
            if this_mean>0:
                print('Anisotropy fitting method is better for '+str(-this_mean)+' minutes')