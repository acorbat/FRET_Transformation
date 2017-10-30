# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 12:32:47 2017

@author: Agus
"""
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def add_differences(df, fluorophores=['YFP','mKate','TFP'], time_col='max_activity', Difference_tags=None):
    """
    Generates a Differences tags list (if not given) from the fluorophores list
    and then calculates the difference between the time columns of each of the 
    fluorophores and adds it to the given DataFrame.
    
    Parameters
    ----------
    df : Pandas DataFrame
        DataFrame containing the data and columns.
    fluorophores : list, optional
        list of fluorophores from which it generates the differences. Default 
        is ['YFP','mKate','TFP']
    time_col : string, optional
        suffix of the columns from which to calculate differences. Default is 
        "max_activity".
    Difference_tags : list, optional
        list of Difference tags used to calculate differences. Default is None
        so the function generates its own from fluorophores list.
    
    Returns
    -------
    df : DataFrame
        Modified DataFrame with the added data in the Differences tag column.
    """
    # First we generate a Difference tags list from the existing fluorophores
    if Difference_tags is None:
        Difference_tags = []
        for fluo1, fluo2 in itertools.combinations(fluorophores, 2):
            Difference_tags.append('_to_'.join([fluo1, fluo2]))
    
    # Generate the differences list and add it to DataFrame
    for tag in Difference_tags:
        fluo1, fluo2 = tag.split('_to_')
        
        times1 = df['_'.join([fluo1, time_col])].values
        times2 = df['_'.join([fluo2, time_col])].values
        
        df[tag] = times2 - times1
    
    return df


def plot_polarhist(x, y, plot_scatter=False):
    """ Plots polar histogram of x vs y. If plot_scatter is True, Default is 
    False, then it shows histogram and prepares polar scatter plot"""
    p = x + 1j* y
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
    
    if plot_scatter:
        plt.show()
        
        plt.polar(p, ls='', marker='o')