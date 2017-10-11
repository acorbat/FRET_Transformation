# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 12:32:47 2017

@author: Agus
"""
import itertools
import pandas as pd

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