# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 10:24:42 2017

@author: Agus
"""
import pathlib
import os
import pandas as pd
import matplotlib.pyplot as plt

this_dir = os.getcwd()
os.chdir(r'C:\Users\Agus\Documents\Laboratorio\Imaging three sensors\FRET_Transformation')
import filter_data as fd
os.chdir(this_dir)

#%% load data

working_path = pathlib.Path(r'C:\Users\Agus\Documents\Laboratorio\Imaging three sensors\old_data')
data_path = working_path.joinpath('2013-12-12.pandas')

df = pd.read_pickle(str(data_path))

#%% Make window fit

df = fd.general_fit(df, y_col='r')

#%% Save window fit
save_path = working_path.joinpath('2017-10-03_window_fit_2013-12-12.pandas')
df.to_pickle(str(save_path))

#%% first filter for data

df = fd.first_filter(df, col_to_filter='r')

#%% save filtered data

save_path = working_path.joinpath('2017-10-03_first_filter_2013-12-12.pandas')
df.to_pickle(str(save_path))

#%% Choose best popts

df = fd.second_filter(df, col_to_filter='r')