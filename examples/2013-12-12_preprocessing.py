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

working_path = pathlib.Path(r'C:\Users\Agus\Documents\Laboratorio\Imaging three sensors\2017-09-04_Images')
data_path = working_path.joinpath('noErode.pandas')

df = pd.read_pickle(str(data_path))
df = df.reset_index()
df = fd.r_from_i_to_df(df)

#%% Make window fit

df = fd.general_fit(df)

#%% Save window fit
save_path = working_path.joinpath('2017-10-16_window_fit_noErode.pandas')
df.to_pickle(str(save_path))

#%% first filter for data

df = fd.first_filter(df)

#%% save filtered data

save_path = working_path.joinpath('2017-10-16_first_filter_noErode.pandas')
df.to_pickle(str(save_path))

#%% Choose best popts

df = fd.second_filter(df)
df = fd.set_popts(df)

#%% save best popts data

save_path = working_path.joinpath('2017-10-16_best_popts_noErode.pandas')
df.to_pickle(str(save_path))
