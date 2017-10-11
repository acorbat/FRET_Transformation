# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 17:47:48 2017

@author: Admin
"""
import itertools as it
import pathlib
import os

os.chdir(r'D:\Agus\Imaging three sensors\FRET_Transformation')
import image_process as ip

#%% Load image

fluorophores = ['YFP', 'mKate', 'TFP']
orientations = ['par', 'per']

general_folder = pathlib.Path(r'D:\Agus\Imaging three sensors\aniso_para_agustin\20131212_pos30')
corrections_folder = general_folder.joinpath('Correction_20131212')
masks_folder = general_folder.joinpath('masks')

Files = {a: general_folder.joinpath('pos30_'+'_'.join(a)+'.TIF') for a in it.product(fluorophores, orientations)}
BG_Files = {a: corrections_folder.joinpath('1'+'_'.join(a)+'_BG.tif') for a in it.product(fluorophores, orientations)}
G_Files = {a: corrections_folder.joinpath('1'+'_'.join(a)+'.tif') for a in it.product(fluorophores, orientations)}
Mask_Files = {t: masks_folder.joinpath('o_30_'+str(t)+'.tiff') for t in range(0,90)}


#%% Execute specific cases

noErode_df = ip.process_images(Files, BG_Files, G_Files, Mask_Files, fast=False)

noErode_df.to_pickle(r'D:\Agus\Imaging three sensors\aniso_para_agustin\20131212_pos30\test_noErode_df.pandas')

erode = 5
Erode_df = ip.process_images(Files, BG_Files, G_Files, masks_folder, erode=erode, fast=False)

Erode_df.to_pickle(r'D:\Agus\Imaging three sensors\aniso_para_agustin\20131212_pos30\test_Erode_'+str(erode)+'_df.pandas')