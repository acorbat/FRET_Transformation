# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 12:22:11 2017

@author: Admin
"""
#%% Import packages to be used
import numpy as np
import pandas as pd
import itertools as it
import pathlib
import tifffile as tif

from scipy import ndimage
from skimage.morphology import erosion

#%% Functions to process images

def prepareBGimages(par, perp, parBG, perpBG, sigma=5):
    """
    smothes the BG images and the correction images and bg substracts the correction
    images
    
    
    
    Parameters
    ----------
    par, perp: parallel and perpendicular correction images (fluorescent dye)
    
    parBG, perpBG: parallel and perpendicular BG images (recorded in absence of dye)
    or integers for a constant background
    
    Returns
    -------
    
    par_s, perp_s: smoothed and bg corrected images of the fluorescent dye
    
    parBG_s, perpBG_s: smoothed BG images
    """
    par_s = par # ndimage.filters.gaussian_filter(par, 5, mode='nearest')
    perp_s = perp # ndimage.filters.gaussian_filter(perp, 5, mode='nearest')
    
    if isinstance(parBG, np.ndarray):
        parBG_s = ndimage.filters.gaussian_filter(parBG, 5, mode='nearest')
        perpBG_s = ndimage.filters.gaussian_filter(perpBG, 5, mode='nearest')
    
    else:
        parBG_s = parBG
        perpBG_s = perpBG
    
    par_s -= parBG_s
    perp_s -= perpBG_s
    

    return par_s, perp_s, parBG_s, perpBG_s
    
    
def shiftimage(im, shiftXY):
    """
    shift the perpendicular image with respect to the parallel image
    
    Parameters
    ----------
    
    im: image to shift
    
    shiftXY: x and y shift to perform
    
    Returns
    -------
    
    im_shifted: the shifted image
    """
    if shiftXY[0] !=0 or shiftXY[1] !=0:
                im = ndimage.shift(im, shiftXY)
                
    return im
    
    
def findBG(im1, im2 ,roi=(35,35)):
    """find the region with the lowest intensity on the image using fourier - 
    transformation and return the average of the region for the two images
    
    Parameters
    ----------
    
    im1, im2: images (np - array)
    
    roi: x/y dimensions of the background ROI
    
    Results
    -------
    bg1, bg2: backgoud values for the two images
    
    """
    
    
    
    with np.errstate(invalid='ignore'):
        im_max = np.maximum(im1, im2)
    nanindex = np.where(np.isnan(im_max))
    im_max[nanindex] = np.nanmax(im_max)
    
    kernel = np.ones((roi[0], roi[1]))
    score = np.fft.ifft2(np.fft.fft2(kernel, im_max.shape) * np.fft.fft2(im_max))
    score = np.roll(np.roll(score, -int(kernel.shape[0]/2), 0),
                                    -int(kernel.shape[1]/2), 1).real
    score =  score[roi[0]:(im_max.shape[0]-roi[0]),
                roi[1]:(im_max.shape[1] - roi[1])]   
    
    #score[np.where(np.isnan(score))] = score[np.where(np.isfinite(score))].max()                   
    cenx, ceny = np.unravel_index(score.argmin(), score.shape) 
    
    cenx += roi[0]
    ceny += roi[1]

    bg1 = np.nanmean(im1[(cenx-roi[0]//2):(cenx+roi[0]//2), 
            (ceny-roi[1]//2):(ceny+roi[1]//2)])
    bg2 = np.nanmean(im2[(cenx-roi[0]//2):(cenx+roi[0]//2), 
            (ceny-roi[1]//2):(ceny+roi[1]//2)])
        
    return bg1, bg2
    
    
def applyGfactor(par, perp, Gfactor=1.0):
    """
    correct parallel and perpendicular image with G-factor
    
    par, perp: parallel and perpendicular fluorescence images (apply shift before)
    
    Gfactor: correction for the instrument effects, can be either of
            number: G = (I_perp / I_par)
            tuple of images: (par_correction, perp_correction)
            
    Returns
    --------
    par_c, perp_c: corrected 
    
    """
    
    
        
    if isinstance(Gfactor, tuple):
        
        if isinstance(Gfactor[0], np.ndarray) and par.shape == Gfactor[0].shape:
            gmax = max(Gfactor[0].max(), Gfactor[1].max())
            par /= Gfactor[0]
            par *= gmax
            perp /= Gfactor[1]
            perp *= gmax
            
        elif isinstance(Gfactor[0], np.ndarray) and par.shape != Gfactor[0].shape: 
            par /= Gfactor[0].mean()
            perp /= Gfactor[1].mean()

            
    elif isinstance(Gfactor, (int, float)):        
        perp /= Gfactor
    
    else:
        print("shit")
    
    return par, perp
    
    
def threshimages(par, perp, thresh):
    """
    set values below the threshold to nan
    
    Parameters
    ---------
    
    par, perp: images for parallel and perpendicular channel
    
    thresh: minimum pixel intensity to keep
    
    Returns
    -------
    
    par, perp: images with low pixels set to nan
    """
    
    with np.errstate(invalid='ignore'):
        min_projection = np.minimum(par,perp)
        nan_pixel = np.where(min_projection < thresh)
    
    par[nan_pixel] = np.nan
    perp[nan_pixel] = np.nan
    
    return par, perp
    
    
    
    
def calculate_anisotropy(par, perp, Gfactor=1, shiftXY=(0,0), bg=None, thresh=50):
    """
    calculate the anisotropy and fluorescent image from parallel 
    and perpendicular images
    
    Parameters
    ----------
    
    par, perp: parallel and perpendicular fluorescence images (apply shift before)
    
    Gfactor: correction for the instrument effects, can be either of
            number: G = (I_perp / I_par)
            tuple of images: (par_correction, perp_correction)
            
            
    bg: background (tuple of background images, tuple of numbers or single number)
    
    Returns
    -------
    
    fluorescence: image of fluorescence (parallel + 2 * perpendicular)
    
    anisotropy: image of anisotropy (parallel - perpendicuar / fluorescence)
    
    background: tuple of backgroundvalues for parallel and perpendicular images
        
    """
    par = par.astype(np.float32) 
    perp = perp.astype(np.float32)
    
    # remove overexposed areas
    par[np.where(par > 4080)] = np.nan
    perp[np.where(perp > 4080)] = np.nan
    
    # threshold the images #should it be before or after corrections to perp)
    par, perp = threshimages(par, perp, 100)
    
    # remove background
    if isinstance(bg, tuple):
        par, perp, newbg_par, newbg_per = prepareBGimages(par, perp, bg[0], bg[1])
        #par -= bg[0]
        #perp -= bg[1]
        
    elif isinstance(bg, (int, float)):
        par -= bg
        perp -= bg
    
    
    # correct for the G-factor    
    par, perp = applyGfactor(par, perp, Gfactor)
    
    
    # correct for additional background effects (eg. medium fluorescence)    
    bg_par, bg_perp = findBG(par, perp) # RuntimeWarning
    par -= bg_par
    perp -= bg_perp
    
    
    # threshold the images #should it be before or after corrections to perp)
    par, perp = threshimages(par, perp, thresh)
    
    
    return par, perp


def my_shift(img, shift_x=0, shift_y=0):
    shape = img.shape
    shifted = np.zeros(shape)
    shifted.fill(0)
    
    dest_y_min = np.clip(shift_y, 0, shape[0])
    dest_y_max = np.clip(shape[0]+shift_y, 0, shape[0])
    dest_x_min = np.clip(shift_x, 0, shape[1])
    dest_x_max = np.clip(shape[1]+shift_x, 0, shape[1])
    
    orig_y_min = np.clip(-shift_y, 0, shape[0])
    orig_y_max = np.clip(shape[0]-shift_y, 0, shape[0])
    orig_x_min = np.clip(-shift_x, 0, shape[1])
    orig_x_max = np.clip(shape[1]-shift_x, 0, shape[1])
    shifted[dest_y_min:dest_y_max, dest_x_min:dest_x_max] = img[orig_y_min:orig_y_max, orig_x_min:orig_x_max]
    
    return shifted


def prepare_mask(par, per, mask, shiftXY=(0, 0)):
    mask[np.isnan(par)] = 0
    shiftXY = (round(-shiftXY[0]), round(-shiftXY[1]))
    mask = mask.astype(np.float32)
    mask_per = shiftimage(mask, shiftXY=shiftXY)
    mask_per[np.isnan(per)] = 0
    mask_par = shiftimage(mask_per, shiftXY=(-shiftXY[0], -shiftXY[1]))
    return mask_par.astype(int), mask_per.astype(int)


def extract_attributes(img, mask, suffix='img'):
    df = []
    for num in range(1, mask.max()+1):
        img_crop = img.copy()
        img_crop[mask!=num] = np.nan
        this_ext = {}
        this_ext['position'] = 30
        this_ext['object'] = num
        if np.any(np.isfinite(img_crop)):
            this_ext[suffix+'_mean'] = np.nanmean(img_crop)
            this_ext[suffix+'_std'] = np.nanstd(img_crop)
            this_ext[suffix+'_p25'] = np.nanpercentile(img_crop, 25)
            this_ext[suffix+'_p75'] = np.nanpercentile(img_crop, 75)
            this_ext[suffix+'_area'] = np.nansum(mask==num)
            this_ext[suffix+'_nanpixs'] = np.nansum(np.isnan(img[mask==num]))
        df.append(this_ext)
    df = pd.DataFrame(df)
    return df

def generate_df(par_df, per_df, r_df, f_df):
    all_df = pd.merge(par_df, per_df, how='outer', on=['position', 'object', 'timepoint'])
    all_df = all_df.merge(r_df, how='outer', on=['position', 'object', 'timepoint'])
    all_df = all_df.merge(f_df, how='outer', on=['position', 'object', 'timepoint'])
    return all_df


#%% Generate all i_par, i_per, anisotropy and fluorescence images

def process_images(Files, BG_Files, G_Files, masks_folder, erode=None, fast=False):
    df = pd.DataFrame()
    for fluo in fluorophores:
        ser_par = np.asarray(tif.imread(str(Files[fluo, 'par'])), dtype=float)
        ser_per = np.asarray(tif.imread(str(Files[fluo, 'per'])), dtype=float)
        
        BG_par = np.asarray(tif.imread(str(BG_Files[fluo, 'par'])), dtype=float)
        BG_per = np.asarray(tif.imread(str(BG_Files[fluo, 'per'])), dtype=float)
        
        G_par = np.asarray(tif.imread(str(G_Files[fluo, 'par'])), dtype=float)
        G_per = np.asarray(tif.imread(str(G_Files[fluo, 'per'])), dtype=float)
        
        all_df = []
        for t, (par, per) in enumerate(zip(ser_par, ser_per)):
            par, per =  calculate_anisotropy(par, per, Gfactor=(G_par, G_per), shiftXY=(11.7,-0.6), bg=(BG_par, BG_per))
            print('Analyzing timepoint %d of %d' % (t, len(ser_par)))
            mask = tif.imread(str(Mask_Files[t]))
            if erode is not None:
                for j in range(erode):
                    mask = erosion(mask)
            if fast:
                for num in range(1, mask.max()+1):
                    if num not in [10, 11, 30, 32]:
                        mask[mask==num] = 0
            mask_par, mask_per = prepare_mask(par, per, mask, shiftXY=(11.7,-0.6))
            par_df = extract_attributes(par, mask_par, suffix=fluo+'_par')
            per_df = extract_attributes(per, mask_per, suffix=fluo+'_per')
            
            par_df['timepoint'] = t
            per_df['timepoint'] = t
            this_all_df = pd.merge(par_df, per_df, how='outer', on=['position', 'object', 'timepoint'])
            all_df.append(this_all_df)
        all_df = pd.concat(all_df, ignore_index=True)
        try:
            df = df.merge(all_df, how='outer', on=['position', 'object', 'timepoint'])
        except KeyError:
            df = all_df
    return df

#%% group timepoints extracted
def group_cell(df, xbase='timepoint', groupby=['position', 'object']):
    cols = [col for col in df.columns if col not in groupby]
    
    tpmax = df[xbase].max()
    header=["position", "object"]
    
    for col in cols:
        header.append(col)
    reslist = []
    grouped = df.groupby(["position", "object"])
    
    for key, obj in grouped:
        resline = [key[0], key[1]]
        for col in cols:
            # create time profile of the corresponding column
            t_profile = obj[col].values 
            t_base = obj[xbase].values
            t = np.zeros((tpmax+1))
            t *= np.nan
            t[t_base] = t_profile
            resline.append(t)  
            
        reslist.append(resline)
        
    df_out = pd.DataFrame.from_records(reslist, columns=header)   
    return df_out


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

noErode_df = process_images(Files, BG_Files, G_Files, masks_folder, fast=False)

noErode_df = group_cell(noErode_df)

noErode_df.to_pickle(r'D:\Agus\Imaging three sensors\aniso_para_agustin\20131212_pos30\pos30_newnoErode_df.pandas')

erode = 5
Erode_df = process_images(Files, BG_Files, G_Files, masks_folder, erode=erode, fast=False)

Erode_df = group_cell(Erode_df)

Erode_df.to_pickle(r'D:\Agus\Imaging three sensors\aniso_para_agustin\20131212_pos30\pos30_newErode_'+str(erode)+'_df.pandas')