# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 12:22:11 2017

@author: Admin
"""
#%% Import packages to be used
import numpy as np
import pandas as pd
import tifffile as tif

from scipy import ndimage
from skimage.morphology import erosion

#%% Functions to process images

def prepareBGimages(par, perp, parBG, perpBG, sigma=5):
    """
    smothes the BG images and the correction images and bg substracts the correction
    images.
    
    
    
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
    
    
    
    
def correct_images(par, perp, Gfactor=1, shiftXY=(0,0), bg=None, thresh=50):
    """
    Applies background correction, under and overexposed correction, G factor correction
    and shift correction to par and perp images.
    
    Parameters
    ----------
    
    par, perp: parallel and perpendicular fluorescence images (apply shift before)
    
    Gfactor: correction for the instrument effects, can be either of
            number: G = (I_perp / I_par)
            tuple of images: (par_correction, perp_correction)
            
            
    bg: background (tuple of background images, tuple of numbers or single number)
    
    Returns
    -------
    
    par: corrected parallel image
    
    perp: corrected perpendicular image
        
    """
    par = par.astype(np.float32) 
    perp = perp.astype(np.float32)
    
    
    # shift the perpendicular image to correct for the polarizer missmatch
    perp = shiftimage(perp, shiftXY)
    
    
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
    """
    Own coarse pixel shift function.
    """
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
    """
    Prepares both par and per masks to be used taking into consideration only pixels
    present in both images.
    
    First it takes out from the mask nan pixels of par image, then it shifts shiftXY
    to match perpendicular image and takes out nan pixels from mask. This mask corresponds 
    to the per mask, while its -shiftXY shift is the par mask. This is done in order not to
    consider pixels on the borders that can't be found on the other image.
    
    Parameters
    ----------
    par : array_like
        Array of the parallel intensity corrected image.
    per : array_like
        Array of the perpendicular intensity corrected image.
    mask : array_like
        Array of the labeled mask.
    shiftXY : tuple, optional
        tuple containing the y and x shift to be applied to the mask.
    
    Returns
    -------
    mask_par : array_like, int
        Array of the mask to be applied to parallel images.
    mask_per : array_like, int
        Array of the mask to be applied to perpendicular images.
    """
    mask[np.isnan(par)] = 0
    shiftXY = (round(-shiftXY[0]), round(-shiftXY[1]))
    mask = mask.astype(np.float32)
    mask_per = shiftimage(mask, shiftXY=shiftXY)
    mask_per[np.isnan(per)] = 0
    mask_par = shiftimage(mask_per, shiftXY=(-shiftXY[0], -shiftXY[1]))
    return mask_par.astype(int), mask_per.astype(int)


def extract_attributes(img, mask, suffix='img'):
    """
    Generates a dataframe containing mean, std, 25 and 75 percentile, area and nan area
    of each label in an image. label is saved to object column and position is 
    hard coded to 30.
    
    Parameters
    ----------
    img : Array-like
        image from which to extract features.
    mask : Array-like
        labeled mask to be applied.
    suffix : string, optional
        suffix to be used for the column names. Default is img.
    
    Returns
    -------
    df : pandas dataframe
        contains the information for each object in the image.
    """
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
    """
    Merges all par, per, r and f dataframes on position, object and timepoint.
    """
    all_df = pd.merge(par_df, per_df, how='outer', on=['position', 'object', 'timepoint'])
    all_df = all_df.merge(r_df, how='outer', on=['position', 'object', 'timepoint'])
    all_df = all_df.merge(f_df, how='outer', on=['position', 'object', 'timepoint'])
    return all_df


#%% group timepoints extracted
def group_cell(df, xbase='timepoint', groupby=['position', 'object']):
    """
    Takes the DataFrame that contains all information separately and groups it so
    as to have time series of each feature.
    
    Parameters
    ----------
    df :  Pandas DataFrame
        Dataframe containing all the information of the different objects and 
        timepoints.
    xbase : string, optional
        Name of the column that orders the time series. Defaults to timepoint.
    groupby : list of strings
        list of strings along which objects must be grouped by. Defaults to 
        ['position', 'object'].
    """
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


#%% Generate all i_par, i_per, anisotropy and fluorescence images

def process_images(Files, BG_Files, G_Files, Mask_Files, erode=None, fast=False):
    """
    Opens all files and manages the processing of all images in the series.
    
    It opens the series of images and for each timepoint it applies the correction, 
    then the feature extraction and finally merges and concatenates all dataframes into 
    a single one.
    
    Parameters
    ----------
    Files : dictionary
        Dictionary containing path to time series of images where keys 
        are [fluo, orientation].
    BG_Files : dictionary
        Dictionary containing path to background correction images where keys 
        are [fluo, orientation].
    G_Files : dictionary
        Dictionary containing path to G factor correction images where keys 
        are [fluo, orientation].
    Mask_Files : dictionary
        Dictionary containing path to G factor correction images where keys 
        are timepoints.
    erode : int, optional
        number of erosion iterations to be applied to mask. Default is None.
    fast : boolean, optional
        if set to true only 10, 11, 30 and 32 objects are analyzed. Defaults
        to False.
    
    Returns
    -------
    df : Pandas DataFrame
        Dataframe containing the information of each object for each timepoint
        separately.
    """
    df = pd.DataFrame()
    fluorophores = set([key[0] for key in Files.keys()])
    for fluo in fluorophores:
        print(fluo)
        ser_par = np.asarray(tif.imread(str(Files[fluo, 'par'])), dtype=float)
        ser_per = np.asarray(tif.imread(str(Files[fluo, 'per'])), dtype=float)
        
        BG_par = np.asarray(tif.imread(str(BG_Files[fluo, 'par'])), dtype=float)
        BG_per = np.asarray(tif.imread(str(BG_Files[fluo, 'per'])), dtype=float)
        
        G_par = np.asarray(tif.imread(str(G_Files[fluo, 'par'])), dtype=float)
        G_per = np.asarray(tif.imread(str(G_Files[fluo, 'per'])), dtype=float)
        
        all_df = []
        for t, (par, per) in enumerate(zip(ser_par, ser_per)):
            par, per =  correct_images(par, per, Gfactor=(G_par, G_per), shiftXY=(11.7,-0.6), bg=(BG_par, BG_per))
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

    df = group_cell(df)
    return df