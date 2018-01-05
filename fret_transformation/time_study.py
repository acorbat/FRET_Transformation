# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 12:32:47 2017

@author: Agus
"""
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pyDOE import lhs
from scipy.interpolate import splrep, splev
from scipy.misc import derivative

from fret_transformation.transformation import get_max_ind
from fret_transformation import anisotropy_functions as af
from fret_transformation import caspase_model as cm

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


def generate_param_sweep(N, space_params = None):
    """
    Generates a random sample of size N of the parameters specified in the
    dictionary space_params using a latin hypercube where keys should be the
    exact name of the parameters and it's associated value a tuple with minimum
    and maximum value that can be taken.

    Parameters
    ----------
    N : int
        Number of samples to generate.
    space_params : dictionary, optional
        Dictionary where key is parameter name and value is a tuple with minimum
        and maximum value that can be taken by the parameter. Defualt is a
        variation {'S3': (1E4, 1E6),
                   'S8': (1E4, 1E6),
                   'S9': (1E4, 1E6)}.
    Returns
    -------
    param_df : pandas DataFrame
        DataFrame containing the list of samples generated where each column is
        a specific parameter.
    """
    if space_params is None:
        space_params = {'S3': (1E4, 1E6),
                        'S8': (1E4, 1E6),
                        'S9': (1E4, 1E6)}

    dim = len(space_params)
    param_percents = lhs(dim, samples=N)
    param_df = pd.DataFrame(columns=list(space_params.keys()))

    for percs in param_percents:
        vals = {}
        for perc, col in zip(percs, param_df.columns):
            minval, maxval = space_params[col]
            val = minval + (maxval - minval) * perc

            vals[col] = val
        vals = pd.DataFrame([vals])
        param_df = param_df.append(vals, ignore_index=True)

    return param_df


def timedif_from_params(params, Differences_tags, fluorophores=['YFP', 'mKate', 'TFP'], pp=None, Colors={'YFP': 'y', 'mKate': 'r', 'TFP': 'g'}):
    """
    Runs a simulation with the given set of parameters and calculates the
    corresponding time difference using the Differences_tags. If it receives a
    pp PdfPages object, it saves the simulation and fit to a pdf.

    Simulation is done with 10 minutes resolution as well as the fit in search
    of sigmoid parameters that are latter used to find the complex by
    derivation. If simulation can't be performed, nan are returned for each
    difference.

    Parameters
    ----------
    params : lmfit.Parameters object
        Parameters to be given to the simulation of caspase model.
    Differences_tags : list of strings
        list of strings determining the differences to be calculated.
    fluorophores :  list of strings, optional
        list of fluorophores to be analyzed. Default is ['YFP', 'mKate', 'TFP'].
    pp : PdfPages object
        PdfPages file where images should be saved.

    Returns
    -------
    difs : dictionary
        dictionary containing the time difference between fluorophores
        calculated using the Differences_tags list.
    """
    t = np.arange(0, 72000, 600)
    earm_params = cm.params.copy()
    for casp in ['S3', 'S8', 'S9']:
        earm_params[casp].set(value=params[casp].value)

    earm_sim = cm.simulate(t, cm.params)
    sim = cm.simulate(t, params)

    earm_sim = sim_to_ani(earm_sim)
    sim = sim_to_ani(sim)

    sim_aborted = {}
    for fluo in fluorophores:
        r = sim[fluo+'_r_from_i'].values[0]
        t = np.arange(0, len(r)*10, 10)
        sim_aborted[fluo] = r[0]==r[-1]

    if not any(sim_aborted.values()):
        sim = find_complex_in_sim(sim)

        if pp is not None:
            for fluo in fluorophores:
                plt.plot(sim.t.values[0], sim[fluo+'_r_from_i'].values[0], 'x'+Colors[fluo])
            pp.savefig()
            plt.close()

            note = ''
            for fluo in fluorophores:
                if len(sim[fluo+'_r_complex'].values[0]) == len(sim.t.values[0]):
                    plt.plot(sim.t.values[0], sim[fluo+'_r_complex'].values[0], 'x'+Colors[fluo])
                plt.scatter(sim[fluo+'_max_activity'].values, [0]*len(sim[fluo+'_max_activity'].values), color=Colors[fluo])
                note = note + str(sim[fluo+'_max_activity'][0]) + '\n'
            pp.attach_note(note, positionRect=[100, 100, 100, 100])
            pp.savefig()
            plt.close()

            for fluo in fluorophores:
                plt.plot(earm_sim.t.values[0], sim[fluo+'_r_from_i'].values[0]/earm_sim[fluo+'_r_from_i'].values[0], 'x'+Colors[fluo])
            pp.savefig()
            plt.close()

        sim = add_differences(sim, Difference_tags=Differences_tags)
        difs = {tag: sim[tag].values for tag in Differences_tags}
    else:
        difs = {tag: np.nan for tag in Differences_tags}

    return difs


def sim_to_ani(df, col='r_from_i'):
    """Takes a DataFrame from simulation and add a column "col" for each
    fluorophore with the corresponding anisotropy."""
    fluo_to_cas = {'YFP': 'SC9',
                   'mKate': 'SC8',
                   'TFP': 'SC3'}
    fluo_to_ani = {'YFP': (.22, .3),
                   'mKate': (.23, .28),
                   'TFP': (.28, .34)}

    for fluo in fluo_to_cas.keys():
        sens_norm = [sens/np.nanmax(sens)
                     for sens in df[fluo_to_cas[fluo]].values]
        anis = [af.Anisotropy_FromFit(m,
                                      fluo_to_ani[fluo][1],
                                      fluo_to_ani[fluo][0],
                                      1)
                for m in sens_norm]

        df['_'.join([fluo, col])] = anis

    return df


def add_times_from_sim(param_df, Differences_tags, pp=None, params=None):
    """Takes a DataFrame with parameters to variate and runs the simulation for
    each set of parameters, adding columns with the Differences_tags specified.
    Plots of the fits and derivations can be saved in pdf to PdfPages object as
    pp."""
    if params is None:
        params = cm.params

    var_cols = param_df.columns

    for tag in Differences_tags:
        param_df[tag] = np.nan

    for i in param_df.index:
        print(i)
        for col in var_cols:
            params[col].set(value=param_df[col][i])
        difs = timedif_from_params(params, Differences_tags, pp=pp)

        for tag in Differences_tags:
            param_df = param_df.set_value(i, tag, difs[tag])

    return param_df


def find_complex_in_sim(df, col_to_der='r_from_i', order=5, timepoints=10, Plot=False):
    """
    Takes the whole dataframe and applies finite differences to data in order
    to find the derivative of the sigmoid region anisotropy data of filtered
    curves.

    This function doesn't select the sigmoid region of the data. Finite
    differences is used to find the first derivative, and filter noise. Spline
    interpolation is used to find maximum at the derived curve. Maximum cannot
    be found at the beginning and ending of curves (this is usually caused by
    interpolation) so these values are discarded.

    Parameters
    ----------
    df : Pandas DataFrame
        DataFrame containing the curves for each fluorophores, the best fit
        values, which are used to filter the curves as well.
    col_to_der : string, optional
        suffix of column to derive. Default is "r_from_i".
    order : optional, odd int
        Number of points to be used in the finite differences. Must be odd.
        Default is 5.
    timepoints : float or int
        Spacing between timepoints. Default is 10.
    Plot : boolean, optional
        True if plots are to be showed. Default is False.

    Returns
    -------
    df : Pandas DataFrame
        Updated Pandas DataFrame
    """
    fluorophores = [col for col in df.columns if col_to_der in col]
    fluorophores = [col.split('_')[0] for col in fluorophores]

    ders = {}
    maxs = {}
    for fluo in fluorophores:
        ders[fluo] = []
        maxs[fluo] = []

    for i in df.index:
        for fluo in fluorophores:
            r = df['_'.join([fluo, col_to_der])][i]
            if r[0]!=r[-1]:
                time = np.arange(0, len(r)*timepoints, timepoints)

                def this_vect(t):
                    ind = t//timepoints
                    ind = np.clip(ind, 0, len(r)-1)
                    return r[ind]

                r_der = [derivative(this_vect, inst, dx=timepoints, order=order) for inst in time]

                t = np.arange(0, len(r)*timepoints)
                f = splrep(time, r_der, k=3, s=0)
                der_interp = splev(t, f, der=0)

                max_act = get_max_ind(der_interp)

                if Plot:
                    fig, axs = plt.subplots(2,1, sharex=True, figsize=(10,12))
                    axs[0].plot(time, r, Colors[fluo]+'--', alpha=0.5)
                    axs[0].set_ylabel('fraction')

                    axs[1].plot(time, r_der)
                    axs[1].plot(t, der_interp, Colors[fluo])
                    axs[1].set_ylabel('complex')
                    axs[1].set_xlabel('time (min.)')

                    plt.suptitle('obj:'+str(i)+' exp:'+df.Content_YFP[i]+' max:'+str(max_act))
                    last_maxs = [fluo+' '+str(maxs[fluo][-1]) for fluo in fluorophores]
                    note = '\n'.join(last_maxs)
                    plt.show()
                    print(note)

                ders[fluo].append(r_der)
                maxs[fluo].append(max_act)


            else:
                ders[fluo].append([np.nan])
                maxs[fluo].append(np.nan)

    for fluo in fluorophores:
        df[fluo+'_r_complex'] = ders[fluo]
        df[fluo+'_max_activity'] = maxs[fluo]

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


def plot_scatter_times(x, y, marker='o', color=None, zoom=True):
    """Scatter plot of data in x,y colors according to quadrant by default."""
    if color is None:
        color = ['r' if this_x>=0 and this_y>=0
        else 'b' if this_x>=0 and this_y<0
        else 'g' if this_x<0 and this_y>=0
        else 'm'
        for this_x, this_y in zip(x, y)]

    plt.scatter(x, y, c=color, marker=marker, alpha=0.5)
    if zoom:
        plt.xlim((-35, 35))
        plt.ylim((-35, 35))
    plt.xlabel('TFP_to_YFP')
    plt.ylabel('TFP_to_mKate')
    ax = plt.gca()
    ax.grid(True)


def count_neighbours(center, dist, df, cols_xy):
    """
    Counts amount of instances of df cols_xy that are less than dist[i] in the i
    direction from the center.

    Takes the center and looks in df.cols_xy values which are between
    center +/- dist.

    Parameters
    ----------
    center : tuple
        x, y values from which to look neighbours.
    dist : tuple
        x, y distance from the center to define the region of interest
    df : pandas DataFrame
        DataFrame containing the data.
    cols_xy : tuple of strings
        Name of the columns of df where coordinates are saved.

    Returns
    -------
    counts : int
        Amount of neighbours to center in the rectangle of size 2*dist.
    """
    vects = []
    for i, col in enumerate(cols_xy):
        vect = [True if val > center[i] - dist[i] and \
                val < center[i] + dist[i] \
                else False \
                for val in df[col].values]
        vects.append(vect)

    counts = np.asarray(vects)
    counts = counts.all(axis=0)
    return np.sum(counts)


def add_counts(param_df, data, dist=(5,5), cols=('TFP_to_YFP', 'TFP_to_mKate')):
    """Calculate amount of neighbours in the area defined as the rectangle of
    size 2*"dist" and center "center" for each instance of times in "cols" in
    param_df."""
    centers = [param_df.TFP_to_YFP.values, param_df.TFP_to_mKate.values]
    centers = np.asarray(centers)

    counts = []
    for center in centers:
        count = count_neighbours(center, dist, data, cols)
        counts.append(count)

    param_df['counts'] = counts

    return param_df
