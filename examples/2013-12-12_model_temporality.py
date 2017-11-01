# import packages to be used
import sys
import pathlib
import numpy as np
import pandas as pd
import lmfit as lm
import matplotlib.pyplot as plt

from pyDOE import lhs
from matplotlib.backends.backend_pdf import PdfPages

from fret_transformation import caspase_model as cm
from fret_transformation import transformation as tf
from fret_transformation import anisotropy_functions as af
from fret_transformation import caspase_fit as cf
from fret_transformation import time_study as ts

#%% Load data
work_dir = pathlib.Path(
            '/mnt/data/Laboratorio/Imaging three sensors/2017-09-04_Images')
data_dir = work_dir.joinpath('2017-10-16_complex_noErode_order05.pandas')
df = pd.read_pickle(str(data_dir))

#%%

fluorophores = ['YFP', 'mKate', 'TFP']
Colors = {'YFP':'y', 'mKate':'r', 'TFP':'g'}
Differences_tags = ['TFP_to_YFP', 'TFP_to_mKate', 'YFP_to_mKate']

def timedif_from_params(params, Differences_tags, pp=None):
    t = np.arange(0, 72000, 600)
    sim = cm.simulate(t, params)

    sim = sim_to_ani(sim)

    for fluo in fluorophores:
        r = sim[fluo+'_r_from_i'].values[0]
        t = np.arange(0, len(r)*10, 10)
        try:
            popt, _, _, _ = tf.nanfit(cf.sigmoid, r, xdata=t, p0=[0.2, 0.3, 0.2, 210])
            sim[fluo+'_base'] = popt[0]
            sim[fluo+'_amplitude'] = popt[1]
            sim[fluo+'_rate'] = popt[2]
            sim[fluo+'_x0'] = popt[3]
        except RuntimeError:
            sim[fluo+'_base'] = np.nan
            sim[fluo+'_amplitude'] = np.nan
            sim[fluo+'_rate'] = np.nan
            sim[fluo+'_x0'] = np.nan
        if pp is not None:
            plt.plot(sim.t.values[0], sim[fluo+'_r_from_i'].values[0], 'x'+Colors[fluo])
            plt.plot(t, cf.sigmoid(t, *popt), Colors[fluo])

    if not any(sim[fluo+'_base'].values == np.nan):
        sim = tf.find_complex(sim)

        if pp is not None:
            pp.savefig()
            plt.close()
            note = ''
            for fluo in fluorophores:
                if len(sim[fluo+'_r_complex'].values[0]) == len(sim.t.values[0]):
                    plt.plot(sim.t.values[0], sim[fluo+'_r_complex'].values[0], 'x'+Colors[fluo])
                plt.scatter(sim[fluo+'_max_activity'].values, [0]*len(sim[fluo+'_max_activity'].values), color=Colors[fluo])
                note = note + str(sim[fluo+'_max_activity'][0]) + '\n'
            pp.attach_note(note, positionRect=[100,100,100,100])
            pp.savefig()
            plt.close()

        sim = ts.add_differences(sim, Difference_tags=Differences_tags)
        difs = {tag: sim[tag].values for tag in Differences_tags}
    else:
        difs = {np.nan for tag in Differences_tags}

    return difs



def sim_to_ani(df, col='r_from_i'):
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

# C3 not inhibited by XIAP
#cm.params['XIAP_ku'].set(value=0.9E-4)
#cm.params['XIAP_kd'].set(value=1E-3)
cm.params['XIAP_kc'].set(value=0)
cm.params['XIAP'].set(value=1E4)

def generate_param_sweep(N, space_params = None):
    if space_params is None:
        space_params = {'S3': (1E4, 1E6),
                        'S8': (1E4, 1E6),
                        'S9': (1E4, 1E6)}#,
                        #'C3_ku': (0.5e-6, 2e-6),
                        #'C8_ku': (0.5e-7, 2e-7),
                        #'C9_ku': (2.5e-9, 9e-9)}

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


def add_times_from_sim(param_df, Differences_tags, pp=None):
    new_param_df = param_df.copy()

    for tag in Differences_tags:
        new_param_df[tag] = np.nan


    for i in param_df.index:

        for col in param_df.columns:
            cm.params[col].set(value=param_df[col][i])
        difs = timedif_from_params(cm.params, Differences_tags, pp=pp)

        for tag in Differences_tags:
            new_param_df = new_param_df.set_value(i, tag, difs[tag])

    return new_param_df


def plot_scatter_times(x, y, marker='o', color=None):
    if color is None:
        color = ['r' if this_x>=0 and this_y>=0
        else 'b' if this_x>=0 and this_y<0
        else 'g' if this_x<0 and this_y>=0
        else 'm'
        for this_x, this_y in zip(x, y)]

    plt.scatter(x, y, c=color, marker=marker, alpha=0.5)
    #plt.xlim((-35,35))
    #plt.ylim((-35,35))
    plt.xlabel('TFP_to_YFP')
    plt.ylabel('TFP_to_mKate')
    ax = plt.gca()
    ax.grid(True)


df = ts.add_differences(df, Difference_tags=Differences_tags)
param_df = generate_param_sweep(2)


pdf_dir = work_dir.joinpath('simfit.pdf')
with PdfPages(str(pdf_dir)) as pp:
    param_df = add_times_from_sim(param_df, Differences_tags, pp=pp)

plot_scatter_times(df.TFP_to_YFP.values, df.TFP_to_mKate.values)
plot_scatter_times(param_df.TFP_to_YFP.values, param_df.TFP_to_mKate.values, marker='x', color='k')
plt.show()
