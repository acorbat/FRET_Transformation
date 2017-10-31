# import packages to be used
import sys
import pathlib
import numpy as np
import pandas as pd
import lmfit as lm
import matplotlib.pyplot as plt

from pyDOE import lhs

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
Differences_tags = ['YFP_to_TFP', 'YFP_to_mKate', 'TFP_to_mKate']

def timedif_from_params(params, pp=None):
    t = np.arange(0, 72000, 600)
    sim = cm.simulate(t, cm.params)

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
            plt.plot(sim.t.values[0], sim[fluo+'_r_from_i'].values[0], 'x')
            plt.plot(t, cf.sigmoid(t, *popt))

    if not any(sim[fluo+'_base'].values == np.nan):
        sim = tf.find_complex(sim)

        if pp is not None:
            pp.savefig()
            plt.close()
            note = ''
            for fluo in fluorophores:
                plt.plot(sim.t.values[0], sim[fluo+'_r_complex'].values[0], 'x')
                note += str(sim[fluo+'_max_activity'][0]+'\n')
            pp.attach_note(note, positionRect=[100,100,100,100])
            pp.savefig()
            plt.close()

        sim = ts.add_differences(sim, Difference_tags=Differences_tags)
        difs = [sim[tag].values for tag in Differences_tags]
    else:
        difs = [np.nan] * 3

    return difs



def sim_to_ani(df, col='r_from_i'):
    fluo_to_cas = {'YFP': 'C9',
                   'mKate': 'C8',
                   'TFP': 'C3'}
    fluo_to_ani = {'YFP': (.22, .3),
                   'mKate': (.23, .28),
                   'TFP': (.28, .34)}

    for fluo in fluo_to_cas.keys():
        sens_norm = [sens/np.nanmax(sens)
                     for sens in df['S'+fluo_to_cas[fluo]].values]
        anis = [af.Anisotropy_FromFit(m,
                                      fluo_to_ani[fluo][1],
                                      fluo_to_ani[fluo][0],
                                      1)
                for m in sens_norm]

        df['_'.join([fluo, col])] = anis

    return df

N = 10
param_percents = lhs(6, samples=N)

space_params = {'S3': (1E3, 1E6),
                'S8': (1E3, 1E6),
                'S9': (1E3, 1E6),
                'C3_ku': (1e-7, 1e-5),
                'C8_ku': (1e-8, 1e-6),
                'C9_ku': (5e-10, 5e-8)}

difs = np.zeros((3, N))
for i, perc in enumerate(param_percents):
    j=0
    for param, (minval, maxval) in space_params.items():
        val = minval + (maxval - minval) * perc[j]

        j+=1
        cm.params[param].set(value=val)

    difs[:, i] = timedif_from_params(cm.params)

df = ts.add_differences(df, Difference_tags=Differences_tags)

plt.scatter(df.YFP_to_TFP.values, df.TFP_to_mKate.values)
plt.scatter(difs[0], difs[1])
plt.show()
