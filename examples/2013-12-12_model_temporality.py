# import packages to be used
import sys
import pathlib
import numpy as np
import pandas as pd
import lmfit as lm
import matplotlib.pyplot as plt

# add path to local library

from fret_transformation import caspase_model as cm
from fret_transformation import transformation as tf
from fret_transformation import anisotropy_functions as af
from fret_transformation import caspase_fit as cf

#%% Load data
work_dir = pathlib.Path(
            '/mnt/data/Laboratorio/Imaging three sensors/2017-09-04_Images')
data_dir = work_dir.joinpath('2017-10-16_complex_noErode_order05.pandas')
df = pd.read_pickle(str(data_dir))

#%%

fluorophores = ['YFP', 'mKate', 'TFP']

t = np.arange(0, 72000, 600)
sim = cm.simulate(t, cm.params)

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

sim = sim_to_ani(sim)

for fluo in fluorophores:
    plt.plot(sim.t.values[0], sim[fluo+'_r_from_i'].values[0], 'x')
    r = sim[fluo+'_r_from_i'].values[0]
    t = np.arange(0, len(r)*10, 10)
    popt, _, _, _ = tf.nanfit(cf.sigmoid, r, xdata=t, p0=[0.2, 0.3, 0.2, 210])
    sim[fluo+'_base'] = popt[0]
    sim[fluo+'_amplitude'] = popt[1]
    sim[fluo+'_rate'] = popt[2]
    sim[fluo+'_x0'] = popt[3]
    plt.plot(t, cf.sigmoid(t, *popt))
plt.show()

sim = tf.find_complex(sim)

for fluo in fluorophores:
    plt.plot(sim.t.values[0], sim[fluo+'_r_complex'].values[0], 'x')
    print(sim[fluo+'_max_activity'][0])
plt.show()
