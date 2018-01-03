# import packages to be used
import sys
import os
import pathlib
import numpy as np
import pandas as pd
import lmfit as lm
import matplotlib.pyplot as plt

from pyDOE import lhs
from matplotlib.backends.backend_pdf import PdfPages

# sys.path.append(os.path.abspath('/mnt/data/Laboratorio/Imaging three sensors/FRET_Transformation'))
from fret_transformation import earm_1_3 as earm
from fret_transformation import transformation as tf
from fret_transformation import anisotropy_functions as af
from fret_transformation import caspase_fit as cf
from fret_transformation import time_study as ts
from multiprocessing import Pool

cm = earm.Model()
# os.system("taskset -p 0xff %d" % os.getpid())
# %% Load data
work_dir = pathlib.Path('/mnt/data/Laboratorio/Imaging three sensors/2017-09-04_Images/')
# data_dir = work_dir.joinpath('2017-10-16_complex_noErode_order05.pandas')
# df = pd.read_pickle(str(data_dir))

# %%

fluorophores = ['YFP', 'mKate', 'TFP']
Colors = {'YFP': 'y', 'mKate': 'r', 'TFP': 'g'}
Differences_tags = ['TFP_to_YFP', 'TFP_to_mKate', 'YFP_to_mKate']


def timedif_from_params(params, Differences_tags, pp=None):
    t = np.arange(0, 72000, 600)
    res = cm.simulate(t, param_values=params)

    sim = sim_to_ani(res)
    sim['t'] = [t / 60]

    sim_aborted = {}
    for fluo in fluorophores:
        r = sim[fluo + '_r_from_i'].values[0]
        t = np.arange(0, len(r) * 10, 10)
        sim_aborted[fluo] = r[0] == r[-1]

        if pp is not None:
            plt.plot(sim.t.values[0], sim[fluo + '_r_from_i'].values[0], 'x' + Colors[fluo])

    if not any(sim_aborted.values()):
        sim = ts.find_complex_in_sim(sim)

        if pp is not None:
            pp.savefig()
            plt.close()
            note = ''
            for fluo in fluorophores:
                if len(sim[fluo + '_r_complex'].values[0]) == len(sim.t.values[0]):
                    plt.plot(sim.t.values[0], sim[fluo + '_r_complex'].values[0], 'x' + Colors[fluo])

                plt.scatter(sim[fluo + '_max_activity'].values, [0] * len(sim[fluo + '_max_activity'].values),
                            color=Colors[fluo])
                note = note + str(sim[fluo + '_max_activity'][0]) + '\n'
            pp.attach_note(note, positionRect=[100, 100, 100, 100])
            pp.savefig()
            plt.close()

        sim = ts.add_differences(sim, Difference_tags=Differences_tags)
        difs = {tag: sim[tag].values for tag in Differences_tags}
    else:
        difs = {tag: np.nan for tag in Differences_tags}

    return difs


def sim_to_ani(res, col='r_from_i'):
    fluo_to_cas = {'YFP': 68,
                   'mKate': 65,
                   'TFP': 62}
    fluo_to_ani = {'YFP': (.22, .3),
                   'mKate': (.23, .28),
                   'TFP': (.28, .34)}

    df = pd.DataFrame()
    for fluo in fluo_to_cas.keys():
        sens = res[:, fluo_to_cas[fluo]]
        sens_norm = sens / np.nanmax(sens)
        anis = [af.Anisotropy_FromFit(m,
                                      fluo_to_ani[fluo][1],
                                      fluo_to_ani[fluo][0],
                                      1)
                for m in sens_norm]

        df['_'.join([fluo, col])] = [anis]

    return df


# C3 not inhibited by XIAP
# cm.params['XIAP_ku'].set(value=0.9E-4)
# cm.params['XIAP_kd'].set(value=1E-3)
# cm.params['XIAP_kc'].set(value=0)
# cm.params['XIAP'].set(value=1E2)
# cm.params['pC9'].set(value=1E3)

def generate_param_sweep(N, space_params=None):
    if space_params is None:
        min_s = 1E2
        max_s = 1E7
        space_params = {206: (min_s, max_s),
                        207: (min_s, max_s),
                        208: (min_s, max_s)}  # ,
        # 'XIAP_kc': (0, 0.05),
        # 'CytoC' : (1E6, 5E6),
        # 'Apaf': (14E5, 18E5),
        # 'XIAP': (1E2, 1E3)}
        # 'C3S_ku': (0.5e-6, 2e-6),
        # 'C8S_ku': (0.5e-7, 2e-7),
        # 'C9S_ku': (2.5e-9, 9e-9)}#,
        # 'L50' : (1E2, 1E5),
        # 'Bcl2c' : (2E4 * 0.1, 2E4 * 2),
        # 'Apaf': (1E3 * min_f * 372, 1E3 * max_f * 372),
        # 'pC9' : (1E3 * min_f * 30, 1E3 * max_f * 30),
        # 'pC3' : (1E3 * min_f * 120, 1E3 * max_f * 120),
        # 'XIAP' : (1E3 * min_f * 63, 1E3 * max_f * 63),
        # 'Smac' : (1E3 * min_f * 126, 1E3 * max_f * 126)}#,
        # 'CytoC' : (1E3 * min_f * 1E4, 1E3 * max_f * 1E4)}
        # space_params = {'S3': (1E6, 1E7),
        #                 'S8': (1E6, 1E7),
        #                 'S9': (1E6, 1E7),
        #                 # 'C3S_ku': (0.5e-6, 2e-6),
        #                 # 'C8S_ku': (0.5e-7, 2e-7),
        #                 # 'C9S_ku': (2.5e-9, 9e-9)}#,
        #                 'Bcl2c' : (3.6E4, 4E4),
        #                 'Apaf': (1E6, 1E7),
        #                 'pC9' : (1.9E6, 3E6),
        #                 'pC3' : (2E6, 5E6),
        #                 'XIAP' : (1.5E6, 5E6),
        #                 'Smac' : (3E6, 2E7)}#,
        #                 # 'CytoC' : (1E3 * min_f * 1E4, 1E3 * max_f * 1E4)}

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
    var_cols = param_df.columns

    for tag in Differences_tags:
        param_df[tag] = np.nan

    for i in range(len(cm.parameters)):
        if cm.parameters[i].name.startswith('ks'):
            cm.parameters[i] = earm.Parameter(cm.parameters[i].name, 0 * cm.parameters[i].value)
    cm.parameters[5] = earm.Parameter('pC3_0', 100000)
    cm.parameters[7] = earm.Parameter('XIAP_0', 10000)
    # cm.parameters[10] = earm.Parameter('Mcl1_0', 200)
    cm.parameters[39] = earm.Parameter('kc8', 0.1)
    for i in param_df.index:
        print(i)
        for col in var_cols:
            cm.parameters[col] = earm.Parameter(cm.parameters[col].name, param_df[col][i])
        difs = timedif_from_params(cm.parameters, Differences_tags, pp=pp)

        for tag in Differences_tags:
            param_df = param_df.set_value(i, tag, difs[tag])

    return param_df


def plot_scatter_times(x, y, marker='o', color=None):
    if color is None:
        color = ['r' if this_x >= 0 and this_y >= 0
                 else 'b' if this_x >= 0 and this_y < 0
        else 'g' if this_x < 0 and this_y >= 0
        else 'm'
                 for this_x, this_y in zip(x, y)]

    plt.scatter(x, y, c=color, marker=marker, alpha=0.5)
    # plt.xlim((-35,35))
    # plt.ylim((-35,35))
    plt.xlabel('TFP_to_YFP')
    plt.ylabel('TFP_to_mKate')
    ax = plt.gca()
    ax.grid(True)


def count_neighbours(center, dist, df, cols_xy):
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


def add_counts(param_df):
    centers = [param_df.TFP_to_YFP.values, param_df.TFP_to_mKate.values]
    centers = np.asarray(centers)

    counts = []
    for center in centers:
        count = count_neighbours(center, (5, 5), df, ('TFP_to_YFP', 'TFP_to_mKate'))
        counts.append(count)

    param_df['counts'] = counts

    return param_df


# df = ts.add_differences(df, Difference_tags=Differences_tags)


# pdf_dir = work_dir.joinpath('simfit.pdf')
# with PdfPages(str(pdf_dir)) as pp:


# param_df = add_counts(param_df)

# print(param_df)

# plot_scatter_times(df.TFP_to_YFP.values, df.TFP_to_mKate.values)
# plot_scatter_times(param_df.TFP_to_YFP.values, param_df.TFP_to_mKate.values, marker='x', color='k')
# plt.show()

save_dir = work_dir.joinpath('sim_params')


def sim_and_save(i):
    param_df = generate_param_sweep(10)
    pdf_path = save_dir.joinpath('complex.pdf')
    with PdfPages(str(pdf_path)) as pp:
        param_df = add_times_from_sim(param_df, Differences_tags, pp)
    savename = save_dir.joinpath('earm13_xiap_1e2_xiapdeg_00_%03d.pandas' % i)
    param_df.to_pickle(str(savename))
    return i


# f = 300
# rehms = {'Apaf': 372,
#          'pC9' : 30,
#          'pC3' : 120,
#          'XIAP' : 63,
#          'Smac' : 126}
#
# for key in rehms.keys():
#     val = rehms[key] * 1E3 * f / 100
#     cm.params[key].set(value=val)
sim_and_save(0)

# cors = 1
# if __name__ == '__main__':
#     with Pool(cors) as p:
#         import time
#         t = time.clock()
#         for a in p.imap_unordered(sim_and_save, range(cors)):
#             print(a)
#         print(time.clock() - t)
