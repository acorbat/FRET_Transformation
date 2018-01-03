# import packages to be used
import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pyDOE import lhs
from matplotlib.backends.backend_pdf import PdfPages

from fret_transformation import caspase_model as cm
from fret_transformation import transformation as tf
from fret_transformation import anisotropy_functions as af
from fret_transformation import caspase_fit as cf
from fret_transformation import time_study as ts

# Load data
work_dir = pathlib.Path('/mnt/data/Laboratorio/Imaging three sensors/2017-09-04_Images/')
data_dir = work_dir.joinpath('2017-10-16_complex_noErode_order05_filtered_derived.pandas')
df = pd.read_pickle(str(data_dir))

# Define lists of parameters to be used

fluorophores = ['YFP', 'mKate', 'TFP']
Colors = {'YFP': 'y', 'mKate': 'r', 'TFP': 'g'}
Differences_tags = ['TFP_to_YFP', 'TFP_to_mKate', 'YFP_to_mKate']

# C3 not inhibited by XIAP
# cm.params['XIAP_ku'].set(value=0.9E-4)
# cm.params['XIAP_kd'].set(value=1E-3)
# cm.params['XIAP_kc'].set(value=0)
# cm.params['XIAP'].set(value=1E2)
# cm.params['pC9'].set(value=1E3)


# 'Apaf': (1E3 * min_f * 372, 1E3 * max_f * 372),
# 'pC9' : (1E3 * min_f * 30, 1E3 * max_f * 30),
# 'pC3' : (1E3 * min_f * 120, 1E3 * max_f * 120),
# 'XIAP' : (1E3 * min_f * 63, 1E3 * max_f * 63),
# 'Smac' : (1E3 * min_f * 126, 1E3 * max_f * 126)}#,
# 'CytoC' : (1E3 * min_f * 1E4, 1E3 * max_f * 1E4)}
sweep_space = {'S3': (1E4, 1E6),
               'S8': (1E4, 1E6),
               'S9': (1E4, 1E6)}  # ,
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

save_dir = work_dir.joinpath('sim_params')


def sim_and_save(i):
    param_df = ts.generate_param_sweep(100, space_params=sweep_space)
    savename = save_dir.joinpath('earm10_%03d.pandas' % i)
    pdf_path = savename.with_suffix('.pdf')
    with PdfPages(str(pdf_path)) as pp:
        param_df = ts.add_times_from_sim(param_df, Differences_tags, pp)
    param_df.to_pickle(str(savename))

    pdf_path = pdf_path.with_name(pdf_path.stem+'_map')
    with PdfPages(str(pdf_path)) as pp:
        ts.plot_scatter_times(df.TFP_to_YFP.values, df.TFP_to_mKate.values)
        ts.plot_scatter_times(param_df.TFP_to_YFP.values, param_df.TFP_to_mKate.values, marker='x', color='k')
        pp.savefig()
        plt.close()

        mask = [all([df[fluo+'_good_der'][i] for fluo in fluorophores]) for i in df.index]
        ts.plot_scatter_times(df.TFP_to_YFP.values[:], df.TFP_to_mKate.values[:], zoom=False)
        ts.plot_scatter_times(param_df.TFP_to_YFP.values, param_df.TFP_to_mKate.values, marker='x', color='k', zoom=False)
        pp.savefig()
        plt.close()

    return param_df

sim_res = sim_and_save(0)
