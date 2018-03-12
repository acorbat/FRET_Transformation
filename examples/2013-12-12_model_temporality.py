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
# data_dir = work_dir.joinpath('OneCasp/OneCasp_derivations_order05_filtered.pandas')
data_dir = work_dir.joinpath('2017-10-16_complex_noErode_order05_filtered_derived_corrected2.pandas')
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
sweep_space = {'S3': (5E6, 1E7),
               'S8': (5E6, 1E7),
               'S9': (5E6, 1E7)}  # ,
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


def sim_and_save(name, other_params):
    param_df = ts.generate_param_sweep(1000, space_params=sweep_space)
    savename = save_dir.joinpath(name)
    pdf_path = savename.with_suffix('.pdf')
    with PdfPages(str(pdf_path)) as pp:
        param_df = ts.add_times_from_sim(param_df, Differences_tags, pp, params=other_params)
    param_df.to_pickle(str(savename))

    pdf_path = pdf_path.with_name(pdf_path.stem+'_map.pdf')
    with PdfPages(str(pdf_path)) as pp:
        ts.plot_scatter_times(df.TFP_to_YFP.values, df.TFP_to_mKate.values)
        ts.plot_scatter_times(param_df.TFP_to_YFP.values, param_df.TFP_to_mKate.values, marker='x', color='k')
        pp.savefig()
        plt.close()

        mask = [all([df[fluo+'_good_der'][i] for fluo in fluorophores]) for i in df.index]
        ts.plot_scatter_times(df.TFP_to_YFP.values[mask], df.TFP_to_mKate.values[mask], zoom=False)
        ts.plot_scatter_times(param_df.TFP_to_YFP.values, param_df.TFP_to_mKate.values, marker='x', color='k', zoom=False)
        pp.savefig()
        plt.close()

        for fluo in fluorophores:
            plt.scatter(param_df[fluo+'_earm_T_S'].values, param_df[fluo+'_T_S'].values, color=Colors[fluo])
            plt.title(fluo + ' T_S')
            plt.xlabel('T_S from EARM')
            plt.ylabel('T_S from sim')
            pp.savefig()
            plt.close()

    return param_df


# xiap = [2, 3, 4]
# xiap_deg = [.1, .05, .01, .005, 0]
# xiap_ku = np.arange(2, 6.2, 1)
# fs = np.arange(2.8, 3.7, 0.2)
ligands = [3, 4, 5]
# mults = [-2, -1, 2, 3, 4]

# for this_xiap in xiap:
#     for this_xiap_deg in xiap_deg:
#         name = 'earm10_varxiap_%01d_varxiapdeg_%03d.pandas' % (this_xiap, this_xiap_deg*100)
#         this_params = cm.params.copy()
#         this_params['XIAP'].set(value=10 ** this_xiap)
#         this_params['XIAP_kc'].set(value=this_xiap_deg)
#         sim_res = sim_and_save(name, this_params)

# for this_xiap_ku in xiap_ku:
#     name = 'earm10_varxiapku_%01d.pandas' % this_xiap_ku
#     this_params = cm.params
#     this_params['XIAP_ku'].set(value=2 / (10 ** this_xiap_ku))
#     sim_res = sim_and_save(name, this_params)

# for f in fs:
#     name = 'earm10_prop4_%03d.pandas' % (f*100)
#     f = 10 ** f
#     this_params = cm.params.copy()
#     this_vals = {'Apaf': f * 372,
#                  'pC9': f * 30,
#                  'pC3': f * 120,
#                  'XIAP': f * 63}  # ,
#                  # 'Smac': f * 126}  # ,
#                  # 'CytoC': f * 1E4}
#     for key, item in this_vals.items():
#         this_params[key].set(value=item)
#     sim_res = sim_and_save(name, this_params)

# name = 'earm13inivals.pandas'
# this_params = cm.params.copy()
# this_params['RnosiRNA'].set(value=1000)
# this_params['flip'].set(value=2000)
# this_params['pC8'].set(value=10000)
# this_params['Bid'].set(value=60000)
# this_params['Bax'].set(value=80000)
# this_params['Bcl2'].set(value=30000)
# sim_res = sim_and_save(name, this_params)
#
# name = 'earm13.pandas'
# this_params['L_kd'].set(value=1E-6)
# this_params['L_kc'].set(value=1E-2)
# this_params['DISC_ku'].set(value=1E-7)
# this_params['C3_ku'].set(value=1E-7)
# this_params['C6_ku'].set(value=1E-7)
# this_params['PARP_kd'].set(value=0.001)
# this_params['PARP_kc'].set(value=20)
# this_params['transloc'].set(value=1)
# this_params['v'].set(value=0.01)
# sim_res = sim_and_save(name, this_params)

# for this_ligand in ligands:
#     name = 'earm10_varligand_%01d_varrecep_3_varxiap_2.pandas' % this_ligand
#     this_params = cm.params.copy()
#     this_params['L50'].set(value=10 ** this_ligand)
#     this_params['RnosiRNA'].set(value=10 ** 3)
#     this_params['XIAP'].set(value=1E2)
#     sim_res = sim_and_save(name, this_params)

# for this_ligand in ligands:
#     for this_recep in ligands:
#         name = 'redVarCS1_earm10_varligand_%01d_varrecep_%01d_varxiap_2.pandas' % (this_ligand, this_recep)
#         this_params = cm.params.copy()
#         this_params['L50'].set(value=10 ** this_ligand)
#         this_params['RnosiRNA'].set(value=10 ** this_recep)
#         this_params['XIAP'].set(value=1E2)
#         sim_res = sim_and_save(name, this_params)
#
# for this_ligand in ligands:
#     for this_recep in ligands:
#         name = 'redVarCS1_earm10_varligand_%01d_varrecep_%01d_varxiap_3.pandas' % (this_ligand, this_recep)
#         this_params = cm.params.copy()
#         this_params['L50'].set(value=10 ** this_ligand)
#         this_params['RnosiRNA'].set(value=10 ** this_recep)
#         this_params['XIAP'].set(value=1E3)
#         sim_res = sim_and_save(name, this_params)

# for L_kc in L_kcs:
#     name = 'earm10_varLkc_%01d.pandas' % L_kc
#     this_params = cm.params.copy()
#     this_params['L_kc'].set(value=2 / (10 ** (-1* L_kc)))
#     sim_res = sim_and_save(name, this_params)

# sens = ['C3S', 'C8S', 'C9S']
# for sen in sens:
#     for mult in mults:
#         name = 'earm10_var%s_%01d.pandas' % (sen, mult)
#         this_params = cm.params.copy()
#         this_params[sen+'_kc'].set(value=this_params[sen+'_kc'].value * (10 ** (mult)))
#         sim_res = sim_and_save(name, this_params)

# name = 'redVarCs_earm10_nomodif.pandas'
# this_params = cm.params.copy()
# sim_res = sim_and_save(name, this_params)
#
name = 'redVarCs_earm10_varligand_3_varrecep_3_varxiap_2.pandas'
this_params = cm.params.copy()
this_params['L50'].set(value=10 ** 3)
this_params['RnosiRNA'].set(value=10 ** 3)
this_params['XIAP'].set(value=1E2)
sim_res = sim_and_save(name, this_params)
