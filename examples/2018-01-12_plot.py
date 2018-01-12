import pathlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly as ply
import seaborn as sns


def load_data(filename='2017-10-16_complex_noErode_order05_filtered_derived'):
    work_dir = pathlib.Path('/mnt/data/Laboratorio/Imaging three sensors/2017-09-04_Images/')
    data_dir = work_dir.joinpath(filename+'.pandas')
    return pd.read_pickle(str(data_dir))


def load_sim(filename='earm10_000'):
    work_dir = pathlib.Path('/mnt/data/Laboratorio/Imaging three sensors/2017-09-04_Images/sim_params')
    data_dir = work_dir.joinpath(filename+'.pandas')
    return pd.read_pickle(str(data_dir))


def plot_2dhist(times):
    sns.kdeplot(times[0], times[1], cmap="Blues", shade=True, shade_lowest=False, clip=((-10, 35), (-10, 35)))


def plot_data(times):
    hist, xedges, yedges = np.histogram2d(times[0], times[1], range=[[-20, 35], [-20, 35]], bins=np.arange(-20, 36, 5))
    center_x = (xedges[:-1] + xedges[1:]) / 2
    center_y = (yedges[:-1] + yedges[1:]) / 2
    mesh = np.meshgrid(center_x, center_y)
    plt.scatter(mesh[0], mesh[1], s=hist, marker='x', color='k')


def plot_show():
    plt.xlabel('caspase-3 to caspase-9')
    plt.ylabel('caspase-3 to caspase-8')

    plt.xlim(-20, 35)
    plt.ylim(-20, 35)

    plt.fill_between([0, 35], [0, 0], [35, 35], alpha=0.5, color='orange')
    plt.fill_between([0, 35], [0, 0], [-35, -35], alpha=0.5, color='b')
    plt.fill_between([-35, 0], [0, 0], [35, 35], alpha=0.5, color='g')
    plt.fill_between([-35, 0], [0, 0], [-35, -35], alpha=0.5, color='y')


fluorophores = ['YFP', 'mKate', 'TFP']
exp_data = load_data()
mask = [all([exp_data[fluo+'_good_der'][i] for fluo in fluorophores]) for i in exp_data.index]
exp_times = (exp_data.TFP_to_YFP.values[mask], exp_data.TFP_to_mKate.values[mask])
exp_hist = np.histogram2d(exp_times[0], exp_times[1], range=[[-20, 35], [-20, 35]], bins=np.arange(-20,36,5))
sim_data = load_sim()
sim_times = (sim_data.TFP_to_YFP.values, sim_data.TFP_to_mKate.values)
plot_2dhist(sim_times)
plot_show()
plot_data(exp_times)
