import pathlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.ndimage import zoom
from matplotlib.mlab import griddata


def load_data(filename='2017-10-16_complex_noErode_order05_filtered_derived'):
    """Loads filename data from 2017-09-04_Images folder."""
    work_dir = pathlib.Path('/mnt/data/Laboratorio/Imaging three sensors/2017-09-04_Images/')
    data_dir = work_dir.joinpath(filename+'.pandas')
    return pd.read_pickle(str(data_dir))


def load_sim(filename='earm10_000'):
    """Loads filename data from sim_params folder"""
    work_dir = pathlib.Path('/mnt/data/Laboratorio/Imaging three sensors/2017-09-04_Images/sim_params')
    data_dir = work_dir.joinpath(filename+'.pandas')
    return pd.read_pickle(str(data_dir))


def mass_histogram(data, interp=False):
    """Generates a 2D histogram where the center of each bin is at the center of mass of each bin. If interp is True,
    then griddata is used to interpolate to generate a uniform grid."""
    hist, xedges, yedges = np.histogram2d(data[0], data[1], range=[[-30, 35], [-30, 35]], bins=np.arange(-30, 36, 3))
    hist = hist.flatten()

    centers_x = []
    centers_y = []
    for yini, yend in zip(yedges[-2::-1], yedges[-1:0:-1]):
        for xini, xend in zip(xedges[:-1], xedges[1:]):
            this_bin = [this for this in zip(data[0], data[1])
                        if xend > this[0] >= xini and
                        yend > this[1] >= yini]

            if this_bin:
                this_mean = np.mean(this_bin, axis=0)
                centers_x.append(this_mean[0])
                centers_y.append(this_mean[1])
            else:
                centers_x.append((xend+xini)/2)
                centers_y.append((yend+yini)/2)

    if interp:
        xcen = (xedges[:-1] + xedges[1:]) / 2
        ycen = (yedges[:-1] + yedges[1:]) / 2
        # xcen = np.linspace(xedges[0], xedges[-1], 25)
        # ycen = np.linspace(yedges[0], yedges[-1], 25)
        xs, ys = np.meshgrid(xcen, ycen, indexing='ij')
        hist = griddata(centers_x, centers_y, hist, xs, ys, interp='linear')
        centers_x = xs[::-1]
        centers_y = ys

    return hist, centers_x, centers_y


def plot_2dhist(times, data=False):
    """Uses gaussian kernel to generate a 2D plot of data distribution."""
    times = (times[0][np.isfinite(times[0])], times[1][np.isfinite(times[1])])
    sns.kdeplot(times[0], times[1], cmap="Blues", shade=True, shade_lowest=False, clip=((-30, 35), (-30, 35)), n_levels=4)


def plot_data(times):
    """Uses mass_histogram to generate a contour plot of the 2D histogram of data."""
    times = (times[0][np.isfinite(times[0])], times[1][np.isfinite(times[1])])
    hist, centers_x, centers_y = np.histogram2d(times[0], times[1], range=[[-30, 35], [-30, 35]], bins=np.arange(-30, 36, 5))  # mass_histogram(times, interp=True)
    # centers_x = (centers_x[:-1] + centers_x[1:])/2
    # centers_y = (centers_y[:-1] + centers_y[1:])/2

    hist = zoom(hist, 4, order=3)
    max_count = np.max(hist)
    hist = hist/max_count
    plt.contour(hist.T, extent=(centers_x[0], centers_x[-1], centers_y[0], centers_y[-1]), levels=[0.15, 0.5, 0.8], cmap='viridis')


def plot_show():
    """Adds general settings for the plot."""
    plt.xlabel('caspase-3 to caspase-9')
    plt.ylabel('caspase-3 to caspase-8')

    plt.xlim(-30, 35)
    plt.ylim(-30, 35)

    # plt.fill_between([0, 35], [0, 0], [35, 35], alpha=0.5, color='orange')
    # plt.fill_between([0, 35], [0, 0], [-35, -35], alpha=0.5, color='b')
    # plt.fill_between([-35, 0], [0, 0], [35, 35], alpha=0.5, color='g')
    # plt.fill_between([-35, 0], [0, 0], [-35, -35], alpha=0.5, color='y')


def fig_4c(sim_name='earm10_varligand_4_varrecep_3_varxiap_2'):
    fluorophores = ['YFP', 'mKate', 'TFP']
    exp_data = load_data()
    exp_data = exp_data.query('Content == "TNF alpha"')
    mask = [all([exp_data[fluo+'_good_der'][i] for fluo in fluorophores]) for i in exp_data.index]
    exp_times = (exp_data.TFP_to_YFP.values[mask], exp_data.TFP_to_mKate.values[mask])
    sim_data = load_sim(sim_name)
    sim_times = np.asarray((sim_data.TFP_to_YFP.values, sim_data.TFP_to_mKate.values))
    sim_times += np.random.normal(0, 2, sim_times.shape)

    plot_2dhist(sim_times)
    plot_show()
    plot_data(exp_times)
    plt.scatter(exp_times[0], exp_times[1], alpha=0.1, color='r')
    plt.show()


fluorophores = ['YFP', 'mKate', 'TFP']
exp_data = load_data()
