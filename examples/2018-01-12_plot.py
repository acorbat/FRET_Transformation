import pathlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly as ply
import seaborn as sns

from matplotlib.mlab import griddata


def load_data(filename='2017-10-16_complex_noErode_order05_filtered_derived'):
    work_dir = pathlib.Path('/mnt/data/Laboratorio/Imaging three sensors/2017-09-04_Images/')
    data_dir = work_dir.joinpath(filename+'.pandas')
    return pd.read_pickle(str(data_dir))


def load_sim(filename='earm10_000'):
    work_dir = pathlib.Path('/mnt/data/Laboratorio/Imaging three sensors/2017-09-04_Images/sim_params')
    data_dir = work_dir.joinpath(filename+'.pandas')
    return pd.read_pickle(str(data_dir))


def mass_histogram(data, interp=False):
    hist, xedges, yedges = np.histogram2d(data[0], data[1], range=[[-20, 35], [-20, 35]], bins=np.arange(-20, 36, 4))
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
        xs, ys = np.meshgrid(xcen, ycen, indexing='ij')
        hist = griddata(centers_x, centers_y, hist, xs, ys, interp='linear')
        centers_x = xs[::-1]
        centers_y = ys


    return hist, centers_x, centers_y


def visvis(data, percentile):
    data = np.asarray(data).T

    c = np.mean(data, axis=0)

    d = np.linalg.norm(data-c, 2, axis=1)

    th = np.percentile(d, percentile)

    sel = d < th

    from scipy.spatial import ConvexHull
    hull = ConvexHull(data[sel, :])

    return hull, data[sel, :]


def plot_2dhist(times):
    times = (times[0][np.isfinite(times[0])], times[1][np.isfinite(times[1])])
    sns.kdeplot(times[0], times[1], cmap="Blues", shade=True, shade_lowest=False, clip=((-20, 35), (-20, 35)), n_levels=4)


def plot_data(times):
    times = (times[0][np.isfinite(times[0])], times[1][np.isfinite(times[1])])
    hist, centers_x, centers_y = mass_histogram(times, interp=True)
    # centers_x = (centers_x[:-1] + centers_x[1:])/2
    # centers_y = (centers_y[:-1] + centers_y[1:])/2

    max_count = np.max(hist)
    hist = hist/max_count
    plt.contour(centers_x, centers_y, hist.T, levels=[0.15, 0.5, 0.8], cmap='viridis')


def plot_show():
    plt.xlabel('caspase-3 to caspase-9')
    plt.ylabel('caspase-3 to caspase-8')

    plt.xlim(-20, 35)
    plt.ylim(-20, 35)

    # plt.fill_between([0, 35], [0, 0], [35, 35], alpha=0.5, color='orange')
    # plt.fill_between([0, 35], [0, 0], [-35, -35], alpha=0.5, color='b')
    # plt.fill_between([-35, 0], [0, 0], [35, 35], alpha=0.5, color='g')
    # plt.fill_between([-35, 0], [0, 0], [-35, -35], alpha=0.5, color='y')


fluorophores = ['YFP', 'mKate', 'TFP']
exp_data = load_data()
mask = [all([exp_data[fluo+'_good_der'][i] for fluo in fluorophores]) for i in exp_data.index]
exp_times = (exp_data.TFP_to_YFP.values[mask], exp_data.TFP_to_mKate.values[mask])
sim_data = load_sim('earm10_varligand_varrecep_3_6_varxiap_2')
sim_times = (sim_data.TFP_to_YFP.values, sim_data.TFP_to_mKate.values)

plot_2dhist(sim_times)
plot_show()
plot_data(exp_times)
plt.scatter(exp_times[0], exp_times[1], alpha=0.1, color='r')
plt.show()
