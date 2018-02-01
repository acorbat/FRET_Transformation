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

sim_data = load_sim('earm10_varligand_4_varrecep_3_varxiap_2')
mask = [True if np.isfinite(sim_data.TFP_to_YFP[i]) and np.isfinite(sim_data.TFP_to_mKate[i]) else False for i in sim_data.index]
sim_times = np.asarray((sim_data.TFP_to_YFP.values[mask], sim_data.TFP_to_mKate.values[mask]))
sim_times += np.random.normal(0, 2, sim_times.shape)

exp_data = load_data()
exp_data = exp_data.query('Content == "TNF alpha"')
mask = [all([exp_data[fluo+'_good_der'][i] for fluo in fluorophores]) for i in exp_data.index]
exp_times = (exp_data.TFP_to_YFP.values[mask], exp_data.TFP_to_mKate.values[mask])

exp_hist, xedges, yedges = np.histogram2d(exp_times[0], exp_times[1], range=[[-30, 35], [-30, 35]], bins=np.arange(-30, 36, 5))
sim_hist, xedges, yedges = np.histogram2d(sim_times[0], sim_times[1], range=[[-30, 35], [-30, 35]], bins=np.arange(-30, 36, 5))


def intersec_hist(hist_1, hist_2):
    assert hist_1.shape == hist_2.shape

    if len(hist_1.shape) > 1:
        hist_1 = hist_1.flatten()
    if len(hist_2.shape) > 1:
        hist_2 = hist_2.flatten()

    min_hist = np.min([hist_1, hist_2], axis=0)

    return np.sum(min_hist) / np.sum(hist_2)


def corr_hist(hist_1, hist_2):
    assert hist_1.shape == hist_2.shape

    return np.corrcoef(hist_1.flatten(), hist_2.flatten())[0, 1]


def test_hist_comp(function):
    # generate 2D normal distributions
    dist_original = np.random.multivariate_normal([0,0], [[1, 0], [0, 1]], size=300)
    dist_origin2 = np.random.multivariate_normal([0,0], [[1, 0], [0, 1]], size=300)
    dist_corr_x1 = np.random.multivariate_normal([0, 0.5], [[1, 0], [0, 1]], size=300)
    dist_corr_x2 = np.random.multivariate_normal([0, 2], [[1, 0], [0, 1]], size=300)
    dist_corr_y1 = np.random.multivariate_normal([0, 0.5], [[1, 0], [0, 1]], size=300)
    dist_corr_y2 = np.random.multivariate_normal([0, 2], [[1, 0], [0, 1]], size=300)
    dist_ellipse = np.random.multivariate_normal([0, 0], [[1, 0.8], [0.8, 1]], size=300)

    dists = {'original': dist_original,
             'original_2' : dist_origin2,
             'corr_x1': dist_corr_x1,
             'corr_x2': dist_corr_x2,
             'corr_y1': dist_corr_y1,
             'corr_y2': dist_corr_y2,
             'ellipse': dist_ellipse}

    hists = dict()
    for key, dist in dists.items():
        hist, xedges, yedges = np.histogram2d(dist[:, 0], dist[:, 1], range=[[-5, 5], [-5, 5]], bins=np.arange(-5, 5, 0.5))
        hists[key] = hist

    fig, axs = plt.subplots(3, 3, figsize=(10, 10))
    fig.subplots_adjust(hspace=1, wspace=1)
    axs = axs.ravel()

    for i, (key, hist) in enumerate(hists.items()):
        val = function(hists['original'], hist)
        axs[i].scatter(dists['original'][:, 0], dists['original'][:, 1], color='b', alpha=0.5)
        axs[i].scatter(dists[key][:, 0], dists[key][:, 1], color='r', alpha=0.5)
        axs[i].set_title(key + ' = ' + str(val))

    hist, xedges, yedges = np.histogram2d(dists['original'][:, 0], dists['original'][:, 1], range=[[-5, 5], [-5, 5]], bins=np.arange(-5, 5, 1))
    hists['bin_1_orig'] = hist

    hist, xedges, yedges = np.histogram2d(dists['original_2'][:, 0], dists['original_2'][:, 1], range=[[-5, 5], [-5, 5]], bins=np.arange(-5, 5, 1))
    hists['bin_1_var'] = hist

    hist, xedges, yedges = np.histogram2d(dists['original'][:, 0], dists['original'][:, 1], range=[[-5, 5], [-5, 5]],
                                          bins=np.arange(-5, 5, 2))
    hists['bin_2_orig'] = hist

    hist, xedges, yedges = np.histogram2d(dists['original_2'][:, 0], dists['original_2'][:, 1],
                                          range=[[-5, 5], [-5, 5]], bins=np.arange(-5, 5, 2))
    hists['bin_2_var'] = hist

    val = function(hists['bin_1_orig'], hists['bin_1_var'])
    axs[7].scatter(dists['original'][:, 0], dists['original'][:, 1], color='b', alpha=0.5)
    axs[7].scatter(dists['original_2'][:, 0], dists['original_2'][:, 1], color='r', alpha=0.5)
    axs[7].set_title('bin_1' + ' = ' + str(val))

    val = function(hists['bin_2_orig'], hists['bin_2_var'])
    axs[8].scatter(dists['original'][:, 0], dists['original'][:, 1], color='b', alpha=0.5)
    axs[8].scatter(dists['original_2'][:, 0], dists['original_2'][:, 1], color='r', alpha=0.5)
    axs[8].set_title('bin_2' + ' = ' + str(val))

    plt.show()


def test_binning(function, dist_type='same'):
    dist_orig = np.random.multivariate_normal([0, 0], [[1, 0], [0, 1]], size=300)
    if dist_type == 'same':
        dist_comp = np.random.multivariate_normal([0, 0], [[1, 0], [0, 1]], size=300)
    elif dist_type == 'ellipse':
        dist_comp = np.random.multivariate_normal([0, 0], [[1, 0.8], [0.8, 1]], size=300)

    binnings = [0.5 , 1, 2, 3, 4]

    vals = []
    for binning in binnings:
        hist_orig, xedges, yedges = np.histogram2d(dist_orig[:, 0], dist_orig[:, 1], range=[[-5, 5], [-5, 5]],
                                                   bins=np.arange(-5, 5, binning))

        hist_comp, xedges, yedges = np.histogram2d(dist_comp[:, 0], dist_comp[:, 1], range=[[-5, 5], [-5, 5]],
                                                   bins=np.arange(-5, 5, binning))

        val = function(hist_orig, hist_comp)
        vals.append(val)

    plt.plot(binnings, vals)
    plt.show()
