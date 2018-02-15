import pathlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.ndimage import zoom
from matplotlib.mlab import griddata
from matplotlib.backends.backend_pdf import PdfPages


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

def load_sim_and_data(filename='earm10_varligand_4_varrecep_3_varxiap_2'):
    sim_data = load_sim(filename)
    mask = [True if np.isfinite(sim_data.TFP_to_YFP[i]) and np.isfinite(sim_data.TFP_to_mKate[i]) else False for i in sim_data.index]
    sim_times = np.asarray((sim_data.TFP_to_YFP.values[mask], sim_data.TFP_to_mKate.values[mask]))
    sim_times += np.random.normal(0, 2, sim_times.shape)

    exp_data = load_data()
    exp_data = exp_data.query('Content == "TNF alpha"')
    mask = [all([exp_data[fluo+'_good_der'][i] for fluo in fluorophores]) for i in exp_data.index]
    exp_times = (exp_data.TFP_to_YFP.values[mask], exp_data.TFP_to_mKate.values[mask])

    return np.asarray(sim_times).T, np.asarray(exp_times).T


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

    # Filter out empty bins of both distributions
    hist_1_f = hist_1.flatten()
    hist_2_f = hist_2.flatten()
    mask = [True if x != 0 or y != 0
                 else False
                 for x, y in zip(hist_1_f, hist_2_f)]
    hist_1_f = hist_1_f[mask]
    hist_2_f = hist_2_f[mask]

    return np.corrcoef(hist_1_f, hist_2_f)[0, 1]


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
    lim = 5
    binnings = [0.5, 1, 2, 3]
    dist_orig = np.random.multivariate_normal([0, 0], [[1, 0], [0, 1]], size=300)
    if dist_type == 'same':
        dist_comp = np.random.multivariate_normal([0, 0], [[1, 0], [0, 1]], size=300)
    elif dist_type == 'ellipse':
        dist_comp = np.random.multivariate_normal([0, 0], [[1, 0.8], [0.8, 1]], size=300)
    elif dist_type == 'data':
        dist_orig, dist_comp = load_sim_and_data()
        lim = 35
        binnings = [0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]



    vals = []
    for binning in binnings:
        hist_orig, xedges, yedges = np.histogram2d(dist_orig[:, 0], dist_orig[:, 1], range=[[-lim, lim], [-lim, lim]],
                                                   bins=np.arange(-lim, lim, binning))

        hist_comp, xedges, yedges = np.histogram2d(dist_comp[:, 0], dist_comp[:, 1], range=[[-lim, lim], [-lim, lim]],
                                                   bins=np.arange(-lim, lim, binning))

        val = function(hist_orig, hist_comp)
        vals.append(val)

    fig, axs = plt.subplots(1, 2, figsize=(10,6))
    fig.subplots_adjust(hspace=1, wspace=1)
    axs = axs.ravel()

    axs[0].scatter(dist_orig[:, 0], dist_orig[:, 1], color='b', alpha=0.5)
    axs[0].scatter(dist_comp[:, 0], dist_comp[:, 1], color='r', alpha=0.5)
    axs[1].plot(binnings, vals)
    plt.show()


def test_func_dists(function, dist_1, dist_2, title='Comparison'):
    binnings = [0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    lim = 130

    vals = []
    for binning in binnings:
        hist_orig, xedges, yedges = np.histogram2d(dist_1[:, 0], dist_1[:, 1], range=[[-lim, lim], [-lim, lim]],
                                                   bins=np.arange(-lim, lim, binning))

        hist_comp, xedges, yedges = np.histogram2d(dist_2[:, 0], dist_2[:, 1], range=[[-lim, lim], [-lim, lim]],
                                                   bins=np.arange(-lim, lim, binning))

        val = function(hist_orig, hist_comp)
        vals.append(val)

    fig, axs = plt.subplots(1, 2, figsize=(10, 6))
    fig.subplots_adjust(hspace=1, wspace=1)
    axs = axs.ravel()

    axs[0].scatter(dist_1[:, 0], dist_1[:, 1], color='b', alpha=0.5)
    axs[0].scatter(dist_2[:, 0], dist_2[:, 1], color='r', alpha=0.5)
    axs[0].set_xlabel('x')
    axs[0].set_ylabel('y')

    axs[1].plot(binnings, vals)
    axs[1].set_xlabel('binning')
    axs[1].set_ylabel('value')
    plt.suptitle(title)

# Definitions for 2D cumulative distribution functions


def cdf_2d(dist, x, y, x_min=-np.inf, y_min=-np.inf):
    vals = [1 if this_x >= x_min and this_x < x
             and this_y >= y_min and this_y < y
            else 0
            for this_x, this_y in dist]

    return np.sum(vals)


def dist_cdf_mat(dist, range):
    xs = np.linspace(range[0][0], range[0][1], 50)
    ys = np.linspace(range[1][0], range[1][1], 50)

    Xs, Ys = np.meshgrid(xs, ys)
    Zs = []
    for X in xs:
        line_Z = []
        for Y in ys:
            Z = cdf_2d(dist, X, Y, x_min=range[0][0], y_min=range[1][0])
            line_Z.append(Z)
        Zs.append(line_Z)

    return Xs, Ys, Zs


def plot_cdf(dist, range):
    Xs, Ys, Zs = dist_cdf_mat(dist, range)

    fig = plt.figure()
    # ax = fig.gca(projection='3d')

    ax.plot_surface(Xs, Ys, Zs)
    plt.show()


## import files and test correlation

# files = ['earm10_varligand_4_varrecep_3_varxiap_2',
#          'earm10_prop4_220',
#          'earm10_prop4_230',
#          'earm10_prop4_240',]


# sim_dir = pathlib.Path('/mnt/data/Laboratorio/Imaging three sensors/2017-09-04_Images/sim_params')
# work_dir = pathlib.Path('/mnt/data/Laboratorio/Imaging three sensors/2017-09-04_Images/compare_hist')
#
# pdf_dir = work_dir.joinpath('corr_hist.pdf')
# pp = PdfPages(str(pdf_dir))
# for file in sim_dir.glob('*.pandas'):
#     file = file.stem
#     sim_times, exp_times = load_sim_and_data(file)
#     test_func_dists(corr_hist, sim_times, exp_times, title=file)
#     pp.savefig()
#     plt.close()
# pp.close()


def compare_sim(function, dist_1, dist_2):
    binnings = [4, 5, 6]
    lim = 130

    vals = []
    for binning in binnings:
        hist_orig, xedges, yedges = np.histogram2d(dist_1[:, 0], dist_1[:, 1], range=[[-lim, lim], [-lim, lim]],
                                                   bins=np.arange(-lim, lim, binning))

        hist_comp, xedges, yedges = np.histogram2d(dist_2[:, 0], dist_2[:, 1], range=[[-lim, lim], [-lim, lim]],
                                                   bins=np.arange(-lim, lim, binning))

        val = function(hist_orig, hist_comp)
        vals.append(val)

    return vals[1], vals[0], vals[2]


def compare_all_sims(function):
    sim_dir = pathlib.Path('/mnt/data/Laboratorio/Imaging three sensors/2017-09-04_Images/sim_params')
    work_dir = pathlib.Path('/mnt/data/Laboratorio/Imaging three sensors/2017-09-04_Images/compare_hist')

    files = []
    all_bin_5 = []
    all_bin_4 = []
    all_bin_6 = []
    for file in sim_dir.glob('*.pandas'):
        file = file.stem

        bins_4 = []
        bins_5 = []
        bins_6 = []
        for _ in range(5):
            sim_times, exp_times = load_sim_and_data(file)
            bin_5, bin_4, bin_6 = compare_sim(function, sim_times, exp_times)
            bins_4.append(bin_4)
            bins_5.append(bin_5)
            bins_6.append(bin_6)
        files.append(file)
        all_bin_4.append(bins_4)
        all_bin_5.append(bins_5)
        all_bin_6.append(bins_6)

    df = pd.DataFrame([files, all_bin_4, all_bin_5, all_bin_6])
    df = df.transpose()
    df.columns = ['file', 'bins_4', 'bins_5', 'bins_6']

    for i in [4, 5, 6]:
        df['bin_'+str(i)+'_mean'] = [np.mean(bins) for bins in df['bins_'+str(i)].values]
        df['bin_'+str(i)+'_std'] = [np.std(bins) for bins in df['bins_' + str(i)].values]
    json_dir = work_dir.joinpath('comparison.json')
    csv_dir = work_dir.joinpath('comparison.csv')
    df.to_json(str(json_dir), orient='index')
    df.to_csv(str(csv_dir))

    plot_comparison_from_df(df)

    return df


def plot_comparison_from_df(df):
    df = df.sort_values(by='bin_5_mean', ascending=False)
    plt.figure(figsize=(14, 30))
    color_dict = {'4':'g', '5':'r', '6':'b'}
    for i, this_c in color_dict.items():
        plt.errorbar(df['bin_'+str(i)+'_mean'].values, np.arange(len(df['bin_'+str(i)+'_mean'].values)),
                     xerr=df['bin_'+str(i)+'_std'].values, marker='o', color=this_c, ecolor=this_c, ls='none',
                     elinewidth=3)
    plt.yticks(np.arange(len(df.file.values)), df.file.values)
    plt.grid()
    plt.tight_layout()
