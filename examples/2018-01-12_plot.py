import pathlib
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.ndimage import zoom
from matplotlib.mlab import griddata
from matplotlib.backends.backend_pdf import PdfPages

from fret_transformation import anisotropy_functions as af
from fret_transformation import transformation as tf
from fret_transformation import caspase_model as cm
from fret_transformation import time_study as ts


def load_data(filename='2017-10-16_complex_noErode_order05_filtered_derived'):
    """Loads filename data from 2017-09-04_Images folder."""
    work_dir = pathlib.Path('/mnt/data/Laboratorio/Imaging three sensors/2017-09-04_Images/')
    data_dir = work_dir.joinpath(filename + '.pandas')
    return pd.read_pickle(str(data_dir))


def load_sim(filename='earm10_000'):
    """Loads filename data from sim_params folder"""
    work_dir = pathlib.Path('/mnt/data/Laboratorio/Imaging three sensors/2017-09-04_Images/sim_params')
    data_dir = work_dir.joinpath(filename + '.pandas')
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
                centers_x.append((xend + xini) / 2)
                centers_y.append((yend + yini) / 2)

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
    sns.kdeplot(times[0], times[1], cmap="Blues", shade=True, shade_lowest=False, clip=((-30, 35), (-30, 35)),
                n_levels=4)


def plot_data(times):
    """Uses mass_histogram to generate a contour plot of the 2D histogram of data."""
    times = (times[0][np.isfinite(times[0])], times[1][np.isfinite(times[1])])
    hist, centers_x, centers_y = np.histogram2d(times[0], times[1], range=[[-30, 35], [-30, 35]],
                                                bins=np.arange(-30, 36, 5))  # mass_histogram(times, interp=True)
    # centers_x = (centers_x[:-1] + centers_x[1:])/2
    # centers_y = (centers_y[:-1] + centers_y[1:])/2

    hist = zoom(hist, 4, order=3)
    max_count = np.max(hist)
    hist = hist / max_count
    plt.contour(hist.T, extent=(centers_x[0], centers_x[-1], centers_y[0], centers_y[-1]), levels=[0.15, 0.5, 0.8],
                cmap='viridis')


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


fluorophores = ['YFP', 'mKate', 'TFP']


def load_sim_and_data(filename='earm10_varligand_4_varrecep_3_varxiap_2'):
    sim_data = load_sim(filename)
    mask = [True if np.isfinite(sim_data.TFP_to_YFP[i]) and np.isfinite(sim_data.TFP_to_mKate[i]) else False for i in
            sim_data.index]
    sim_times = np.asarray((sim_data.TFP_to_YFP.values[mask], sim_data.TFP_to_mKate.values[mask]))
    sim_times += np.random.normal(0, 2, sim_times.shape)

    exp_data = load_data()
    exp_data = exp_data.query('Content == "TNF alpha"')
    mask = [all([exp_data[fluo + '_good_der'][i] for fluo in fluorophores]) for i in exp_data.index]
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
    dist_original = np.random.multivariate_normal([0, 0], [[1, 0], [0, 1]], size=300)
    dist_origin2 = np.random.multivariate_normal([0, 0], [[1, 0], [0, 1]], size=300)
    dist_corr_x1 = np.random.multivariate_normal([0, 0.5], [[1, 0], [0, 1]], size=300)
    dist_corr_x2 = np.random.multivariate_normal([0, 2], [[1, 0], [0, 1]], size=300)
    dist_corr_y1 = np.random.multivariate_normal([0, 0.5], [[1, 0], [0, 1]], size=300)
    dist_corr_y2 = np.random.multivariate_normal([0, 2], [[1, 0], [0, 1]], size=300)
    dist_ellipse = np.random.multivariate_normal([0, 0], [[1, 0.8], [0.8, 1]], size=300)

    dists = {'original': dist_original,
             'original_2': dist_origin2,
             'corr_x1': dist_corr_x1,
             'corr_x2': dist_corr_x2,
             'corr_y1': dist_corr_y1,
             'corr_y2': dist_corr_y2,
             'ellipse': dist_ellipse}

    hists = dict()
    for key, dist in dists.items():
        hist, xedges, yedges = np.histogram2d(dist[:, 0], dist[:, 1], range=[[-5, 5], [-5, 5]],
                                              bins=np.arange(-5, 5, 0.5))
        hists[key] = hist

    fig, axs = plt.subplots(3, 3, figsize=(10, 10))
    fig.subplots_adjust(hspace=1, wspace=1)
    axs = axs.ravel()

    for i, (key, hist) in enumerate(hists.items()):
        val = function(hists['original'], hist)
        axs[i].scatter(dists['original'][:, 0], dists['original'][:, 1], color='b', alpha=0.5)
        axs[i].scatter(dists[key][:, 0], dists[key][:, 1], color='r', alpha=0.5)
        axs[i].set_title(key + ' = ' + str(val))

    hist, xedges, yedges = np.histogram2d(dists['original'][:, 0], dists['original'][:, 1], range=[[-5, 5], [-5, 5]],
                                          bins=np.arange(-5, 5, 1))
    hists['bin_1_orig'] = hist

    hist, xedges, yedges = np.histogram2d(dists['original_2'][:, 0], dists['original_2'][:, 1],
                                          range=[[-5, 5], [-5, 5]], bins=np.arange(-5, 5, 1))
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

    fig, axs = plt.subplots(1, 2, figsize=(10, 6))
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
        df['bin_' + str(i) + '_mean'] = [np.mean(bins) for bins in df['bins_' + str(i)].values]
        df['bin_' + str(i) + '_std'] = [np.std(bins) for bins in df['bins_' + str(i)].values]
    json_dir = work_dir.joinpath('comparison.json')
    csv_dir = work_dir.joinpath('comparison.csv')
    df.to_json(str(json_dir), orient='index')
    df.to_csv(str(csv_dir))

    plot_comparison_from_df(df)

    return df


def plot_comparison_from_df(df):
    df = df.sort_values(by='bin_5_mean', ascending=False)
    plt.figure(figsize=(14, 30))
    color_dict = {'4': 'g', '5': 'r', '6': 'b'}
    for i, this_c in color_dict.items():
        plt.errorbar(df['bin_' + str(i) + '_mean'].values, np.arange(len(df['bin_' + str(i) + '_mean'].values)),
                     xerr=df['bin_' + str(i) + '_std'].values, marker='o', color=this_c, ecolor=this_c, ls='none',
                     elinewidth=3)
    plt.yticks(np.arange(len(df.file.values)), df.file.values)
    plt.grid()
    plt.tight_layout()


def estimate_pre_and_post():
    df = load_data()
    for fluo in fluorophores:
        pres = []
        poss = []
        pre_means = []
        pos_means = []
        pre_stds = []
        pos_stds = []
        for i in df.index:
            if df[fluo + '_good_der'][i]:
                pre = tf.pre_region(df[fluo + '_x0'][i], df[fluo + '_rate'][i], df[fluo + '_r_from_i'][i])
                pos = tf.post_region(df[fluo + '_x0'][i], df[fluo + '_rate'][i], df[fluo + '_r_from_i'][i])

                pres.append(pre)
                poss.append(pos)
                pre_means.append(np.nanmean(pre))
                pos_means.append(np.nanmean(pos))
                pre_stds.append(np.nanstd(pre))
                pos_stds.append(np.nanstd(pos))

            else:
                pres.append(np.nan)
                poss.append(np.nan)
                pre_means.append(np.nan)
                pos_means.append(np.nan)
                pre_stds.append(np.nan)
                pos_stds.append(np.nan)

        df[fluo + '_pre'] = pres
        df[fluo + '_pos'] = poss
        df[fluo + '_pre_mean'] = pre_means
        df[fluo + '_pos_mean'] = pos_means
        df[fluo + '_pre_std'] = pre_stds
        df[fluo + '_pos_std'] = pos_stds


def anisos_pdf(df):
    Colors = {'YFP': 'y',
              'mKate': 'r',
              'TFP': 'g'}
    work_dir = pathlib.Path('/mnt/data/Laboratorio/Imaging three sensors/2017-09-04_Images')
    pdf_dir = work_dir.joinpath('pre_and_post_aniso.pdf')
    pp = PdfPages(str(pdf_dir))
    for i in df.index:
        fig_created = False
        for fluo in fluorophores:
            if df[fluo + '_good_der'][i]:
                fig_created = True
                plt.plot(df[fluo + '_r_from_i'][i], Colors[fluo])
                plt.plot([0, 90], [df[fluo + '_pre_mean'][i]] * 2, Colors[fluo] + '--')
                plt.plot([0, 90], [df[fluo + '_pos_mean'][i]] * 2, Colors[fluo] + '--')
                plt.title(str(i))

        if fig_created:
            pp.savefig()
            plt.close()
    pp.close()

    hist_dir = pathlib.Path('/mnt/data/Laboratorio/Imaging three sensors/img/Figura 1/')
    for fluo in fluorophores:
        mask_pre = np.isfinite(df[fluo + '_pre_mean'].values)
        plt.hist(df[fluo + '_pre_mean'].values[mask_pre], bins=20, color=Colors[fluo], alpha=0.6)
        mask_pos = np.isfinite(df[fluo + '_pos_mean'].values)
        plt.hist(df[fluo + '_pos_mean'].values[mask_pos], bins=20, color=Colors[fluo], alpha=0.6)
        pp.savefig()
        plt.close()

        difs = df[fluo + '_pos_mean'].values - df[fluo + '_pre_mean'].values
        mask_dif = np.isfinite(difs)
        plt.hist(difs[mask_dif], bins=20, color=Colors[fluo], alpha=0.6)
        pp.savefig()
        plt.close()
    pp.close()


def plot_anis_hist(vals, color='b', alpha=0.6, binsize=0.002):
    mask_pre = np.isfinite(vals)
    vals = vals[mask_pre]
    first_bin = min(vals) - min(vals) % binsize
    last_bin = max(vals) + binsize
    plt.hist(vals,
             bins=np.arange(first_bin, last_bin, binsize),
             color=color,
             alpha=alpha,
             edgecolor='black')
    plt.ylabel('counts')
    plt.xlabel('Anisotropy')


def fig_anisos_hists(df):
    fluorophores = ['YFP', 'mKate', 'TFP']
    Colors = {'YFP': 'y',
              'mKate': 'r',
              'TFP': 'g'}
    img_dir = pathlib.Path('/mnt/data/Laboratorio/Imaging three sensors/img/figure_1/')

    for fluo in fluorophores:
        hist_anis_name = img_dir.joinpath('anis_hist_' + fluo + '.png')
        plot_anis_hist(df[fluo + '_pre_mean'].values, color=Colors[fluo])

        plot_anis_hist(df[fluo + '_pos_mean'].values, color=Colors[fluo])
        plt.savefig(str(hist_anis_name))
        plt.close()

        hist_difs_name = img_dir.joinpath('dif_hist_' + fluo + '.png')
        difs = df[fluo + '_pos_mean'].values - df[fluo + '_pre_mean'].values
        plot_anis_hist(difs, color=Colors[fluo], alpha=1)
        plt.savefig(str(hist_difs_name))
        plt.close()


def fig_anisos_box(df):
    fluorophores = ['TFP', 'YFP', 'mKate']
    Colors = {'YFP': (189 / 255, 214 / 255, 48 / 255),
              'mKate': (240 / 255, 77 / 255, 35 / 255),
              'TFP': (59 / 255, 198 / 255, 244 / 255)}
    img_dir = pathlib.Path('/mnt/data/Laboratorio/Imaging three sensors/img/figure_1/')
    box_dir = img_dir.joinpath('boxaniso.png')

    pres = []
    poss = []
    difs = []
    for fluo in fluorophores:
        pre = df[fluo + '_pre_mean'].values
        mask_pre = np.isfinite(pre)
        pre = pre[mask_pre]
        pres.append(pre)

        pos = df[fluo + '_pos_mean'].values
        mask_pos = np.isfinite(pos)
        pos = pos[mask_pos]
        poss.append(pos)

        dif = df[fluo + '_pos_mean'].values - df[fluo + '_pre_mean'].values
        mask_dif = np.isfinite(dif)
        dif = dif[mask_dif]
        difs.append(dif)

    plt.rc('xtick', labelsize=24)
    plt.rc('ytick', labelsize=18)
    plt.rc('axes', labelsize=20)

    fig, axs = plt.subplots(2, 1, figsize=(20, 14), sharex=True)

    boxprops = axs[0].boxplot(pres,
                              patch_artist=True)
    for n, fluo in enumerate(fluorophores):
        plt.setp(boxprops['boxes'][n], facecolor=Colors[fluo])
        plt.setp(boxprops['medians'][n], color='k')
        plt.setp(boxprops['fliers'][n], markerfacecolor=Colors[fluo], alpha=0.5)

    boxprops = axs[0].boxplot(poss,
                              patch_artist=True)
    for n, fluo in enumerate(fluorophores):
        plt.setp(boxprops['boxes'][n], facecolor=Colors[fluo], ls='--')
        plt.setp(boxprops['medians'][n], color='k')
        plt.setp(boxprops['fliers'][n], markerfacecolor=Colors[fluo], alpha=0.5)
    axs[0].set_ylabel('Anisotropy')

    boxprops = axs[1].boxplot(difs,
                              patch_artist=True)
    for n, fluo in enumerate(fluorophores):
        plt.setp(boxprops['boxes'][n], facecolor=Colors[fluo])
        plt.setp(boxprops['medians'][n], color='k')
        plt.setp(boxprops['fliers'][n], markerfacecolor=Colors[fluo], alpha=0.5)
    axs[1].set_ylabel('Difference')

    # font = {'family': 'normal',
    #         'weight': 'bold',
    #         'size': 28}
    #
    # matplotlib.rc('font', **font)

    plt.xticks([1, 2, 3], ['tagBFP/mCerulean', 'mCitrine/mCitrine', 'mCherry/mKate'])
    plt.tight_layout()
    plt.subplots_adjust(hspace=.0)
    plt.savefig(str(box_dir))


def pdf_best_curves():
    good_curves = [2188,
                   2330,
                   2355,
                   2357,
                   2435,
                   2512,
                   2514,
                   2527,
                   2535,
                   2557,
                   2584,
                   2772,
                   2776,
                   2865,
                   2930]

    img_dir = pathlib.Path('/mnt/data/Laboratorio/Imaging three sensors/img/figure_2/')
    pdf_dir = img_dir.joinpath('best_curves.pdf')
    pp = PdfPages(str(pdf_dir))

    for i in good_curves:
        fig, axs = plt.subplots(2, 1, sharex=True, figsize=(20, 14))
        for fluo in fluorophores:
            time = np.linspace(0, 15, 90)
            axs[0].plot(time, df[fluo + '_r_from_i'][i], color=Colors[fluo])
            axs[0].set_ylabel('Anisotropy')

            axs[1].plot(time, df[fluo + '_r_complex'][i], color=Colors[fluo])
            axs[1].set_ylabel('Derivative')
            axs[1].set_xlabel('Time (hr)')
        plt.suptitle(str(i))
        pp.savefig()
        plt.close()

    pp.close()


def fig_2(df, fluo, ind):
    img_dir = pathlib.Path('/mnt/data/Laboratorio/Imaging three sensors/img/figure_2/')
    img_dir = img_dir.joinpath('typical_analysis.png')

    Colors = {'YFP': (189 / 255, 214 / 255, 48 / 255),
              'mKate': (240 / 255, 77 / 255, 35 / 255),
              'TFP': (59 / 255, 198 / 255, 244 / 255)}

    fig, axs = plt.subplots(3, 1, sharex=True, figsize=(20, 16))

    time = np.linspace(0, 15, 90)
    f = af.Fluos_FromInt(df[fluo + '_par_mean'][ind], df[fluo + '_per_mean'][ind])
    axs[0].plot(time, f, color=Colors[fluo])
    axs[0].plot(time, df[fluo + '_par_area'][ind], color=Colors[fluo])

    axs[1].plot(time, df[fluo + '_r_from_i'][ind], color=Colors[fluo])
    axs[1].set_ylabel('Anisotropy')

    axs[2].plot(time, df[fluo + '_r_complex'][ind], color=Colors[fluo])
    axs[2].set_ylabel('Derivative')
    axs[2].set_xlabel('Time (hr)')

    plt.tight_layout()
    plt.subplots_adjust(hspace=.0)
    plt.savefig(str(img_dir))
    plt.close()


def fig_2_sim(fluo):
    img_dir = pathlib.Path('/mnt/data/Laboratorio/Imaging three sensors/img/figure_2/')
    img_dir = img_dir.joinpath('sim_analysis.png')

    fig, axs = plt.subplots(2, 1, sharex=True, figsize=(20, 14))

    Colors = {'YFP': (189 / 255, 214 / 255, 48 / 255),
              'mKate': (240 / 255, 77 / 255, 35 / 255),
              'TFP': (59 / 255, 198 / 255, 244 / 255)}

    fluo_to_cplx = {'YFP': 'S9:C9',
                    'mKate': 'S8:C8',
                    'TFP': 'S3:C3'}

    t = np.arange(0, 54000, 600)
    earm_params = cm.params.copy()

    ccs = [5E5, 5E6, 5E7]
    for cc in ccs:
        for casp in ['S3', 'S8', 'S9']:
            earm_params[casp].set(value=cc)

        ani_vals = {'YFP': (.23, .29),
                    'mKate': (.26, .29),
                    'TFP': (.28, .32)}

        model = cm.simulate(t, earm_params)
        model = ts.sim_to_ani(model, fluo_to_ani=ani_vals)
        model = ts.find_complex_in_sim(model)

        time = np.linspace(0, 15, 90)
        axs[0].plot(time, model[fluo + '_r_from_i'][0], color=Colors[fluo])
        axs[0].axvline(x=model[fluo + '_max_activity'][0] / 60, color='gray', ls='--', alpha=0.6)
        axs[0].set_ylabel('Anisotropy')

        last1, = axs[1].plot(time, model[fluo_to_cplx[fluo]][0] / np.max(model[fluo_to_cplx[fluo]][0]),
                    color=Colors[fluo], label='Activity')
        last2 = axs[1].scatter(time, model[fluo + '_r_complex'][0] / np.max(model[fluo + '_r_complex'][0]),
                    c=Colors[fluo], edgecolors='k', label='Derivative')
        axs[1].axvline(x=model[fluo + '_max_activity'][0]/60, color='gray', ls='--', alpha=0.6)
        axs[1].set_ylabel('Derivative')
        axs[1].set_xlabel('Time (hr)')

    plt.legend([last1, last2], ['Activity', 'Derivative'])
    plt.tight_layout()
    plt.subplots_adjust(hspace=.0)
    plt.savefig(str(img_dir))
    plt.close()


def fig_3a(df, ind):
    img_dir = pathlib.Path('/mnt/data/Laboratorio/Imaging three sensors/img/figure_3/')
    dat_dir = img_dir.joinpath('onecasp_curves.png')

    Colors = {'YFP': (189 / 255, 214 / 255, 48 / 255),
              'mKate': (240 / 255, 77 / 255, 35 / 255),
              'TFP': (59 / 255, 198 / 255, 244 / 255)}

    fig, axs = plt.subplots(2, 1, sharex=True, figsize=(20, 14))
    for fluo in fluorophores:
        time = np.arange(0, 50 * 15, 15) / 60
        axs[0].plot(time, df['r_' + fluo][ind], color=Colors[fluo])
        axs[0].axvline(x=df[fluo + '_max_activity'][ind] / 60, color=Colors[fluo], ls='--', alpha=0.6)
        axs[0].set_ylabel('Anisotropy')

        axs[1].plot(time, df[fluo + '_r_complex'][ind], color=Colors[fluo])
        axs[1].axvline(x=df[fluo + '_max_activity'][ind] / 60, color=Colors[fluo], ls='--', alpha=0.6)
        axs[1].set_ylabel('Derivative')
        axs[1].set_xlabel('Time (hr)')

    plt.tight_layout()
    plt.subplots_adjust(hspace=.0)
    plt.savefig(str(dat_dir))
    plt.close()


def fig_3a_superposed(df):
    img_dir = pathlib.Path('/mnt/data/Laboratorio/Imaging three sensors/img/figure_3/')
    dat_dir = img_dir.joinpath('onecasp_curves_superposed.png')

    Colors = {'YFP': (189 / 255, 214 / 255, 48 / 255),
              'mKate': (240 / 255, 77 / 255, 35 / 255),
              'TFP': (59 / 255, 198 / 255, 244 / 255)}

    fig, axs = plt.subplots(2, 1, sharex=True, figsize=(20, 14))
    for i in df.index:
        if all([df[fluo + '_good_der'][i] for fluo in fluorophores]):
            for fluo in fluorophores:
                time = np.arange(0, 50 * 15, 15) - df.TFP_max_activity[i]
                time /= 60
                axs[0].plot(time, df['r_' + fluo][i], color=Colors[fluo], alpha=0.6)
                axs[0].set_ylabel('Anisotropy')
                axs[0].set_xlim([-2.5, 2.5])
                axs[0].set_ylim([0.18, 0.35])

                axs[1].plot(time, df[fluo + '_r_complex'][i], color=Colors[fluo], alpha=0.6)
                axs[1].set_ylabel('Derivative')
                axs[1].set_xlabel('Time (hr)')
                axs[1].set_xlim([-2.5, 2.5])
                axs[1].set_ylim([-0.001, 0.004])

    plt.tight_layout()
    plt.subplots_adjust(hspace=.0)
    plt.savefig(str(dat_dir))
    plt.close()


def fig_4b(df, ind):
    img_dir = pathlib.Path('/mnt/data/Laboratorio/Imaging three sensors/img/figure_4/')
    dat_dir = img_dir.joinpath('max_act_times_from_data.png')

    Colors = {'YFP': (189 / 255, 214 / 255, 48 / 255),
              'mKate': (240 / 255, 77 / 255, 35 / 255),
              'TFP': (59 / 255, 198 / 255, 244 / 255)}

    fig, axs = plt.subplots(2, 1, sharex=True, figsize=(20, 14))
    for fluo in fluorophores:
        time = np.linspace(0, 15, 90)
        axs[0].plot(time, df[fluo + '_r_from_i'][ind], color=Colors[fluo])
        axs[0].set_ylabel('Anisotropy')

        axs[1].plot(time, df[fluo + '_r_complex'][ind], color=Colors[fluo])
        axs[1].set_ylabel('Derivative')
        axs[1].set_xlabel('Time (hr)')

    plt.tight_layout()
    plt.subplots_adjust(hspace=.0)
    plt.savefig(str(dat_dir))
    plt.close()

    sim_dir = img_dir.joinpath('max_act_times_from_sim.png')

    t = np.arange(0, 54000, 600)
    earm_params = cm.params.copy()
    for casp in ['S3', 'S8', 'S9']:
        earm_params[casp].set(value=5E6)
    earm_params['XIAP'].set(value=1E2)
    earm_params['L50'].set(value=1E4)
    earm_params['RnosiRNA'].set(value=1E3)

    ani_vals = {'YFP': (.23, .29),
                'mKate': (.26, .29),
                'TFP': (.28, .32)}

    model = cm.simulate(t, earm_params)
    model = ts.sim_to_ani(model, fluo_to_ani=ani_vals)
    model = ts.find_complex_in_sim(model)

    fig, axs = plt.subplots(2, 1, sharex=True, figsize=(20, 14))
    for fluo in fluorophores:
        time = np.linspace(0, 15, 90)
        axs[0].plot(time, model[fluo + '_r_from_i'][0], color=Colors[fluo])
        axs[0].set_ylabel('Anisotropy')

        axs[1].plot(time, model[fluo + '_r_complex'][0], color=Colors[fluo])
        axs[1].set_ylabel('Derivative')
        axs[1].set_xlabel('Time (hr)')

    plt.tight_layout()
    plt.subplots_adjust(hspace=.0)
    plt.savefig(str(sim_dir))
    plt.close()


def fig_4c(sim_name='earm10_varligand_4_varrecep_3_varxiap_2'):
    img_dir = pathlib.Path('/mnt/data/Laboratorio/Imaging three sensors/img/figure_4/')
    img_dir = img_dir.joinpath('hist2d_comparison_' + sim_name + '_.png')

    fluorophores = ['YFP', 'mKate', 'TFP']
    exp_data = load_data()
    exp_data = exp_data.query('Content == "TNF alpha"')
    mask = [all([exp_data[fluo + '_good_der'][i] for fluo in fluorophores]) for i in exp_data.index]
    exp_times = exp_data.TFP_to_YFP.values[mask], exp_data.TFP_to_mKate.values[mask]
    exp_times = np.asarray(exp_times).T
    sim_data = load_sim(sim_name)
    sim_times = np.asarray((sim_data.TFP_to_YFP.values, sim_data.TFP_to_mKate.values)).T
    sim_times += np.random.normal(0, 2, sim_times.shape)
    df = pd.DataFrame(exp_times, columns=['TFP_to_YFP', 'TFP_to_mKate'])
    df['origin'] = 'exp'
    df_sim = pd.DataFrame(sim_times, columns=['TFP_to_YFP', 'TFP_to_mKate'])
    df_sim['origin'] = 'sim'
    df_sim = df_sim.dropna(axis=0, how='any')
    df = df.append(df_sim)

    cmap_dict = {'sim': 'Blues',
                 'exp': 'Reds'}
    color_dict = {'sim': 'b',
                  'exp': 'r'}
    g = sns.JointGrid("TFP_to_YFP", "TFP_to_mKate", df, xlim=(-35, 35), ylim=(-35, 35))
    for origin, this_df in df.groupby("origin"):
        sns.distplot(this_df["TFP_to_YFP"], ax=g.ax_marg_x, color=color_dict[origin])
        sns.distplot(this_df["TFP_to_mKate"], ax=g.ax_marg_y, vertical=True, color=color_dict[origin])
        if origin == 'sim':
            sns.kdeplot(this_df["TFP_to_YFP"], this_df["TFP_to_mKate"], cmap=cmap_dict[origin], alpha=0.8,
                        ax=g.ax_joint)
        elif origin == 'exp':
            plt.sca(g.ax_joint)
            times = (this_df["TFP_to_YFP"], this_df["TFP_to_mKate"])
            times = (times[0][np.isfinite(times[0])], times[1][np.isfinite(times[1])])
            hist, centers_x, centers_y = np.histogram2d(times[0], times[1], range=[[-30, 35], [-30, 35]],
                                                        bins=np.arange(-30, 36,
                                                                       5))  # mass_histogram(times, interp=True)
            # centers_x = (centers_x[:-1] + centers_x[1:])/2
            # centers_y = (centers_y[:-1] + centers_y[1:])/2

            hist = zoom(hist, 4, order=3)
            max_count = np.max(hist)
            hist = hist / max_count
            plt.contour(hist.T, extent=(centers_x[0], centers_x[-1], centers_y[0], centers_y[-1]),
                        levels=[0.15, 0.5, 0.8],
                        cmap=cmap_dict[origin])

    # plot_2dhist(sim_times)
    # plot_show()
    # plot_data(exp_times)
    plt.scatter(exp_times[:, 0], exp_times[:, 1], alpha=0.1, color='r')
    plt.tight_layout()
    plt.savefig(str(img_dir))
    plt.close()


def fig_sup_pairtimes(df, name):
    work_dir = pathlib.Path('/mnt/data/Laboratorio/Imaging three sensors/img/supplementary/')
    img_dir = work_dir.joinpath(name + '_max_act_pairplot.png')

    df = df[['TFP_max_activity', 'YFP_max_activity', 'mKate_max_activity']]
    df = df.dropna(axis=0, how='any')

    fig = sns.pairplot(df, size=5, kind='reg')
    fig.savefig(str(img_dir))
