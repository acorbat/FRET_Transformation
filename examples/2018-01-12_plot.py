import pathlib
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.ndimage import zoom
from scipy.stats import gaussian_kde
from scipy.interpolate import splrep, splev
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, zoomed_inset_axes, mark_inset
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from matplotlib.mlab import griddata
from matplotlib.backends.backend_pdf import PdfPages

from img_manager import tifffile as tif

from fret_transformation import anisotropy_functions as af
from fret_transformation import transformation as tf
from fret_transformation import caspase_model as cm
from fret_transformation import time_study as ts

matplotlib.rcParams.update({'font.size': 8})

def load_data(filename='2017-10-16_complex_noErode_order05_filtered_derived_corrected_refiltered'):
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

Colors = {'YFP': (189 / 255, 214 / 255, 48 / 255),
          'mKate': (240 / 255, 77 / 255, 35 / 255),
          'TFP': (59 / 255, 198 / 255, 244 / 255)}


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
    json_dir = work_dir.joinpath('comparison_corr.json')
    csv_dir = work_dir.joinpath('comparison_corr.csv')
    df.to_json(str(json_dir), orient='index')
    df.to_csv(str(csv_dir))

    plot_comparison_from_df(df)

    return df


def plot_comparison_from_df(df, savename, height=None):
    img_dir = pathlib.Path('/mnt/data/Laboratorio/Imaging three sensors/img/supplementary/')
    img_dir = img_dir.joinpath(savename + '.svg')

    if height is None:
        height = len(df) * 0.25

    df_nomodif = df.query('file.str.contains("no modification")', engine='python')
    df = df.drop(df_nomodif.index[0])

    df = df.sort_values(by='bin_5_mean', ascending=False)
    df = df.append(df_nomodif, ignore_index=True)
    plt.figure(figsize=(3.2, height))
    # color_dict = {'4': 'g', '5': 'r', '6': 'b'}
    color_dict = {'5': 'r'}
    for i, this_c in color_dict.items():
        plt.errorbar(df['bin_' + str(i) + '_mean'].values, np.arange(len(df['bin_' + str(i) + '_mean'].values)),
                     xerr=df['bin_' + str(i) + '_std'].values, marker='o', color=this_c, ecolor=this_c, ls='none',
                     elinewidth=3)
    plt.ylim((-1, len(df.file.values)))
    plt.yticks(np.arange(len(df.file.values)), df.file.values)
    plt.grid()
    plt.tight_layout()
    plt.savefig(str(img_dir), format='svg')


def classify_and_plot_comparison():
    work_dir = pathlib.Path('/mnt/data/Laboratorio/Imaging three sensors/2017-09-04_Images/compare_hist')
    csv_dir = work_dir.joinpath('comparison_corr.csv')
    df = pd.read_csv(str(csv_dir))
    df = df.dropna()

    df_q = df.query('file.str.contains("prop")', engine='python')

    df_nomodif = df.query('file.str.contains("redVarCs_earm10_nomodif")', engine='python')
    df_nomodif.set_value(df_nomodif.index[0], 'file', 'no modification')

    df_onlyXIAP = df.query('file.str.contains("earm10_varxiap_2_varxiapdeg_010")', engine='python')
    name = '2' + ' | ' + '2' + ' | ' + '3'
    name = '%.1f | %.1f | %.1f' % (np.log10(10 ** 2 / 10 ** 5),
                                   np.log10(200 / 200),
                                   np.log10(3000 / 3000))
    df_onlyXIAP.set_value(df_onlyXIAP.index[0], 'file', name)

    df_lig_rec = df.query('file == "earm10_varligand_3_varrecep_3"', engine='python')
    name = '5' + ' | ' + '3' + ' | ' + '3'
    name = '%.1f | %.1f | %.1f' % (np.log10(10 ** 5 / 10 ** 5),
                                   np.log10(1000 / 200),
                                   np.log10(1000 / 3000))
    df_lig_rec.set_value(df_lig_rec.index[0], 'file', name)

    for i in df_q.index:
        num = int(df_q.file[i].split('_')[-1])
        if num > 10:
            num /= 100
        num = "%.2f" % num
        if 'prop4' in df.file[i]:
            name = '5 | ' + num
            df_q.set_value(i, 'file', name)

        elif 'prop5' in df.file[i]:
            name = '6 | ' + num
            df_q.set_value(i, 'file', name)

        else:
            name = '4 | ' + num
            df_q.set_value(i, 'file', name)

    df_q = df_q.append(df_nomodif)
    plot_comparison_from_df(df_q, 'proportionals', height=7.5)

    df_q = df.query('file.str.contains("redVarCS1_")', engine='python')

    for i in df_q.index:
        if 'nomodif' in df.file[i]:
            name = 'No modification'
            df_q.set_value(i, 'file', name)

        else:
            seps = df_q.file[i].split('_')
            lig = seps[3]
            rec = seps[5]
            xia = seps[-1]
            name = xia + ' | ' + rec + ' | ' + lig
            name = '%.1f | %.1f | %.1f' % (np.log10(10 ** float(xia) / 10 ** 5),
                                           np.log10(10 ** float(rec) / 200),
                                           np.log10(10 ** float(lig) / 3000))
            df_q.set_value(i, 'file', name)

    df_q = df_q.append(df_nomodif)
    df_q = df_q.append(df_onlyXIAP)
    df_q = df_q.append(df_lig_rec)
    plot_comparison_from_df(df_q, 'redVarCs', height=7.5)


def estimate_pre_and_post(df=None, tp=10):
    if df is None:
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
                pre = tf.pre_region(df[fluo + '_x0'][i], df[fluo + '_rate'][i], df[fluo + '_r_from_i'][i], timepoints=tp)
                pos = tf.post_region(df[fluo + '_x0'][i], df[fluo + '_rate'][i], df[fluo + '_r_from_i'][i], timepoints=tp)

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

    return df


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


def fig_anisos_violin(df):
    fluorophores = ['TFP', 'YFP', 'mKate']
    Colors = {'YFP': (189 / 255, 214 / 255, 48 / 255),
              'mKate': (240 / 255, 77 / 255, 35 / 255),
              'TFP': (59 / 255, 198 / 255, 244 / 255)}
    img_dir = pathlib.Path('/mnt/data/Laboratorio/Imaging three sensors/img/figure_1/')
    box_dir = img_dir.joinpath('violinaniso.svg')

    pres = {fluo: [] for fluo in fluorophores}
    poss = {fluo: [] for fluo in fluorophores}
    difs = {fluo: [] for fluo in fluorophores}
    for fluo in fluorophores:
        pre = df[fluo + '_pre_mean'].values
        mask_pre = np.isfinite(pre)
        pre = pre[mask_pre]
        pres[fluo] = pre

        pos = df[fluo + '_pos_mean'].values
        mask_pos = np.isfinite(pos)
        pos = pos[mask_pos]
        poss[fluo] = pos

        dif = df[fluo + '_pos_mean'].values - df[fluo + '_pre_mean'].values
        mask_dif = np.isfinite(dif)
        dif = dif[mask_dif]
        difs[fluo] = dif

    fig, axs = plt.subplots(2, 1, figsize=(3.3, 3.8), sharex=True)

    df_all = pd.DataFrame()
    for fluo in fluorophores:
        df_pre = pd.DataFrame(pres[fluo], columns=['ani'])
        df_pre['time'] = 'pre'
        df_pre['fluo'] = fluo
        df_pos = pd.DataFrame(poss[fluo], columns=['ani'])
        df_pos['time'] = 'pos'
        df_pos['fluo'] = fluo

        df_all = df_all.append(df_pre, ignore_index=True)
        df_all = df_all.append(df_pos, ignore_index=True)

    sns.violinplot(x='fluo', y='ani', hue='time', data=df_all, ax=axs[0], split=True, scale="count", inner='quartile',
                   scale_hue=False, order=fluorophores)
    plt.sca(axs[0])
    plt.ylim([0.21, 0.37])
    plt.yticks([0.22, 0.26, 0.30, 0.34])
    plt.ylabel('Anisotropy')

    df_difs = pd.DataFrame()
    for fluo in fluorophores:
        df_dif = pd.DataFrame(difs[fluo], columns=['dif'])
        df_dif['time'] = 'pre'
        df_dif['fluo'] = fluo

        df_difs = df_difs.append(df_dif, ignore_index=True)

    sns.violinplot(x='fluo', y='dif', data=df_difs, ax=axs[1], split=True, scale="count", inner='quartile',
                   scale_hue=False, order=fluorophores, palette=Colors)

    # boxprops = axs[1].boxplot([difs['TFP'][0], difs['mKate'][0], difs['YFP'][0]],
    #                           patch_artist=True)
    # for n, fluo in enumerate(fluorophores):
    #     plt.setp(boxprops['boxes'][n], facecolor=Colors[fluo])
    #     plt.setp(boxprops['medians'][n], color='k')
    #     plt.setp(boxprops['fliers'][n], markerfacecolor=Colors[fluo], alpha=0.5)
    # axs[1].set_ylabel('Difference')

    plt.sca(axs[1])
    plt.ylim([0, 0.09])
    plt.xticks([0, 1, 2], ['x-b', 'x-y', 'x-r'])
    plt.ylabel('Difference')
    plt.xlabel('')
    plt.tight_layout()
    plt.subplots_adjust(hspace=.0)
    plt.savefig(str(box_dir), format='svg')


def fig_1_cells(fluorophores=['TFP', 'YFP', 'mKate']):
    sav_dir = pathlib.Path('/mnt/data/Laboratorio/Imaging three sensors/img/figure_1/')
    sav_dir = sav_dir.joinpath('cells_anis_change.svg')

    mask_dir = pathlib.Path('/mnt/data/Laboratorio/Imaging three sensors/AnisoTestData/results/Regions/')
    mask = {}
    for t in [4, 50]:
        filename = '030_%03.f_%s_%s.tif' % (t, 'f', 'YFP')
        this_mask_dir = mask_dir.joinpath(filename)
        this_mask = tif.TiffFile(str(this_mask_dir)).asarray()
        mask[t] = this_mask

    mask[4] = mask[4] != 11
    mask[50] = mask[50] != 9

    imgs_r = {}
    imgs_r['ini'] = full_img(4)
    imgs_r['end'] = full_img(50)
    for fluo in fluorophores:
        imgs_r['ini'][fluo][[mask[4]]] = 0
        imgs_r['end'][fluo][[mask[50]]] = 0

    rect = {}
    rect['xy_ini'] = (1120, 260)
    rect['height_ini'] = 180
    rect['width_ini'] = 180
    rect['xy_end'] = (1200, 270)
    rect['height_end'] = 110
    rect['width_end'] = 110

    titles = {'TFP': 'x-b', 'mKate': 'x-r', 'YFP': 'x-y'}

    r_min = 0.17
    r_max = 0.36

    fig, axs = plt.subplots(2, 3, figsize=(4.3, 3.8))
    # plt.gcf().subplots_adjust(left=0.01, right=.9)

    for n, fluo in enumerate(fluorophores):
        for j, this_time in enumerate(['ini', 'end']):

            img = imgs_r[this_time][fluo]
            crop_loc = (rect['xy_' + this_time][0],
                        rect['xy_' + this_time][0] + rect['width_' + this_time],
                        rect['xy_' + this_time][1],
                        rect['xy_' + this_time][1] + rect['height_' + this_time])
            img = img[crop_loc[2]:crop_loc[3], crop_loc[0]:crop_loc[1]]
            img = np.nan_to_num(img)
            cmap = plt.cm.plasma
            cmap = cmap.set_under('black')
            im_r = axs[j][n].imshow(img, vmin=r_min, vmax=r_max, cmap='plasma')

            axs[j][n].tick_params(bottom='False', left='False', labelbottom='False', labelleft='False')

            if j == 0:
                axs[j][n].set_title(titles[fluo])

            if n == 2:
                cax = inset_axes(axs[j][n],
                                 width="7%",
                                 height="100%",
                                 bbox_transform=axs[j][n].transAxes,
                                 bbox_to_anchor=(0.2, .05, 1, 1),
                                 loc=1)
                norm = matplotlib.colors.Normalize(vmin=r_min, vmax=r_max)
                cb1 = matplotlib.colorbar.ColorbarBase(cax,
                                                       cmap='plasma', norm=norm,
                                                       orientation='vertical')
                cb1.set_ticks([r_min, r_max])

    # Add scalebar
    # 1um <-> 4.9751
    scalebar = AnchoredSizeBar(axs[0][0].transData,
                               50, '          ', 'lower right',
                               pad=0.1,
                               color='white',
                               frameon=False,
                               size_vertical=2,
                               label_top=True)  # ,
    # fontproperties=fontprops)

    axs[0][0].add_artist(scalebar)
    axs[0][0].set_ylabel('pre')
    axs[1][0].set_ylabel('post')
    plt.subplots_adjust(hspace=.0, wspace=0.1)
    plt.savefig(str(sav_dir), format='svg')


def pdf_best_curves():
    good_curves = [2188,
                   2177,
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


def fig_2_cells():
    sav_dir = pathlib.Path('/mnt/data/Laboratorio/Imaging three sensors/img/figure_2/')
    sav_dir = sav_dir.joinpath('cell_imgs.svg')
    img_dir = pathlib.Path('/mnt/data/Laboratorio/Imaging three sensors/Klauss/Casp3_YFP_20130214_po007_f_crop/')
    f_dir = img_dir.joinpath('Casp3_YFP_20130214_po007_f_crop.tif')
    r_dir = img_dir.joinpath('Casp3_YFP_20130214_po007_r_crop.tif')
    m_dir = img_dir.joinpath('masks.tif')

    f_imgs = tif.TiffFile(str(f_dir))
    f_series = f_imgs.asarray()
    r_imgs = tif.TiffFile(str(r_dir))
    r_series = r_imgs.asarray()
    m_imgs = tif.TiffFile(str(m_dir))
    masks = m_imgs.asarray()

    inds = [0, 22, 23, 24, 40]
    titles = {
        0  : '0 h',
        22 : '6 h',
        23 : '6 h 30 min',
        24 : '6 h 50 min',
        40 : '10 h'
    }
    fig, axs = plt.subplots(2, 5, figsize=(6.4, 2.2))
    plt.gcf().subplots_adjust(left=0.01, right=.9)

    for i, ind in enumerate(inds):
        img = f_series[ind]
        img[masks[ind] > 125] = 0
        cmap = plt.cm.Blues_r
        cmap.set_under(color='black')
        im1 = axs[0][i].imshow(img, vmin=0.026, vmax=0.23, cmap=cmap)
        axs[0][i].set_title(titles[ind])
        axs[0][i].axis('off')
        axs[0][0].set_ylabel('Fluorescence Intensity')
        axs[1][0].set_ylabel('Anisotropy')

        img = r_series[ind]
        img[masks[ind] > 125] = 0
        im2 = axs[1][i].imshow(img, vmin=0.20, vmax=0.33, cmap='plasma')
        axs[1][i].axis('off')

    cax = inset_axes(axs[0][4],
                     width="7%",
                     height="100%",
                     bbox_transform=axs[0][4].transAxes,
                     bbox_to_anchor=(0.2, 0.08, 1, 1),
                     loc=1)
    norm = matplotlib.colors.Normalize(vmin=0.026, vmax=0.23)
    cb1 = matplotlib.colorbar.ColorbarBase(cax,
                              cmap=cmap, norm=norm,
                              orientation='vertical')
    cb1.set_ticks([0.05, 0.10, 0.15, 0.20, 0.25])

    cax = inset_axes(axs[1][4],
                     width="7%",
                     height="100%",
                     bbox_transform=axs[1][4].transAxes,
                     bbox_to_anchor=(0.2, 0.08, 1, 1),
                     loc=1)
    norm = matplotlib.colors.Normalize(vmin=0.20, vmax=0.33)
    cb1 = matplotlib.colorbar.ColorbarBase(cax,
                                           cmap=matplotlib.cm.plasma, norm=norm,
                                           orientation='vertical')
    cb1.set_ticks([0.21, 0.26, 0.31])

    # Add scalebar
    # 1um <-> 4.9751
    scalebar = AnchoredSizeBar(axs[0][0].transData,
                               75, '          ', 'lower right',
                               pad=0.1,
                               color='white',
                               frameon=False,
                               size_vertical=2,
                               label_top=True)  # ,
    # fontproperties=fontprops)

    axs[0][0].add_artist(scalebar)
    plt.subplots_adjust(hspace=-.1, wspace=0.1)
    plt.savefig(str(sav_dir), format='svg')


def fig_2a(df, fluo, ind):
    img_dir = pathlib.Path('/mnt/data/Laboratorio/Imaging three sensors/img/figure_2/')
    img_dir = img_dir.joinpath('typical_analysis.svg')

    img_times = [0, 6 * 60, 6 * 60 + 30, 6 * 60 + 50, 10 * 60]

    Colors = {'YFP': (189 / 255, 214 / 255, 48 / 255),
              'mKate': (240 / 255, 77 / 255, 35 / 255),
              'TFP': (59 / 255, 198 / 255, 244 / 255)}

    fig, axs = plt.subplots(2, 1, sharex=True, figsize=(6.4, 3.3))

    time = np.linspace(0, 15, 90)
    # f = af.Fluos_FromInt(df[fluo + '_par_mean'][ind], df[fluo + '_per_mean'][ind])
    # axs[0].plot(time, f, color=Colors[fluo])
    # axs[0].plot(time, df[fluo + '_par_area'][ind], color=Colors[fluo])

    axs[0].plot(time, df[fluo + '_r_from_i'][ind], color='k')
    axs[0].axvline(x=df[fluo + '_max_activity'][ind] / 60, color='k', ls='--', alpha=0.6)
    for img_time in img_times:
        axs[0].axvline(x=img_time / 60, color='b', lw=1, alpha=0.6)
    axs[0].set_ylabel('Anisotropy')
    axs[0].set_yticks([0.23, 0.25, 0.27, 0.29])

    axs[1].plot(time, df[fluo + '_r_complex'][ind] / np.nanmax(df[fluo + '_r_complex'][ind]), color='k')
    axs[1].axvline(x=df[fluo + '_max_activity'][ind] / 60, color='k', ls='--', alpha=0.6)
    for img_time in img_times:
        axs[1].axvline(x=img_time / 60, color='b', lw=1, alpha=0.6)
    axs[1].set_ylabel('Activity (a.u.)')
    axs[1].set_xlabel('Time (h)')
    axs[1].set_yticks([0, 0.25, 0.5, 0.75, 1])

    plt.tight_layout()
    plt.subplots_adjust(hspace=.0)
    plt.savefig(str(img_dir), format='svg')


def fig_2_sim():
    img_dir = pathlib.Path('/mnt/data/Laboratorio/Imaging three sensors/img/figure_2/')
    img_dir = img_dir.joinpath('sim_analysis.svg')

    fig, axs = plt.subplots(2, 1, sharex=True, figsize=(6.4, 3.3))

    force_max_time = 5 + 30 / 60

    Colors = {'YFP': 'r',
              'mKate': (240 / 255, 77 / 255, 35 / 255),
              'TFP': 'b'}
    likes = {'YFP': 'hyperbolic-like',
              'mKate': (240 / 255, 77 / 255, 35 / 255),
              'TFP': 'sigmoid-like'}

    fluo_to_cplx = {'YFP': 'SC9',
                    'mKate': 'SC8',
                    'TFP': 'SC3'}

    t = np.arange(0, 54000, 600)
    earm_params = cm.params.copy()

    ccs = [5E5, 5E7]
    for cc in ccs:
        for casp in ['S3', 'S8', 'S9']:
            earm_params[casp].set(value=cc)

        ani_vals = {'YFP': (.23, .28, 1),
                    'mKate': (.23, .28, 1),
                    'TFP': (.23, .28, 1)}

        model = cm.simulate(t, earm_params)
        model = ts.sim_to_ani(model, fluo_to_ani=ani_vals)
        model = ts.find_complex_in_sim(model)

        real_vals = pd.DataFrame()
        for fluo in fluorophores:
            real_vals[fluo + '_r_from_i'] = model[fluo_to_cplx[fluo]].values
        real_vals = ts.find_complex_in_sim(real_vals)

        time = np.linspace(0, 15, 90)
        casp_plot = {fluo: [] for fluo in fluorophores}
        for fluo in fluorophores:
            if (fluo == 'TFP' and cc == 5E5) or (fluo == 'YFP' and cc == 5E7):
                dif_time = force_max_time - model[fluo + '_max_activity'][0] / 60
                this_time = time + dif_time

                f = splrep(this_time, model[fluo + '_r_from_i'][0], k=3)
                this_time_fine = np.arange(this_time[0], this_time[-1]+0.001, 1/60)
                interp = splev(this_time_fine, f, der=0)
                val_half = (ani_vals[fluo][1] + ani_vals[fluo][0]) / 2
                ind_half = np.where(interp >= val_half)[0][0]
                time_half = this_time_fine[ind_half]
                val_half = interp[ind_half]

                axs[0].plot(this_time, model[fluo + '_r_from_i'][0], color=Colors[fluo], label=likes[fluo])
                axs[0].scatter(time_half, val_half, color=Colors[fluo], marker='*', s=100, edgecolors='k')
                axs[0].axvline(x=force_max_time, color='k', ls='--', lw=2, alpha=0.6)
                axs[0].set_ylabel('Anisotropy')
                axs[0].legend(loc=2)

                # casp_plot[fluo], = axs[1].plot(time, model[fluo_to_cplx[fluo]][0] / np.max(model[fluo_to_cplx[fluo]][0]),
                #             color=Colors[fluo], label='Activity')
                last2 = axs[1].plot(this_time, model[fluo + '_r_complex'][0] / np.max(model[fluo + '_r_complex'][0]),
                                       color=Colors[fluo])
                axs[1].axvline(x=force_max_time, color='k', ls='--', lw=2, alpha=0.6)
                # max_time = real_vals[fluo + '_max_activity'][0] / 60
                dx = 0.12
                dy = 0.06
                axs[1].arrow(force_max_time-dx, 1+dy, dx, -dy, head_width=0.05, head_length=0.15, fc='k', ec='k',
                             length_includes_head=True)
                axs[1].set_ylabel('Activity (a.u.)')
                axs[1].set_xlabel('Time (h)')
                axs[1].set_ylim([-0.01, 1.15])

    plt.xlim(2, 10)
    # plt.legend([casp_plot['TFP'], casp_plot['mKate'], casp_plot['YFP'], last2], ['TFP Activity', 'mKate Activity', 'YFP Activity', 'Derivative'])
    plt.tight_layout()
    plt.subplots_adjust(hspace=.0)
    plt.savefig(str(img_dir), format='svg')
    # plt.close()


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


def fig_3a_r_single(df, ind, ax):
    time = np.arange(0, 50 * 15, 15) / 60
    for fluo in fluorophores:
        ax.plot(time, df['r_' + fluo][ind], color=Colors[fluo])
        ax.axvline(x=df[fluo + '_max_activity'][ind] / 60, color=Colors[fluo], ls='--')
        ax.set_ylabel('Anisotropy')


def fig_3a_der_single(df, ind, ax):
    time = np.arange(0, 50 * 15) / 60
    axins = zoomed_inset_axes(ax, 4, loc=1)
    for fluo in fluorophores:
        ax.plot(time, df[fluo + '_m_interp'][ind]/np.nanmax(df[fluo + '_m_interp'][ind]), color=Colors[fluo])
        ax.axvline(x=df[fluo + '_max_activity'][ind] / 60, color=Colors[fluo], ls='--')
        ax.set_ylabel('Activity (a.u.)')
        ax.set_xlabel('Time (h)')

        axins.plot(time, df[fluo + '_m_interp'][ind]/np.nanmax(df[fluo + '_m_interp'][ind]), color=Colors[fluo])
        axins.axvline(x=df[fluo + '_max_activity'][ind] / 60, color=Colors[fluo], ls='--')

    # sub region of the original image
    x1, x2, y1, y2 = 8.8, 9.2, 0.8, 1.01
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)

    plt.xticks([])
    plt.yticks([])

    # draw a bbox of the region of the inset axes in the parent axes and
    # connecting lines between the bbox and the inset axes area
    mark_inset(ax, axins, loc1=1, loc2=3, fc="none", ec="0.5")


def fig_3a_r_all(df):
    for i in df.index:
        if all([df[fluo + '_good_der'][i] for fluo in fluorophores]):
            for fluo in fluorophores:
                time = np.arange(0, 50 * 15, 15) - df.TFP_max_activity[i]
                time /= 60
                plt.plot(time, df['r_' + fluo][i], color=Colors[fluo], alpha=0.4)
                # plt.ylabel('Anisotropy')
                plt.xticks([])
                plt.yticks([])
                plt.xlim([-2.5, 2.5])
                plt.ylim([0.18, 0.35])


def fig_3a_der_all(df):
    for i in df.index:
        if all([df[fluo + '_good_der'][i] for fluo in fluorophores]):
            for fluo in fluorophores:
                time = np.arange(0, 50 * 15, 15) - df.TFP_max_activity[i]
                time /= 60

                comp = df[fluo + '_r_complex'][i]
                # comp = comp / np.nanmax(comp)

                plt.plot(time, comp, color=Colors[fluo], alpha=0.4)
                # plt.ylabel('Derivative')
                # plt.xlabel('Time (hr)')
                plt.xticks([])
                plt.yticks([])
                plt.xlim([-2.5, 2.5])
                plt.ylim([-0.001, 0.004])


def fig_3a_inlet(df, ind=142):
    img_dir = pathlib.Path('/mnt/data/Laboratorio/Imaging three sensors/img/figure_3/')
    dat_dir = img_dir.joinpath('onecasp_curves_inlet_mod.svg')

    fig, axs = plt.subplots(3, 1, sharex=True, figsize=(3.3, 3), gridspec_kw = {'height_ratios':[1, 15, 15]})

    times = {fluo: [] for fluo in fluorophores}
    for i in df.index:
        if all([df[fluo + '_good_der'][i] for fluo in fluorophores]):
            for fluo in fluorophores:
                times[fluo].append(df[fluo + '_max_activity'][i] / 60)
    for fluo in fluorophores:
        sns.rugplot(times['TFP'], height=1, ax=axs[0], color=Colors[fluo], lw=2, alpha=0.2)
    axs[0].axis('off')

    fig_3a_r_single(df, ind, axs[1])
    inset_axes(axs[1], width='30%', height='30%', loc=2)
    fig_3a_r_all(df)
    axs[1].set_yticks([0.23, 0.25, 0.27, 0.29, 0.31])

    fig_3a_der_single(df, ind, axs[2])
    # inset_axes(axs[2], width='30%', height='30%', loc=2)
    # fig_3a_der_all(df)

    plt.tight_layout()
    plt.subplots_adjust(hspace=.0)
    plt.savefig(str(dat_dir), format='svg')
    # plt.close()


def fig_3b(df):
    img_dir = pathlib.Path('/mnt/data/Laboratorio/Imaging three sensors/img/figure_3/')
    dat_dir = img_dir.joinpath('pairplot_onecasp.png')

    Colors = {'YFP': (189 / 255, 214 / 255, 48 / 255),
              'mKate': (240 / 255, 77 / 255, 35 / 255),
              'TFP': (59 / 255, 198 / 255, 244 / 255)}

    times = {}
    for fluo in fluorophores:
        times[fluo] = df[fluo + '_max_activity'].values[df[fluo + '_good_der'].values]

    for fluo in fluorophores:
        plt.hist(times[fluo], color=Colors[fluo])
        his_dir = img_dir.joinpath(fluo + '_onecasp_hist.png')
        plt.savefig(str(his_dir))
        plt.close()

    df_fil = pd.DataFrame()
    for i in df.index:
        if all([df[fluo + '_good_der'][i] for fluo in fluorophores]):
            df_fil = df_fil.append(df.loc[i])

    df = df_fil[['TFP_max_activity', 'YFP_max_activity', 'mKate_max_activity']]

    fig = sns.pairplot(df, size=5, kind='reg')
    fig.savefig(str(dat_dir))


def fig_3b_r_single(df, ind, ax):
    time = np.arange(0, 90 * 10, 10) / 60
    for fluo in fluorophores:
        ax.plot(time, df[fluo + '_r_from_i'][ind], color=Colors[fluo])
        ax.axvline(x=df[fluo + '_max_activity'][ind] / 60, color=Colors[fluo], ls='--')
        ax.set_ylabel('Anisotropy')


def fig_3b_der_single(df, ind, ax):
    time = np.arange(0, 90 * 10) / 60
    axins = zoomed_inset_axes(ax, 7, loc=1)
    for fluo in fluorophores:
        ax.plot(time, df[fluo + '_m_interp'][ind]/np.nanmax(df[fluo + '_m_interp'][ind]), color=Colors[fluo])
        ax.axvline(x=df[fluo + '_max_activity'][ind] / 60, color=Colors[fluo], ls='--')
        ax.set_ylabel('Activity (a.u.)')
        ax.set_xlabel('Time (h)')
        ax.set_ylim([-0.1, 1.1])

        axins.plot(time, df[fluo + '_m_interp'][ind] / np.nanmax(df[fluo + '_m_interp'][ind]), color=Colors[fluo])
        axins.axvline(x=df[fluo + '_max_activity'][ind] / 60, color=Colors[fluo], ls='--')

        # sub region of the original image
    x1, x2, y1, y2 = 6, 6.4, 0.9, 1.01
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)

    plt.xticks([])
    plt.yticks([])

    # draw a bbox of the region of the inset axes in the parent axes and
    # connecting lines between the bbox and the inset axes area
    mark_inset(ax, axins, loc1=1, loc2=3, fc="none", ec="0.5")


def fig_3b_r_all(df):
    for i in df.index:
        if i == 2072:
            print('skip 2072')
            continue
        if i == 1575:
            print('skip 1575')
            continue
        if all([df[fluo + '_good_der'][i] for fluo in fluorophores]):
            for fluo in fluorophores:
                time = np.arange(0, 90 * 10, 10) - df.TFP_max_activity[i]
                time /= 60
                plt.plot(time, df[fluo + '_r_from_i'][i], color=Colors[fluo], alpha=0.4)
                # plt.ylabel('Anisotropy')
                plt.xticks([])
                plt.yticks([])
                plt.xlim([-2.5, 2.5])
                plt.ylim([0.18, 0.35])


def fig_3b_der_all(df):
    for i in df.index:
        if i == 2072:
            print('skip 2072')
            continue
        if i == 1575:
            print('skip 1575')
            continue
        if all([df[fluo + '_good_der'][i] for fluo in fluorophores]):
            for fluo in fluorophores:
                time = np.arange(0, 90 * 10, 10) - df.TFP_max_activity[i]
                time /= 60
                try:
                    ind_start = np.where(time < -2.5)[0][-1]
                except IndexError:
                    ind_start = 0
                try:
                    ind_end = np.where(time > 2.5)[0][0]
                except IndexError:
                    ind_end = 900
                time = time[ind_start:ind_end]
                comp = df[fluo + '_r_complex'][i][ind_start:ind_end]
                # comp = comp / np.nanmax(comp)

                plt.plot(time, comp, color=Colors[fluo], alpha=0.4)
                # plt.ylabel('Derivative')
                # plt.xlabel('Time (hr)')
                plt.xticks([])
                plt.yticks([])
                plt.xlim([-2.5, 2.5])
                plt.ylim([-0.001, 0.004])


def fig_3b_inlet(df, ind=2355):
    img_dir = pathlib.Path('/mnt/data/Laboratorio/Imaging three sensors/img/figure_3/')
    dat_dir = img_dir.joinpath('exp_curves_inlet_mod.svg')

    fig, axs = plt.subplots(3, 1, sharex=True, figsize=(3.3, 3), gridspec_kw = {'height_ratios':[1, 15, 15]})

    times = {fluo: [] for fluo in fluorophores}
    for i in df.index:
        if all([df[fluo + '_good_der'][i] for fluo in fluorophores]):
            for fluo in fluorophores:
                times[fluo].append(df[fluo + '_max_activity'][i] / 60)
    for fluo in fluorophores:
        sns.rugplot(times['TFP'], height=1, ax=axs[0], color=Colors[fluo], lw=2, alpha=0.2)
    axs[0].axis('off')

    fig_3b_r_single(df, ind, axs[1])
    inset_axes(axs[1], width='30%', height='30%', loc=2)
    axs[1].set_yticks([0.23, 0.25, 0.27, 0.29, 0.31])
    fig_3b_r_all(df)

    fig_3b_der_single(df, ind, axs[2])
    # inset_axes(axs[2], width='30%', height='30%', loc=2)
    # fig_3b_der_all(df)

    plt.tight_layout()
    plt.subplots_adjust(hspace=.0)
    plt.savefig(str(dat_dir), format='svg')
    # plt.close()


def get_level_for_kde(x, y, level):
    kde = gaussian_kde([x, y])
    probs = kde.pdf([x, y])
    arr = np.asarray([x, y, probs]).T
    arr = arr[arr[:, 2].argsort()]
    ind = np.ceil(len(x) * level).astype(int)
    return arr[ind, 2]


def get_levels_for_kde(x, y, levels):
    vals = []
    for level in levels:
        val = get_level_for_kde(x, y, level)
        vals.append(val)
    return vals


def fig_3c(df):
    img_dir = pathlib.Path('/mnt/data/Laboratorio/Imaging three sensors/img/figure_3/')
    img_dir = img_dir.joinpath('onecasp_hist2d.svg')

    df_fil = pd.DataFrame()
    for i in df.index:
        if all([df[fluo + '_good_der'][i] for fluo in fluorophores]):
            df_fil = df_fil.append(df.loc[i])

    df_fil = df_fil.query('TFP_to_YFP < 20 and TFP_to_YFP >-20 and TFP_to_mKate < 20 and TFP_to_mKate > -20')

    g = sns.JointGrid("TFP_to_YFP", "TFP_to_mKate", df_fil, xlim=(-45, 25), ylim=(-10, 20), size=3.3)
    # original size: xlim=(-20, 20), ylim=(-20, 20)
    # fig 3D size : xlim=(-45, 25), ylim=(-10, 20)
    # zoomed :xlim=(-10, 10), ylim=(-10, 10)

    sns.distplot(df_fil["TFP_to_YFP"], kde=False, bins=30, ax=g.ax_marg_x)
    sns.distplot(df_fil["TFP_to_mKate"], kde=False, bins=30, ax=g.ax_marg_y, vertical=True)
    g.ax_joint.hexbin(df_fil["TFP_to_YFP"], df_fil["TFP_to_mKate"], gridsize=20, mincnt=1, cmap='Greys')
    sns.kdeplot(df_fil["TFP_to_YFP"], df_fil["TFP_to_mKate"], cmap='viridis', alpha=0.6, levels=get_levels_for_kde(df_fil["TFP_to_YFP"], df_fil["TFP_to_mKate"], [0.34, 0.68]),
                ax=g.ax_joint)
    g.ax_joint.axvline(x=0, color='k', lw=1, ls='--', alpha=0.5)
    g.ax_joint.axhline(y=0, color='k', lw=1, ls='--', alpha=0.5)
    # plt.sca(g.ax_joint)
    # times = np.asarray([df_fil["TFP_to_YFP"], df_fil["TFP_to_mKate"]])
    # plt.scatter(times[0], times[1], alpha=0.1, color='r')
    g.set_axis_labels('$\Delta$t (Cas3-b, Cas3-y) (min.)', '$\Delta$t (Cas3-b, Cas3-r) (min.)')
    g.ax_marg_x.set_xlabel('')
    g.ax_marg_y.set_ylabel('')
    plt.tight_layout()
    plt.savefig(str(img_dir), format='svg')


def fig_3d(filename='2017-10-16_complex_noErode_order05_filtered_derived'):
    img_dir = pathlib.Path('/mnt/data/Laboratorio/Imaging three sensors/img/figure_3/')
    img_dir = img_dir.joinpath('exp_and_sim_hist2d.svg')

    cov = np.array([[27.4241789, 2.48916841], [2.48916841, 21.80573026]])

    fluorophores = ['YFP', 'mKate', 'TFP']
    my_lvls = [0.34, 0.68]
    exp_data = pd.read_pickle('/mnt/data/Laboratorio/Imaging three sensors/2017-09-04_Images/' + filename + '.pandas')
    exp_data = exp_data.query('Content == "TNF alpha"')
    mask = [all([exp_data[fluo + '_good_der'][i] for fluo in fluorophores]) for i in exp_data.index]
    exp_times = exp_data.TFP_to_YFP.values[mask], exp_data.TFP_to_mKate.values[mask]
    exp_times = np.asarray(exp_times).T

    sim_mod_data = load_sim(filename='redVarCs_earm10_varligand_3_varrecep_3_varxiap_2')
    sim_mod_times = np.asarray((sim_mod_data.TFP_to_YFP.values, sim_mod_data.TFP_to_mKate.values)).T
    sim_mod_times += np.random.multivariate_normal([0, 0], cov*0.75, sim_mod_times.shape[0])

    sim_ear_data = load_sim(filename='redVarCs_earm10_nomodif')
    sim_ear_times = np.asarray((sim_ear_data.TFP_to_YFP.values, sim_ear_data.TFP_to_mKate.values)).T
    sim_ear_times += np.random.multivariate_normal([0, 0], cov*0.75, sim_ear_times.shape[0])

    df = pd.DataFrame(exp_times, columns=['TFP_to_YFP', 'TFP_to_mKate'])
    df['origin'] = 'exp'
    df_sim_mod = pd.DataFrame(sim_mod_times, columns=['TFP_to_YFP', 'TFP_to_mKate'])
    df_sim_mod['origin'] = 'sim_mod'
    df_sim_mod = df_sim_mod.dropna(axis=0, how='any')

    df_sim_ear = pd.DataFrame(sim_ear_times, columns=['TFP_to_YFP', 'TFP_to_mKate'])
    df_sim_ear['origin'] = 'sim_ear'
    df_sim_ear = df_sim_ear.dropna(axis=0, how='any')

    df = df.append(df_sim_ear)
    df = df.append(df_sim_mod)

    cmap_dict = {'sim_mod': 'Blues_d',
                 'sim_ear': 'Purples_d',
                 'exp': 'Oranges_d'}
    color_dict = {'sim_mod': 'b',
                  'sim_ear': [64 / 255, 2 / 255, 126 / 255],
                  'exp': [226 / 255, 85 / 255, 8 / 255]}
    color_dict = {'sim_mod': [[0 / 255, 0 / 255, 255 / 255], [140 / 255, 140 / 255, 255 / 255]],
                  'sim_ear': [[64 / 255, 2 / 255, 126 / 255], [146 / 255, 39 / 255, 252 / 255]],
                  'exp': [[226 / 255, 85 / 255, 8 / 255], [250 / 255, 155 / 255, 103 / 255]]}
    labels = {
        'sim_mod': 'Modified Model',
        'sim_ear': 'EARM V1.0',
        'exp': 'Observed'
    }
    g = sns.JointGrid("TFP_to_YFP", "TFP_to_mKate", df, xlim=(-45, 25), ylim=(-10, 20), size=3.3)
    for origin, this_df in df.groupby("origin"):
        if 'sim' in origin:
            sns.kdeplot(this_df["TFP_to_YFP"], shade=True, ax=g.ax_marg_x, color=color_dict[origin][0])
            sns.kdeplot(this_df["TFP_to_mKate"], shade=True, ax=g.ax_marg_y, vertical=True, color=color_dict[origin][0])
        elif origin == 'exp':
            sns.distplot(this_df["TFP_to_YFP"], bins=50, kde=False, ax=g.ax_marg_x, color=color_dict[origin][0], hist_kws={'normed': True})
            sns.distplot(this_df["TFP_to_mKate"], bins=50, kde=False, ax=g.ax_marg_y, vertical=True, color=color_dict[origin][0], hist_kws={'normed': True})

        if 'sim' in origin:
            cs = sns.kdeplot(this_df["TFP_to_YFP"], this_df["TFP_to_mKate"], colors=color_dict[origin], cmap=None, alpha=1,
                        levels=get_levels_for_kde(this_df["TFP_to_YFP"], this_df["TFP_to_mKate"], my_lvls),
                        ax=g.ax_joint, shade_lowest=False)
            cs.collections[-1].set_label(labels[origin])

        elif origin == 'exp':
            plt.sca(g.ax_joint)
            # times = (this_df["TFP_to_YFP"], this_df["TFP_to_mKate"])
            # times = (times[0][np.isfinite(times[0])], times[1][np.isfinite(times[1])])
            # hist, centers_x, centers_y = np.histogram2d(times[0], times[1], range=[[-30, 35], [-30, 35]],
            #                                             bins=np.arange(-30, 36,
            #                                                            5))  # mass_histogram(times, interp=True)
            # # centers_x = (centers_x[:-1] + centers_x[1:])/2
            # # centers_y = (centers_y[:-1] + centers_y[1:])/2
            #
            # hist = zoom(hist, 4, order=3)
            # max_count = np.max(hist)
            # hist = hist / max_count
            # plt.contour(hist.T, extent=(centers_x[0], centers_x[-1], centers_y[0], centers_y[-1]),
            #             levels=[0.25, 0.5, 0.75, 0.9],
            #             cmap=cmap_dict[origin])
            this_df = this_df.query('TFP_to_YFP > -30 and TFP_to_YFP < 35 and TFP_to_mKate > -30 and TFP_to_mKate < 35')
            cs = sns.kdeplot(this_df["TFP_to_YFP"], this_df["TFP_to_mKate"], colors=color_dict[origin], cmap=None, alpha=1, clip=((-30, 35), (-30, 35)),
                        levels=get_levels_for_kde(this_df["TFP_to_YFP"], this_df["TFP_to_mKate"], my_lvls),
                        ax=g.ax_joint, shade_lowest=False)
            cs.collections[-1].set_label(labels[origin])

    onecasp_df = pd.read_pickle('/mnt/data/Laboratorio/Imaging three sensors/2017-09-04_Images/OneCasp/OneCasp_derivations_order05_filtered_corrected.pandas')
    onecasp_df = onecasp_df.query('TFP_to_YFP < 20 and TFP_to_YFP >-20 and TFP_to_mKate < 20 and TFP_to_mKate > -20')
    cs = sns.kdeplot(onecasp_df["TFP_to_YFP"], onecasp_df["TFP_to_mKate"], colors='grey', cmap=None, linestyles='dashed', alpha=0.4,
                levels=get_levels_for_kde(onecasp_df["TFP_to_YFP"], onecasp_df["TFP_to_mKate"], my_lvls),
                ax=g.ax_joint)
    cs.collections[-1].set_label('Control')

    g.ax_joint.axvline(x=0, color='k', lw=1, ls='--', alpha=0.5)
    g.ax_joint.axhline(y=0, color='k', lw=1, ls='--', alpha=0.5)
    plt.legend(loc=3, framealpha=0)
    # plot_2dhist(sim_times)
    # plot_show()
    # plot_data(exp_times)
    # plt.scatter(exp_times[:, 0], exp_times[:, 1], alpha=0.1, color='r')
    g.set_axis_labels('$\Delta$t (Cas3-b, Cas9-y) (min.)', '$\Delta$t (Cas3-b, Cas8-r) (min.)')
    g.ax_marg_x.set_xlabel('')
    g.ax_marg_y.set_ylabel('')
    g.ax_marg_x.legend_.remove()
    g.ax_marg_y.legend_.remove()
    plt.tight_layout()
    plt.savefig(str(img_dir), format='svg')
    # plt.close()


def fig_sup_3(filename='2017-10-16_complex_noErode_order05_filtered_derived'):
    img_dir = pathlib.Path('/mnt/data/Laboratorio/Imaging three sensors/img/supplementary/')
    img_dir = img_dir.joinpath('sup_fig_3.svg')

    cov = np.array([[27.4241789, 2.48916841], [2.48916841, 21.80573026]])

    fluorophores = ['YFP', 'mKate', 'TFP']
    my_lvls = [0.34, 0.68]
    exp_data = pd.read_pickle('/mnt/data/Laboratorio/Imaging three sensors/2017-09-04_Images/' + filename + '.pandas')
    exp_data = exp_data.query('Content == "TNF alpha"')
    mask = [all([exp_data[fluo + '_good_der'][i] for fluo in fluorophores]) for i in exp_data.index]
    exp_times = exp_data.TFP_to_YFP.values[mask], exp_data.TFP_to_mKate.values[mask]
    exp_times = np.asarray(exp_times).T

    sim_mod_data = load_sim(filename='redVarCs_earm10_varligand_3_varrecep_3')
    sim_mod_times = np.asarray((sim_mod_data.TFP_to_YFP.values, sim_mod_data.TFP_to_mKate.values)).T
    sim_mod_times += np.random.multivariate_normal([0, 0], cov*0.75, sim_mod_times.shape[0])

    df = pd.DataFrame(exp_times, columns=['TFP_to_YFP', 'TFP_to_mKate'])
    df['origin'] = 'exp'
    df_sim_mod = pd.DataFrame(sim_mod_times, columns=['TFP_to_YFP', 'TFP_to_mKate'])
    df_sim_mod['origin'] = 'sim_mod'
    df_sim_mod = df_sim_mod.dropna(axis=0, how='any')

    df = df.append(df_sim_mod)

    color_dict = {'sim_mod': [[0 / 255, 0 / 255, 255 / 255], [140 / 255, 140 / 255, 255 / 255]],
                  'exp': [[226 / 255, 85 / 255, 8 / 255], [250 / 255, 155 / 255, 103 / 255]]}
    labels = {
        'sim_mod': 'Modified Model',
        'exp': 'Observed'
    }
    g = sns.JointGrid("TFP_to_YFP", "TFP_to_mKate", df, xlim=(-45, 25), ylim=(-10, 20), size=3.3)
    for origin, this_df in df.groupby("origin"):
        if 'sim' in origin:
            sns.kdeplot(this_df["TFP_to_YFP"], shade=True, ax=g.ax_marg_x, color=color_dict[origin][0])
            sns.kdeplot(this_df["TFP_to_mKate"], shade=True, ax=g.ax_marg_y, vertical=True, color=color_dict[origin][0])
        elif origin == 'exp':
            sns.distplot(this_df["TFP_to_YFP"], bins=50, kde=False, ax=g.ax_marg_x, color=color_dict[origin][0], hist_kws={'normed': True})
            sns.distplot(this_df["TFP_to_mKate"], bins=50, kde=False, ax=g.ax_marg_y, vertical=True, color=color_dict[origin][0], hist_kws={'normed': True})

        if 'sim' in origin:
            cs = sns.kdeplot(this_df["TFP_to_YFP"], this_df["TFP_to_mKate"], colors=color_dict[origin], cmap=None, alpha=1,
                        levels=get_levels_for_kde(this_df["TFP_to_YFP"], this_df["TFP_to_mKate"], my_lvls),
                        ax=g.ax_joint, shade_lowest=False)
            cs.collections[-1].set_label(labels[origin])

        elif origin == 'exp':
            plt.sca(g.ax_joint)
            # times = (this_df["TFP_to_YFP"], this_df["TFP_to_mKate"])
            # times = (times[0][np.isfinite(times[0])], times[1][np.isfinite(times[1])])
            # hist, centers_x, centers_y = np.histogram2d(times[0], times[1], range=[[-30, 35], [-30, 35]],
            #                                             bins=np.arange(-30, 36,
            #                                                            5))  # mass_histogram(times, interp=True)
            # # centers_x = (centers_x[:-1] + centers_x[1:])/2
            # # centers_y = (centers_y[:-1] + centers_y[1:])/2
            #
            # hist = zoom(hist, 4, order=3)
            # max_count = np.max(hist)
            # hist = hist / max_count
            # plt.contour(hist.T, extent=(centers_x[0], centers_x[-1], centers_y[0], centers_y[-1]),
            #             levels=[0.25, 0.5, 0.75, 0.9],
            #             cmap=cmap_dict[origin])
            this_df = this_df.query('TFP_to_YFP > -30 and TFP_to_YFP < 35 and TFP_to_mKate > -30 and TFP_to_mKate < 35')
            cs = sns.kdeplot(this_df["TFP_to_YFP"], this_df["TFP_to_mKate"], colors=color_dict[origin], cmap=None, alpha=1, clip=((-30, 35), (-30, 35)),
                        levels=get_levels_for_kde(this_df["TFP_to_YFP"], this_df["TFP_to_mKate"], my_lvls),
                        ax=g.ax_joint, shade_lowest=False)
            cs.collections[-1].set_label(labels[origin])

    onecasp_df = pd.read_pickle('/mnt/data/Laboratorio/Imaging three sensors/2017-09-04_Images/OneCasp/OneCasp_derivations_order05_filtered_corrected.pandas')
    onecasp_df = onecasp_df.query('TFP_to_YFP < 20 and TFP_to_YFP >-20 and TFP_to_mKate < 20 and TFP_to_mKate > -20')
    cs = sns.kdeplot(onecasp_df["TFP_to_YFP"], onecasp_df["TFP_to_mKate"], colors='grey', cmap=None, linestyles='dashed', alpha=0.4,
                levels=get_levels_for_kde(onecasp_df["TFP_to_YFP"], onecasp_df["TFP_to_mKate"], my_lvls),
                ax=g.ax_joint)
    cs.collections[-1].set_label('Control')

    g.ax_joint.axvline(x=0, color='k', lw=1, ls='--', alpha=0.5)
    g.ax_joint.axhline(y=0, color='k', lw=1, ls='--', alpha=0.5)
    plt.legend(loc=3, framealpha=0)
    # plot_2dhist(sim_times)
    # plot_show()
    # plot_data(exp_times)
    # plt.scatter(exp_times[:, 0], exp_times[:, 1], alpha=0.1, color='r')
    g.set_axis_labels('$\Delta$t (Cas3-b, Cas9-y) (min.)', '$\Delta$t (Cas3-b, Cas8-r) (min.)')
    g.ax_marg_x.set_xlabel('')
    g.ax_marg_y.set_ylabel('')
    g.ax_marg_x.legend_.remove()
    g.ax_marg_y.legend_.remove()
    plt.tight_layout()
    plt.savefig(str(img_dir), format='svg')
    # plt.close()


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


def full_img(timepoint, kind='r', fluorophores=['TFP', 'mKate', 'YFP']):
    imgs = {fluo: [] for fluo in fluorophores}
    img_dir = pathlib.Path('/mnt/data/Laboratorio/Imaging three sensors/AnisoTestData/results/Anisotropy/pos030/')
    for fluo in fluorophores:
        filename = '030_%03.f_%s_%s.tif' % (timepoint, kind, fluo)
        this_img_dir = img_dir.joinpath(filename)
        this_img = tif.TiffFile(str(this_img_dir)).asarray()
        imgs[fluo] = this_img

    return imgs


def fig_sup_1a():
    sav_dir = pathlib.Path('/mnt/data/Laboratorio/Imaging three sensors/img/supplementary/')
    sav_dir = sav_dir.joinpath('timelapse_example.svg')

    fig = plt.figure(figsize=(6.4, 2.5))
    axs = plt.gca()
    axs.axis('off')

    imgs = {}
    imgs['ini'] = full_img(2)
    imgs['end'] = full_img(51)

    pos = {}
    pos['ini'] = np.asarray([-.55, .1, 1, 1])
    pos['end'] = np.asarray([.15, .1, 1, 1])

    rect = {}
    rect['xy_ini'] = (1120, 260)
    rect['height_ini'] = 180
    rect['width_ini'] = 180
    rect['xy_end'] = (1170, 270)
    rect['height_end'] = 110
    rect['width_end'] = 150

    for this_time in ['ini', 'end']:
        pos_dif = np.asarray([-.08, -.2, 0, 0])

        for k, fluo in enumerate(fluorophores):
            inset_axes(axs, width='40%', height='70%', bbox_to_anchor=pos[this_time] + k * pos_dif,
                       bbox_transform=axs.transAxes)
            axs_in = plt.gca()
            axs_in.tick_params(bottom='False', left='False', labelbottom='False', labelleft='False')
            img = np.nan_to_num(imgs[this_time][fluo])
            cmap = plt.cm.plasma
            cmap = cmap.set_under('black')
            axs_in.imshow(img, cmap='plasma', vmin=0.17, vmax=0.36)
            rect_plot = matplotlib.patches.Rectangle(rect['xy_' + this_time], rect['width_' + this_time],
                                                     rect['height_' + this_time], fc=(1, 1, 0, 0), ec=(1, 0, 0, 1),
                                                     lw=3)
            axs_in.add_patch(rect_plot)
            if k == 2:
                # Add scalebar
                # 1um <-> 4.9751
                scalebar = AnchoredSizeBar(axs_in.transData,
                                           200, '          ', 'lower right',
                                           pad=0.1,
                                           color='white',
                                           frameon=False,
                                           size_vertical=3,
                                           label_top=True)  # ,
                # fontproperties=fontprops)

                axs_in.add_artist(scalebar)

    plt.savefig(str(sav_dir), format='svg')


def fig_sup_1b(fluorophores=['TFP', 'YFP', 'mKate']):
    sav_dir = pathlib.Path('/mnt/data/Laboratorio/Imaging three sensors/img/supplementary/')
    sav_dir = sav_dir.joinpath('sensors_example.svg')

    mask_dir = pathlib.Path('/mnt/data/Laboratorio/Imaging three sensors/AnisoTestData/results/Regions/')
    mask = {}
    for t in [4, 50]:
        filename = '030_%03.f_%s_%s.tif' % (t, 'f', 'YFP')
        this_mask_dir = mask_dir.joinpath(filename)
        this_mask = tif.TiffFile(str(this_mask_dir)).asarray()
        mask[t] = this_mask

    mask[4] = mask[4] != 11
    mask[50] = mask[50] != 9

    imgs_r = {}
    imgs_r['ini'] = full_img(4)
    imgs_r['end'] = full_img(50)
    imgs_f = {}
    imgs_f['ini'] = full_img(4)
    imgs_f['end'] = full_img(50)
    for fluo in fluorophores:
        imgs_r['ini'][fluo][[mask[4]]] = 0
        imgs_r['end'][fluo][[mask[50]]] = 0
        imgs_f['ini'][fluo][[mask[4]]] = 0
        imgs_f['end'][fluo][[mask[50]]] = 0

    rect = {}
    rect['xy_ini'] = (1120, 260)
    rect['height_ini'] = 180
    rect['width_ini'] = 180
    rect['xy_end'] = (1200, 270)
    rect['height_end'] = 110
    rect['width_end'] = 110

    titles = {'TFP': 'x-b', 'mKate': 'x-r', 'YFP': 'x-y'}
    Colors = {'TFP': plt.cm.Blues_r, 'mKate': plt.cm.Reds_r, 'YFP': plt.cm.Greens_r}

    f_min = 0.06
    f_max = 0.5
    r_min = 0.17
    r_max = 0.36

    fig, axs = plt.subplots(4, 3, figsize=(6.4, 6.4))
    # plt.gcf().subplots_adjust(left=0.01, right=.9)

    for n, fluo in enumerate(fluorophores):
        for j, this_time in enumerate(['ini', 'end']):
            j *= 2
            img = imgs_f[this_time][fluo]
            crop_loc = (rect['xy_' + this_time][0],
                        rect['xy_' + this_time][0] + rect['width_' + this_time],
                        rect['xy_' + this_time][1],
                        rect['xy_' + this_time][1] + rect['height_' + this_time])
            img = img[crop_loc[2]:crop_loc[3], crop_loc[0]:crop_loc[1]]
            img = np.nan_to_num(img)
            cmap = Colors[fluo]
            cmap.set_under(color='black')
            im_f = axs[j][n].imshow(img, vmin=f_min, vmax=f_max, cmap=cmap)

            if j == 0:
                axs[j][n].set_title(titles[fluo])
            if n == 0:
                axs[j][n].set_ylabel('Fluorescence\n Intensity (a.u.)')
            axs[j][n].tick_params(bottom='False', left='False', labelbottom='False', labelleft='False')

            # Add colorbar
            cax = inset_axes(axs[j][n],
                             width="7%",
                             height="100%",
                             bbox_transform=axs[j][n].transAxes,
                             bbox_to_anchor=(0.2, .05, 1, 1),
                             loc=1)
            norm = matplotlib.colors.Normalize(vmin=f_min, vmax=f_max)
            cb1 = matplotlib.colorbar.ColorbarBase(cax,
                                                   cmap=cmap, norm=norm,
                                                   orientation='vertical')
            cb1.set_ticks([0.1, 0.2, 0.3, 0.4, 0.5])

            j += 1
            img = imgs_r[this_time][fluo]
            crop_loc = (rect['xy_' + this_time][0],
                        rect['xy_' + this_time][0] + rect['width_' + this_time],
                        rect['xy_' + this_time][1],
                        rect['xy_' + this_time][1] + rect['height_' + this_time])
            img = img[crop_loc[2]:crop_loc[3], crop_loc[0]:crop_loc[1]]
            img = np.nan_to_num(img)
            cmap = plt.cm.plasma
            cmap.set_under(color='black')
            im_r = axs[j][n].imshow(img, vmin=r_min, vmax=r_max, cmap='plasma')
            if n == 0:
                axs[j][n].set_ylabel('Anisotropy')
            axs[j][n].tick_params(bottom='False', left='False', labelbottom='False', labelleft='False')

            # Add colorbar
            cax = inset_axes(axs[j][n],
                             width="7%",
                             height="100%",
                             bbox_transform=axs[j][n].transAxes,
                             bbox_to_anchor=(0.2, .05, 1, 1),
                             loc=1)
            norm = matplotlib.colors.Normalize(vmin=r_min, vmax=r_max)
            cb1 = matplotlib.colorbar.ColorbarBase(cax,
                                                   cmap='plasma', norm=norm,
                                                   orientation='vertical')
            cb1.set_ticks([0.20, 0.23, 0.26, 0.29, 0.33])

    # Add scalebar
    # 1um <-> 4.9751
    scalebar = AnchoredSizeBar(axs[0][0].transData,
                               50, '          ', 'lower right',
                               pad=0.1,
                               color='white',
                               frameon=False,
                               size_vertical=2,
                               label_top=True) #  ,
                               # fontproperties=fontprops)

    axs[0][0].add_artist(scalebar)

    # plt.subplots_adjust(hspace=-.1, wspace=0.1)
    plt.savefig(str(sav_dir), format='svg')


def plot_anis_in_bars(ax, data, constructs, bar_width, color=None):
    if color is None:
        color_mono = 'b'
        color_di = 'r'
    elif color == 'yellows':
        color_mono = (175 / 255, 192 / 255, 69 / 255)
        color_di = (255 / 255, 230 / 255, 128 / 255)
    elif color == 'blues':
        color_mono = (82 / 255, 186 / 255, 221 / 255)
        color_di = (170 / 255, 238 / 255, 255 / 255)
    elif color == 'reds':
        color_mono = (214 / 255, 92 / 255, 61 / 255)
        color_di = (224 / 255, 131 / 255, 108 / 255)

    index = {'monomer': [], 'dimer': []}
    yticklabels = {'monomer': [], 'dimer': []}
    vals = {'monomer': [], 'dimer': []}

    for n, group in enumerate(constructs[-1::-1]):
        for i, this_construct in enumerate(group[-1::-1]):
            this_index = n + i * bar_width
            this_ani = data[data.construct == this_construct].anisotropy.values[0]
            this_mer = data[data.construct == this_construct].mer.values[0]

            index[this_mer].append(this_index)
            yticklabels[this_mer].append(this_construct)
            vals[this_mer].append(this_ani)

    rects1 = ax.barh(index['dimer'], vals['dimer'], bar_width,
                     color=color_di, edgecolor='k',
                     label='dimer')

    rects2 = ax.barh(index['monomer'], vals['monomer'], bar_width,
                     color=color_mono, edgecolor='k',
                     label='monomer')

    for rects in [rects1, rects2]:
        for bar in rects:
            bar.set_edgecolor("k")
            bar.set_linewidth(1.5)

    ax.set_xlim((0.15, 0.33))
    yticks = np.concatenate((index['monomer'], index['dimer']))
    ax.set_yticks(yticks)
    yticklabels = np.concatenate((yticklabels['monomer'], yticklabels['dimer']))
    ax.set_yticklabels(yticklabels, ha='right')
    ax.legend(loc=2)


def fig_sup_2():
    sav_dir = pathlib.Path('/mnt/data/Laboratorio/Imaging three sensors/img/supplementary/')
    sav_dir = sav_dir.joinpath('constructs_anisotropy.svg')

    data_dir = pathlib.Path('/mnt/data/Laboratorio/Imaging three sensors/constructs/')
    data_dir = data_dir.joinpath('constructs_anisotropy_org.pandas')
    data = pd.read_pickle(str(data_dir))

    hetero_blues = [('TagBFP', 'EBFP2', 'TagBFP-EBFP2 (DAPI)', 'TagBFP-EBFP2 (CFP)'),
                    ('TagBFP', 'Cerulean', 'TagBFP-Cerulean (DAPI)', 'TagBFP-Cerulean (CFP)'),
                    # ('Cerulean', 'TagBFP', 'Cerulean-TagBFP (DAPI)', 'Cerulean-TagBFP (CFP)'),
                    ('TagBFP', 'TFP', 'TagBFP-TFP (DAPI)', 'TagBFP-TFP (CFP)')]
                    # ('TFP', 'TagBFP', 'TFP-TagBFP (DAPI)', 'TFP-TagBFP (CFP)')]
    yellows = [('TFP', 'TFP-TFP'), ('EGFP', 'EGFP-EGFP'), ('mCitrine', 'mCitrine-mCitrine')]
    reds = [('TagRFP', 'TagRFP-TagRFP'), ('mCherry', 'mCherry-mCherry'), ('mKate2', 'mKate2-mKate2'),
            ('mCherry', 'mKate2', 'mCherry-mKate2')]

    fig, axs = plt.subplots(3, 1, sharex=True, figsize=(3.3, 8.5), gridspec_kw={'height_ratios': [5, 3, 4]})

    plot_anis_in_bars(axs[0], data, hetero_blues, 0.20, color='blues')
    plot_anis_in_bars(axs[1], data, yellows, 0.33, color='yellows')
    plot_anis_in_bars(axs[2], data, reds, 0.28, color='reds')
    axs[0].set_title('Anisotropy')

    fig.tight_layout()
    plt.subplots_adjust(hspace=.0)
    plt.savefig(str(sav_dir), format='svg')
