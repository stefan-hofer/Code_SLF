import glob
import xarray as xr
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
import matplotlib.path as mpath
import cartopy.crs as ccrs
import cartopy.feature
import matplotlib.gridspec as gridspec
import xesmf as xe
import datetime as dt
import matplotlib.pyplot as plt

import cartopy.feature as feat

folder = ['/projects/NS9252K/noresm/cases/WP4_shofer/n.n202.NSSP585frc2.f09_tn14.ssp585.001_global/atm/hist/COSP/cloud_feedbacks/',
          '/projects/NS9252K/noresm/cases/WP4_shofer/n.n202.NSSP585frc2.f09_tn14.ssp585.002_NH/atm/hist/COSP/cloud_feedbacks/',
          '/projects/NS9252K/noresm/cases/WP4_shofer/n.n202.NSSP585frc2.f09_tn14.ssp585.003_SH/atm/hist/COSP/cloud_feedbacks/']


def weighted_mean(ds):
    # Create weights
    weights_n = np.cos(np.deg2rad(ds.lat))
    weighted = ds.weighted(weights_n)
    mean_arctic = weighted.mean(dim=['lat', 'lon'])

    return mean_arctic


def preprocess(ds):
    ds['year'] = pd.to_datetime(str(ds.year.values))
    return ds


def cld_feedback_df(ds_list, area=['Global', 'NH', 'SH']):
    # Extract all the cloud feedbacks
    df_list = []
    for i in range(3):
        try:
            test = weighted_mean(ds_list[i]).mean(dim='month')
        except AttributeError:
            test = ds_list[i].mean(dim='month')
        df = test.isel(year=[-2, -1]).to_dataframe()
        df['year'] = df.index
        df['Area'] = area[i]
        # df.melt reshapes the dataframe for plotting with seaborn
        df_new = df.melt(id_vars=['year', 'Area'])

        df_list.append(df_new)
        all_dfs = pd.concat(df_list)

    return all_dfs


# Lists where 3x3 feedback outputs are stored: SW, LW and NET for all runs
list_SW = []
list_LW = []
list_net = []

names = ['Global', 'NH', 'SH', 'Ctrl']
colors = ['black', '#1f78b4', '#b2df8a']
i = 0
fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(8, 5))
for f in folder:
    list_files_SW = sorted(glob.glob(f + 'SW_cloud_feedbacks_ALL*.nc'))
    list_files_LW = sorted(glob.glob(f + 'LW_cloud_feedbacks_ALL*.nc'))

    ds_SW = xr.open_mfdataset(
        list_files_SW, concat_dim='year', combine='nested', preprocess=preprocess)
    ds_LW = xr.open_mfdataset(
        list_files_LW, concat_dim='year', combine='nested', preprocess=preprocess)
    net = xr.open_dataset(f + 'NET_cloud_feedbacks_ALL_combined.nc')

    weighted_mean(ds_SW.SWcld_tot).mean(dim=['month']
                                        ).isel(year=[-2, -1]).plot(label=names[i] + '_SW', linestyle='dotted', color=colors[i], ax=axs, marker='*')
    weighted_mean(ds_LW.LWcld_tot).mean(dim=['month']
                                        ).isel(year=[-2, -1]).plot(label=names[i] + '_LW', linestyle='dashed', color=colors[i], ax=axs, marker='*')

    # net = ds_SW.SWcld_tot.mean(
    #     dim=['lat', 'lon', 'month']) + ds_LW.LWcld_tot.mean(dim=['lat', 'lon', 'month'])
    weighted_mean(net['NETcld_tot']).mean(dim=['month']
                                          ).isel(year=[-2, -1]).plot(label=names[i] + '_Net',
                                                                     linestyle='solid', lw=3, color=colors[i], ax=axs, marker='*')

    list_SW.append(ds_SW)
    list_LW.append(ds_LW)
    list_net.append(net)

    # # extract all net feedbacks
    # new = xr.Dataset()
    # new['NETcld_tot'] = ds_SW.SWcld_tot + ds_LW.LWcld_tot
    # new['NETcld_amt'] = ds_SW.SWcld_amt + ds_LW.LWcld_amt
    # new['NETcld_alt'] = ds_SW.SWcld_alt + ds_LW.LWcld_alt
    # new['NETcld_tau'] = ds_SW.SWcld_tau + ds_LW.LWcld_tau
    # new['NETcld_err'] = ds_SW.SWcld_err + ds_LW.LWcld_err
    #
    # new.to_netcdf(f + 'NET_cloud_feedbacks_ALL_combined.nc')

    i += 1

plt.legend()
fig.tight_layout()
sns.despine()
fig.savefig(
    '/projects/NS9252K/noresm/cases/WP4_shofer/Figures/cloud_feedbacks_ts.png')

# Extract all the averages and compare in a DataFrame


net_dfs = cld_feedback_df(list_net)
sw_dfs = cld_feedback_df(list_SW)
lw_dfs = cld_feedback_df(list_LW)

titles = ['NET', 'SW', 'LW']
ticks = ['Total', 'Amt', 'Alt', 'Tau', 'Err']
palette = [sns.color_palette("rocket")[0], sns.color_palette(
    "rocket")[-3], sns.color_palette("rocket")[-1]]
plt.close('all')
fig, axs = plt.subplots(nrows=1, ncols=3, sharey='row',
                        squeeze=True, figsize=(10, 4))

sns.barplot(x='variable', y='value', hue='Area',
            data=net_dfs, palette=palette, ax=axs[0], ci=None)
sns.barplot(x='variable', y='value', hue='Area',
            data=sw_dfs, palette=palette, ax=axs[1], ci=None)
sns.barplot(x='variable', y='value', hue='Area',
            data=lw_dfs, palette=palette, ax=axs[2], ci=None)

for i in range(3):
    axs[i].set_title(titles[i])
    axs[i].set_xticklabels(ticks)
    axs[i].legend(frameon=False)
    axs[i].set_ylabel(r'$\lambda (Wm^{-2}K^{-1})$', fontsize=13)
    axs[i].set_xlabel('Cloud Feedback', fontsize=13)

sns.despine()
fig.tight_layout()

fig.savefig(
    '/projects/NS9252K/noresm/cases/WP4_shofer/Figures/cloud_feedbacks_bar_2050.png')


# =======================================================================================
# =============== LATITUDE CROSS SECTIONS ===============================================
# =======================================================================================
names = ['Global', 'NH', 'SH']
vars = ['NETcld_tot', 'NET']

plt.close('all')
fig, axs = plt.subplots(nrows=1, ncols=3, squeeze=True,
                        sharey=True, figsize=(12, 7))
for i in range(3):

    list_net[i]['NETcld_tot'].isel(
        year=-1).mean(dim=['lon', 'month']).plot(label=names[i], ax=axs[0])
    list_SW[i]['SWcld_tot'].isel(
        year=-1).mean(dim=['lon', 'month']).plot(label=names[i], ax=axs[1])
    list_LW[i]['LWcld_tot'].isel(
        year=-1).mean(dim=['lon', 'month']).plot(label=names[i], ax=axs[2])

sns.despine()
plt.legend()
fig.tight_layout()

fig.savefig(
    '/projects/NS9252K/noresm/cases/WP4_shofer/Figures/cloud_feedbacks_crosssec.png')

# =======================================================================
# Latitude plots (N.ET, S.ET, Global, Arctic)
# ======================================================================
lat_min_list = [-90, 30, -90]
lat_max_list = [90, 90, -30]

dict_all = {}

names_list = ['Global', 'N-ET', 'S-ET']
for i in range(3):
    list_net = []
    list_SW = []
    list_LW = []
    for f in folder:
        list_files_SW = sorted(glob.glob(f + 'SW_cloud_feedbacks_ALL*.nc'))
        list_files_LW = sorted(glob.glob(f + 'LW_cloud_feedbacks_ALL*.nc'))

        ds_SW = xr.open_mfdataset(
            list_files_SW, concat_dim='year', combine='nested', preprocess=preprocess)
        ds_LW = xr.open_mfdataset(
            list_files_LW, concat_dim='year', combine='nested', preprocess=preprocess)
        net = xr.open_dataset(f + 'NET_cloud_feedbacks_ALL_combined.nc')

        ds_SW = weighted_mean(ds_SW.sel(
            lat=slice(lat_min_list[i], lat_max_list[i])))
        ds_LW = weighted_mean(ds_LW.sel(
            lat=slice(lat_min_list[i], lat_max_list[i])))
        net = weighted_mean(net.sel(
            lat=slice(lat_min_list[i], lat_max_list[i])))

        list_SW.append(ds_SW)
        list_LW.append(ds_LW)
        list_net.append(net)

    net_dfs = cld_feedback_df(list_net)
    sw_dfs = cld_feedback_df(list_SW)
    lw_dfs = cld_feedback_df(list_LW)

    list_all = [net_dfs, sw_dfs, lw_dfs]

    dict_all[names_list[i]] = list_all

names_list = ['Global', 'N-ET', 'S-ET']
titles = ['NET', 'SW', 'LW']
ticks = ['Total', 'Amt', 'Alt', 'Tau', 'Err']
palette = [sns.color_palette("rocket")[0], sns.color_palette(
    "rocket")[-3], sns.color_palette("rocket")[-1]]


# Plot 4x3 Bar plots
# NOTE NEED TO REMOVE ARCTIC
# ==========================
plt.close('all')
fig, axs = plt.subplots(nrows=4, ncols=3, sharey='all',
                        squeeze=True, figsize=(10, 10))

i = 0
for name in names_list:

    sns.barplot(x='variable', y='value', hue='Area',
                data=dict_all[name][0], palette=palette, ax=axs[i][0], ci=None)
    sns.barplot(x='variable', y='value', hue='Area',
                data=dict_all[name][1], palette=palette, ax=axs[i][1], ci=None)
    sns.barplot(x='variable', y='value', hue='Area',
                data=dict_all[name][2], palette=palette, ax=axs[i][2], ci=None)
    i += 1

label_arr = [['A', 'B', 'C'],
             ['D', 'E', 'F'],
             ['G', 'H', 'I'],
             ['J', 'K', 'L']]
for i in range(4):
    for j in range(3):
        axs[0][j].set_title(titles[j], fontsize=22)
        axs[i][j].set_xticklabels(ticks)
        axs[i][j].get_legend().remove()
        axs[i][0].legend(frameon=False)
        axs[i][0].set_ylabel(r'$\lambda (Wm^{-2}K^{-1})$', fontsize=13)
        if j > 0:
            axs[i][j].set_ylabel('')
        # axs[i][j].set_xlabel('Cloud Feedback', fontsize=13)
        axs[i][j].set_xlabel('')
        axs[i][j].text(-0.2, 0.8, label_arr[i][j], fontsize=26, weight='bold')

    axs[i][0].text(0.5, 0.05, names_list[i], fontsize=30,
                   transform=axs[i][0].transAxes, alpha=0.6)


sns.despine()
fig.tight_layout()

fig.savefig(
    '/projects/NS9252K/noresm/cases/WP4_shofer/Figures/cloud_feedbacks_bar_4x3.png')

