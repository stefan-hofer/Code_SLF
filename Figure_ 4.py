import pandas as pd
import xarray as xr
import numpy as np
import seaborn as sns
import cartopy.crs as ccrs
import cartopy.feature as feat

folder = '/projects/NS9252K/noresm/cases/WP4_shofer/final/'
folder = '/home/shofer/Dropbox/Academic/Data/SLF_paper/'

ds_one = xr.open_dataset(
    folder + 'RESTOM_final_001.nc')
ds_two = xr.open_dataset(
    folder + 'RESTOM_final_002.nc')
ds_three = xr.open_dataset(
    folder + 'RESTOM_final_003.nc')
ds_four = xr.open_dataset(
    folder + 'RESTOM_final_004_ctrl.nc')
# ds_five = xr.open_dataset(
#     '/projects/NS9252K/noresm/cases/WP4_shofer/final/HEIKO_RESTOM_final_005_ctrl.nc')


# FUNCTIONS NEEDED FOR COMPUTATION
def weighted_mean(ds):
    # Create weights
    weights_n = np.cos(np.deg2rad(ds.lat))
    weighted = ds.weighted(weights_n)
    mean_arctic = weighted.mean(dim=['lat', 'lon'])

    return mean_arctic


list_ds = [ds_one, ds_two, ds_three, ds_four]


def restom_vs_tanom(ds):
    roll = ds.rolling(time=20, center=True, min_periods=3).mean()
    cum = (roll['RESTOM_w'].cumsum(dim='time') - roll['RESTOM_w'].cumsum(dim='time').sel(time='2014')
           [0]).sel(time=slice('2015', '2100'))
    SSP_warming = (roll['TREFHT_w'] - roll['TREFHT_w'].sel(time='2014')
                   [0]).sel(time=slice('2015', '2100'))

    return cum, SSP_warming


def restom_vs_tanom_weighted(ds, lat_min=-90, lat_max=90):
    ds_new = weighted_mean(ds.sel(lat=slice(lat_min, lat_max)))
    roll = ds_new.rolling(time=20, center=True, min_periods=10).mean()
    cum = (roll['RESTOM'].cumsum(dim='time') - roll['RESTOM'].cumsum(dim='time').sel(time='2014')
           [0]).sel(time=slice('2015', '2100'))
    SSP_warming = (roll['TREFHT'] - roll['TREFHT'].sel(time='2014')
                   [0]).sel(time=slice('2015', '2100'))

    return cum, SSP_warming


def find_nearest(a, a0):

    return a[np.abs(a - a0).argmin()]


# ============================================
# ============== 4 x 1 Plot ==================
# ============================================
labels = ['A', 'B', 'C', 'D']

lat_min_list = [-90, 60, -90, 30]
lat_max_list = [90, 90, -30, 90]
names_list = ['Global', 'N-ET', 'S-ET', 'CTRL']
# For in figure labels
text_list = ['Global', 'Arctic', 'S-ET', 'N-ET']
plt.close('all')
fig, axs = plt.subplots(2, 2, figsize=(11, 11), squeeze=True)
axs = fig.axes
for j in range(4):

    cum_list = []
    cum_list_global = []
    warming_list = []
    for ds in list_ds:
        cum, SSP_warming = restom_vs_tanom_weighted(
            ds, lat_min=lat_min_list[j], lat_max=lat_max_list[j])
        cum_global, SSP_warming_global = restom_vs_tanom(ds)
        cum_list.append(cum)
        warming_list.append(SSP_warming)
        # Global needed to plot Arctic T vs global RESTOM
        cum_list_global.append(cum_global)

    for i in range(4):
        axs[j].scatter(cum_list_global[i], warming_list[i],
                       label=names_list[i])
    axs[j].legend(frameon=False, fontsize=11)
    axs[j].set_xlabel(r'Cumulative RESTOM $(Wm^{-2})$', fontsize=22)
    axs[j].set_ylabel(r'$\Delta$ T: 2015-2100 ($^{\circ}$C)', fontsize=22)
    axs[j].xaxis.set_tick_params(labelsize=11)
    axs[j].yaxis.set_tick_params(labelsize=11)
    axs[j].text(0.5, 0.05, text_list[j], fontsize=30,
                transform=axs[j].transAxes, alpha=0.6)
    axs[j].text(0.05, 0.7, labels[j], fontsize=30,
                weight='bold', transform=axs[j].transAxes)

sns.despine()
fig.tight_layout()
fig.savefig(
    '/projects/NS9252K/noresm/cases/WP4_shofer/Figures/Sensitivity_global_4x4.png')

# ==================================================================
# =================== 3 x 1 Plot ===================================
# ==================================================================
labels = ['A', 'B', 'C']

lat_min_list = [-90, 30, -90]
lat_max_list = [90, 90, -30]
names_list = ['Global', 'N-ET', 'S-ET', 'CTRL']
# For in figure labels
text_list = ['Global', 'N-ET', 'S-ET']
plt.close('all')
fig, axs = plt.subplots(1, 3, figsize=(12, 5), squeeze=True)
axs = fig.axes
df_list = []
for j in range(3):

    cum_list = []
    cum_list_global = []
    warming_list = []
    for ds in list_ds:
        cum, SSP_warming = restom_vs_tanom_weighted(
            ds, lat_min=lat_min_list[j], lat_max=lat_max_list[j])
        cum_global, SSP_warming_global = restom_vs_tanom(ds)
        cum_list.append(cum)
        warming_list.append(SSP_warming)
        # Global needed to plot Arctic T vs global RESTOM
        cum_list_global.append(cum_global)

    for i in range(4):
        axs[j].scatter(cum_list_global[i], warming_list[i],
                       label=names_list[i])
    axs[j].legend(frameon=False, fontsize=11)
    axs[j].set_xlabel(r'Cumulative RESTOM $(Wm^{-2})$', fontsize=14)
    axs[j].set_ylabel(r'$\Delta$ T: 2015-2100 ($^{\circ}$C)', fontsize=14)
    axs[j].xaxis.set_tick_params(labelsize=11)
    axs[j].yaxis.set_tick_params(labelsize=11)
    axs[j].text(0.5, 0.05, text_list[j], fontsize=30,
                transform=axs[j].transAxes, alpha=0.6)
    axs[j].text(-0.15, 1, labels[j], fontsize=30,
                weight='bold', transform=axs[j].transAxes)

    # This is to extract warming on different forcing levels
    toa_values = np.array([find_nearest(cum_list_global[k], imbalance).values for k in range(4)
                           for imbalance in [25, 50, 75, 100]])

    toa_years = np.array([find_nearest(cum_list_global[k], imbalance).coords["time"].values.astype("datetime64[ns]") for k in range(4)
                          for imbalance in [25, 50, 75, 100]])

    # This can be used to index the warming_list data arrays
    toa_years_test = np.array([find_nearest(cum_list_global[k], imbalance).coords["time"].values for k in range(4)
                               for imbalance in [25, 50, 75, 100]])

    warming_extracted = []
    l = 0
    while l <= 3:
        for m in range(4):
            for n in range(4):
                warming = warming_list[m].sel(
                    time=toa_years_test[n + l * 4]).values
                warming_extracted.append(warming)
                if n == 3:
                    l += 1

    data = {"TOA_Imbalance": toa_values, 'Warming': np.array(warming_extracted), "TOA_YEARS": toa_years,
            "Simulation": 4 * ["Global"] + 4 * ["N-ET"] + 4 * ["S-ET"] + 4 * ["Control"]}

    df = pd.DataFrame(data)

    df_list.append(df)

sns.despine()
fig.tight_layout()
fig.savefig(
    '/projects/NS9252K/noresm/cases/WP4_shofer/Figures/Sensitivity_global_3x1.png', dpi=300)
fig.savefig(
    '/projects/NS9252K/noresm/cases/WP4_shofer/Figures/Sensitivity_global_3x1.pdf', format='PDF')
