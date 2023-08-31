import numpy as np
import seaborn as sns
import xarray as xr
import matplotlib.pyplot as plt

# CALIOP DATA
folder = '/tos-project2/NS9600K/shofer/'
data_folder = 'caliop_olimpia_new/netcdf_format/'

files_ann = ['bulk_slfs_annual.nc', 'ct_slfs_annual.nc']
files_seasonal = ['bulk_slfs_seasonal.nc', 'ct_slfs_seasonal.nc']

seasonal_bulk = xr.open_dataset(folder + data_folder + files_seasonal[0])
seasonal_ct = xr.open_dataset(folder + data_folder + files_seasonal[1])

ann_bulk = xr.open_dataset(folder + data_folder + files_ann[0])
ann_ct = xr.open_dataset(folder + data_folder + files_ann[1])

# =======================================================
# FUNCTIONS
# =======================================================


def arctic_slf_weighted(ds, s_bnd=66.6, n_bnd=90, ss_bnd=None, nn_bnd=None):
    '''Computes the mean of the CALIOP SLF in the Arctic,
    weighted by the size of the grid cell.
    '''
    ds_arctic = ds.sel(lat=slice(s_bnd, n_bnd))

    weighted_arctic = ds_arctic.weighted(ds_arctic.cell_weight)
    mean_arctic = weighted_arctic.mean(dim=['lat', 'lon']).SLF

    return mean_arctic


def plot_slf_iso(ds, fig=False, axs=False):
    if axs is False:
        fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(4, 6))
    else:
        fig = fig
        axs = axs
    xr.plot.line(ds * 100, y='isotherm', yincrease=False, ax=axs)
    sns.despine()
    fig.tight_layout()

    return fig, axs


plt.close('all')
colors = sns.color_palette("colorblind", 4)
fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(
    5, 11), sharex='col', sharey='all')
axes = axs.flatten()
# ============= FIGURES ==========================
# This is the CALIOP data globally
global_bulk = arctic_slf_weighted(ann_bulk * 100, s_bnd=-82, n_bnd=82)
global_ct = arctic_slf_weighted(ann_ct * 100, s_bnd=-82, n_bnd=82)

global_bulk_seasonal = arctic_slf_weighted(
    seasonal_bulk * 100, s_bnd=-82, n_bnd=82)
global_ct_seasonal = arctic_slf_weighted(
    seasonal_ct * 100, s_bnd=-82, n_bnd=82)

season = ['DJF', 'MAM', 'JJA', 'SON']
for i in range(4):
    global_ct_seasonal.isel(time=i).plot(
        label=season[i] + ' Top', color=colors[i], ls='--', ax=axes[0], y='isotherm', yincrease=False)
    global_bulk_seasonal.isel(time=i).plot(
        label=season[i] + ' Bulk', ax=axes[0], color=colors[i], y='isotherm', yincrease=False)

# ============= FIGURES ==========================
# This is the CALIOP data for the northern extratropics
global_bulk = arctic_slf_weighted(ann_bulk * 100, s_bnd=30, n_bnd=82)
global_ct = arctic_slf_weighted(ann_ct * 100, s_bnd=30, n_bnd=82)

global_bulk_seasonal = arctic_slf_weighted(
    seasonal_bulk * 100, s_bnd=30, n_bnd=82)
global_ct_seasonal = arctic_slf_weighted(
    seasonal_ct * 100, s_bnd=30, n_bnd=82)

season = ['DJF', 'MAM', 'JJA', 'SON']
for i in range(4):
    global_ct_seasonal.isel(time=i).plot(
        label=season[i] + ' Top', color=colors[i], ls='--', ax=axes[1], y='isotherm', yincrease=False)
    global_bulk_seasonal.isel(time=i).plot(
        label=season[i] + ' Bulk', ax=axes[1], color=colors[i], y='isotherm', yincrease=False)


# ============= FIGURES ==========================
# This is the CALIOP data for the southern extratropics
global_bulk = arctic_slf_weighted(ann_bulk * 100, s_bnd=-82, n_bnd=-30)
global_ct = arctic_slf_weighted(ann_ct * 100, s_bnd=-82, n_bnd=-30)

global_bulk_seasonal = arctic_slf_weighted(
    seasonal_bulk * 100, s_bnd=-82, n_bnd=-30)
global_ct_seasonal = arctic_slf_weighted(
    seasonal_ct * 100, s_bnd=-82, n_bnd=-30)

season = ['DJF', 'MAM', 'JJA', 'SON']
for i in range(4):
    global_ct_seasonal.isel(time=i).plot(
        label=season[i] + ' Top', color=colors[i], ls='--', ax=axes[2], y='isotherm', yincrease=False)
    global_bulk_seasonal.isel(time=i).plot(
        label=season[i] + ' Bulk', ax=axes[2], color=colors[i], y='isotherm', yincrease=False)


axes[0].legend(frameon=False, fontsize=9)

names = ['Global', 'N-ET', 'S-ET']
labels = ['A', 'B', 'C']
for j in range(3):
    axs[j].set_ylabel(r'Isotherm ($^{\circ}$C)', fontsize=14)
    axs[j].set_xlabel('SLF (%)', fontsize=14)
    axs[j].xaxis.set_tick_params(labelsize=11)
    axs[j].yaxis.set_tick_params(labelsize=11)
    axs[j].text(0.4, 0.05, names[j], fontsize=30,
                transform=axs[j].transAxes, alpha=0.6)
    axs[j].text(0.9, 0.9, labels[j], fontsize=30,
                weight='bold', transform=axs[j].transAxes)
    axs[j].set_title('')

sns.despine()
fig.tight_layout()

fig.savefig(
    '/projects/NS9252K/noresm/cases/WP4_shofer/Figures/SLF_seasonality_3x1.png', dpi=300)
fig.savefig(
    '/projects/NS9252K/noresm/cases/WP4_shofer/Figures/SLF_seasonality_3x1.pdf', format='PDF')
