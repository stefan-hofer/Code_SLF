import numpy as np
import seaborn as sns
import xarray as xr
import matplotlib.pyplot as plt
from cmcrameri import cm
import cartopy.crs as ccrs
import cartopy.feature as feat
import matplotlib.gridspec as gridspec


# CALIOP DATA
folder = '/projects/NS9600K/shofer/'
data_folder = 'caliop_olimpia_new/netcdf_format/'

files_ann = ['bulk_slfs_annual.nc', 'ct_slfs_annual.nc']
files_seasonal = ['bulk_slfs_seasonal.nc', 'ct_slfs_seasonal.nc']

ann_bulk = xr.open_dataset(folder + data_folder + files_ann[0])
ann_ct = xr.open_dataset(folder + data_folder + files_ann[1])

seasonal_bulk = xr.open_dataset(folder + data_folder + files_seasonal[0])
seasonal_ct = xr.open_dataset(folder + data_folder + files_seasonal[1])

diff = ann_ct.SLF - ann_bulk.SLF
# Difference between DJF and JJA cloud top minus interior SLF difference
diff_seasonal = (seasonal_ct.SLF.isel(time=0) - seasonal_bulk.SLF.isel(time=0)) - \
    (seasonal_ct.SLF.isel(time=2) - seasonal_bulk.SLF.isel(time=2))

rolling = (ann_ct.SLF.isel(
    isotherm=4) * 100 - ann_bulk.SLF.isel(isotherm=4) * 100).rolling(lon=10, lat=5, min_periods=2).mean()

rolling_seasonal = (diff_seasonal.isel(isotherm=4) *
                    100).rolling(lon=10, lat=5, min_periods=2).mean()


plt.close('all')
proj = ccrs.Robinson()

fig = plt.figure(figsize=(14, 7))
spec2 = gridspec.GridSpec(ncols=9, nrows=10, figure=fig)
ax1 = fig.add_subplot(spec2[0:9, :8], projection=proj)
# ax2 = fig.add_subplot(spec2[0:2, 2:8])
# ax2.xaxis.set_ticks_position('top')
# plt.setp(ax2.get_yticklabels(), visible=False)
ax3 = fig.add_subplot(spec2[0:9, 8:])
ax3.yaxis.set_ticks_position('right')
ax4 = fig.add_subplot(spec2[9:, 1:7])


cont = xr.plot.pcolormesh(rolling, 'lon', 'lat', robust=True,
                          transform=ccrs.PlateCarree(), cmap=cm.bilbao, ax=ax1, add_colorbar=False)
cb = fig.colorbar(cont, cax=ax4, orientation='horizontal', shrink=0.8)
cb.set_label(r'$\Delta$ SLF (%, cloud top - interior)', fontsize=16)
cb.ax.tick_params(labelsize=11)
ax1.add_feature(feat.COASTLINE)
ax1.set_title('')
sns.despine(ax=ax1, bottom=True, left=True, top=True, right=True)

# plot on the line above the map
# (ann_ct.SLF.isel(
#     isotherm=4) * 100).mean(dim='lat').plot(ax=ax2, label='CT')
# (ann_bulk.SLF.isel(isotherm=4) * 100).mean(dim='lat').plot(ax=ax2, label='Interior')
# sns.despine(ax=ax2, bottom=True, left=True, top=True, right=True)
# ax2.set_title('')
# plot on the right of the map with flipped axes
rolling.mean(dim='lon').plot(ax=ax3, y='lat')
#
sns.despine(ax=ax3, bottom=True, left=True, top=True, right=True)
ax3.set_title('')
ax3.grid()
ax3.set_ylabel("Latitude")
ax3.yaxis.set_label_position("right")
fig.tight_layout()

fig.savefig('/projects/NS9600K/shofer/keyclim/plots/SLF_map.png',
            format='PNG', dpi=300)
fig.savefig('/projects/NS9600K/shofer/keyclim/plots/SLF_map.pdf',
            format='PDF')


plt.close('all')
proj = ccrs.Robinson()

fig = plt.figure(figsize=(14, 7))
spec2 = gridspec.GridSpec(ncols=9, nrows=10, figure=fig)
ax1 = fig.add_subplot(spec2[0:9, :8], projection=proj)
# ax2 = fig.add_subplot(spec2[0:2, 2:8])
# ax2.xaxis.set_ticks_position('top')
# plt.setp(ax2.get_yticklabels(), visible=False)
ax3 = fig.add_subplot(spec2[0:9, 8:])
ax3.yaxis.set_ticks_position('right')
ax4 = fig.add_subplot(spec2[9:, 1:7])


cont = xr.plot.pcolormesh(rolling_seasonal, 'lon', 'lat', robust=True,
                          transform=ccrs.PlateCarree(), cmap=cm.bilbao, ax=ax1, add_colorbar=False)
cb = fig.colorbar(cont, cax=ax4, orientation='horizontal', shrink=0.8)
cb.set_label(r'$\Delta$ SLF (%, cloud top - interior)', fontsize=16)
cb.ax.tick_params(labelsize=11)
ax1.add_feature(feat.COASTLINE)
ax1.set_title('')
sns.despine(ax=ax1, bottom=True, left=True, top=True, right=True)

# plot on the line above the map
# (ann_ct.SLF.isel(
#     isotherm=4) * 100).mean(dim='lat').plot(ax=ax2, label='CT')
# (ann_bulk.SLF.isel(isotherm=4) * 100).mean(dim='lat').plot(ax=ax2, label='Interior')
# sns.despine(ax=ax2, bottom=True, left=True, top=True, right=True)
# ax2.set_title('')
# plot on the right of the map with flipped axes
rolling_seasonal.mean(dim='lon').plot(ax=ax3, y='lat')
#
sns.despine(ax=ax3, bottom=True, left=True, top=True, right=True)
ax3.set_title('')
