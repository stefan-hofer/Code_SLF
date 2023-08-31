import cartopy.crs as ccrs
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from cartopy.util import add_cyclic_point
import seaborn as sns
import cartopy.crs as ccrs
import cartopy.feature as feat
import glob
from cmcrameri import cm
import matplotlib.gridspec as gridspec

obs_dir = '/projects/NS9600K/shofer/GOCCP_data/2Ddata'
obs_dir = '/home/shofer/mount/GOCCP_data/2Ddata'


opaq_files = glob.glob('%s/20??/Map_OPAQ*.nc' % obs_dir)
opaq_files.sort()

opaq_ds = xr.open_mfdataset(opaq_files)


def weighted_mean(ds):
    # Weighted mean if doing latitude based analysis
    # Create weights
    weights_n = np.cos(np.deg2rad(ds.lat))
    weighted = ds.weighted(weights_n)
    mean_arctic = weighted.mean(dim=['lat', 'lon'])

    return mean_arctic


def weighted_std(ds):
    # Weighted mean if doing latitude based analysis
    # Create weights
    weights_n = np.cos(np.deg2rad(ds.lat))
    weighted = ds.weighted(weights_n)
    std_arctic = weighted.std(dim=['lat', 'lon'])

    return std_arctic


def average(ds, year_start, year_end, var='PRECT'):
    mean_ref = ds[var].sel(time=slice(year_start, year_end)).mean(dim='time')

    return mean_ref


# opaq_ds_subset = opaq_ds.sel(time=slice('2009-06-01','2013-05-31'),latitude=slice(59,90))
opaq_ds_subset = opaq_ds.sel(time=slice(
    '2009-06-01', '2013-05-31'))
opaq_ds_subset = opaq_ds_subset.rename({'latitude': 'lat', 'longitude': 'lon'})

# thin cloud fraction
cltcalipso_thin_Tavg = opaq_ds_subset['cltcalipso_thin'].mean('time')
# opaque cloud fraction
cltcalipso_opaque_Tavg = opaq_ds_subset['cltcalipso_opaque'].mean('time')
# Opaque cloud fraction
calipso_opaque_frac = cltcalipso_opaque_Tavg / \
    (cltcalipso_opaque_Tavg + cltcalipso_thin_Tavg)
# depth of penetration in oqaque clouds
caliop_depth = (opaq_ds_subset['cltcalipso_opaque_z'] -
                opaq_ds_subset['zopaque']).mean('time')

# Calculate Arctic Averages:
cltcalipso_thin_ArcAvg = weighted_mean(cltcalipso_thin_Tavg)
cltcalipso_opaque_ArcAvg = weighted_mean(cltcalipso_opaque_Tavg)
calipso_opaque_frac_ArcAvg = weighted_mean(calipso_opaque_frac)
caliop_depth_ArcAvg = weighted_mean(caliop_depth)

# This yield the standard deviation
depth_std = weighted_std(caliop_depth)


# ==========================================================
# ========== PLOT ONLY PENETRATION DEPTH FIRST =============
# ==========================================================
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


cont = xr.plot.pcolormesh(caliop_depth, 'lon', 'lat', robust=True,
                          transform=ccrs.PlateCarree(), cmap=cm.batlowW, ax=ax1,
                          add_colorbar=False, vmin=0.5, vmax=2.5)

ax1.set_title('Opaque Cloud Penetration Depth: Mean = %.2f km' %
              caliop_depth_ArcAvg, fontsize=15)
cb = fig.colorbar(cont, cax=ax4, orientation='horizontal', shrink=0.8)
cb.set_label(label='Opaque Cloud Penetration Depth (km)', fontsize=16)
cb.ax.tick_params(labelsize=11)
ax1.add_feature(feat.COASTLINE)
# ax1.set_title('')
sns.despine(ax=ax1, bottom=True, left=True, top=True, right=True)

caliop_depth.mean(dim="lon").plot(ax=ax3, y='lat')
ax3.set_xlim([0, 3])
ax3.set_xticks([0, 1, 2, 3])
#
sns.despine(ax=ax3, bottom=True, left=True, top=True, right=True)
ax3.set_title('')
ax3.set_xlabel("Height (km)")
ax3.set_ylabel("Latitude")
ax3.grid()
ax3.yaxis.set_label_position("right")
fig.tight_layout()

fig.savefig('/projects/NS9600K/shofer/keyclim/plots/penetration_CALIOP_1x1.pdf',
            format='PDF')
fig.savefig('/projects/NS9600K/shofer/keyclim/plots/penetration_CALIOP_1x1.png',
            format='PNG', dpi=300)


# ============================================================================
# Plot thin and opaque cloud fraction with penetration depth for all months.
# ============================================================================
proj = ccrs.PlateCarree()
plt.close('all')
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(
    15, 15), subplot_kw={'projection': proj})
axes = axes.flatten()

plt.subplots_adjust(hspace=0.25)


im0 = cltcalipso_thin_Tavg.plot(ax=axes[0], cmap=cm.oslo_r, transform=ccrs.PlateCarree(),
                                add_colorbar=False, robust=True, vmin=0, vmax=0.9)
axes[0].set_title('CALIPSO Thin Cloud Fraction: Mean = %.2f' %
                  cltcalipso_thin_ArcAvg, fontsize=15)
cax0 = fig.add_axes([0.15, 0.56, 0.3, 0.01])
cbar0 = plt.colorbar(im0, cax=cax0, orientation='horizontal')
cbar0.set_label(label='CALIPSO Thin Cloud Fraction', fontsize=13)

im1 = cltcalipso_opaque_Tavg.plot(ax=axes[1], cmap=cm.oslo_r, transform=ccrs.PlateCarree(),
                                  add_colorbar=False, robust=True, vmin=0, vmax=0.9)
axes[1].set_title('CALIPSO Opaque Cloud Fraction: Mean = %.2f' %
                  cltcalipso_opaque_ArcAvg, fontsize=15)
cax1 = fig.add_axes([0.5735, 0.56, 0.3, 0.01])
cbar1 = plt.colorbar(im1, cax=cax1, orientation='horizontal')
cbar1.set_label(label='CALIPSO Opaque Cloud Fraction', fontsize=13)


im2 = calipso_opaque_frac.plot(ax=axes[2], cmap=cm.oslo_r, transform=ccrs.PlateCarree(),
                               add_colorbar=False, robust=True, vmin=0, vmax=1.0)
axes[2].set_title('Opaque Fraction (opaque / thin+opaque): Mean = %.2f' %
                  calipso_opaque_frac_ArcAvg, fontsize=15)
cax2 = fig.add_axes([0.15, 0.13, 0.3, 0.01])
cbar2 = plt.colorbar(im2, cax=cax2, orientation='horizontal')
cbar2.set_label(label='Opaque Fraction', fontsize=13)

# test_palette = sns.cubehelix_palette(9,0.3,0.8,0.6,0.8,0.8,0.2,as_cmap=True)
im3 = caliop_depth.plot(ax=axes[3], transform=ccrs.PlateCarree(), cmap=cm.batlowW,
                        add_colorbar=False, robust=False, vmin=0.5, vmax=2.5)
axes[3].set_title('Opaque Cloud Penetration Depth: Mean = %.2f km' %
                  caliop_depth_ArcAvg, fontsize=15)
cax3 = fig.add_axes([0.5735, 0.13, 0.3, 0.01])
cbar3 = plt.colorbar(im3, cax=cax3, orientation='horizontal')
cbar3.set_label(label='Opaque Cloud Penetration Depth (km)', fontsize=13)

for ax in axes:
    ax.add_feature(feat.COASTLINE.with_scale(
        '50m'), zorder=1, edgecolor='black')
fig.tight_layout()

fig.savefig('/projects/NS9600K/shofer/keyclim/plots/penetration_CALIOP.pdf',
            format='PDF')
fig.savefig('/projects/NS9600K/shofer/keyclim/plots/penetration_CALIOP.png',
            format='PNG', dpi=300)
