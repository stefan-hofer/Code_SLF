import numpy as np
import seaborn as sns
import xarray as xr
import matplotlib.pyplot as plt
from cmcrameri import cm
import cartopy.crs as ccrs
import cartopy.feature as feat
import matplotlib.gridspec as gridspec
import xesmf as xe


# Data created as a mean from monthly values between start and end
start = '2009-06'
end = '2010-05'
ds = xr.open_dataset('/home/shofer/mount/CERES/NorESM2-CERES_bias_new.nc')

ceres = xr.open_dataset('/home/shofer/mount/CERES/CERES_EBAF-TOA_Ed4.1_Subset_200901-201012.nc')
ceres.coords['lon'] = (ceres.coords['lon'] + 180) % 360 - 180
ceres = ceres.sortby(ceres.lon).sel(time=slice(start, end)).mean(dim='time')

figdir = '/mount/coupled_runs/Figures/'

def weighted_mean(ds):
    # Create weights
    weights_n = np.cos(np.deg2rad(ds.lat))
    weighted = ds.weighted(weights_n)
    mean_arctic = weighted.mean(dim=['lat', 'lon'])

    return mean_arctic

def weighted_mean_1d(ds):
    # Create weights
    weights_n = np.cos(np.deg2rad(ds.lat))
    weighted = ds.weighted(weights_n)
    mean_arctic_1d = weighted.mean(dim=['lat']).values

    return mean_arctic_1d

# ======================= TEST AREA ===============================
bias = ds['TOA_bias']
areas = bias.constrain.values.tolist()
linestyles = ['solid', 'dashed', 'dashdot']

plt.close('all')
colors = ['#1b9e77','#d95f02','#7570b3']
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,4))
i = 0
for area in areas[1:]:
    difference_absolute = np.abs(bias.sel(constrain=area)) - np.abs(bias.sel(constrain='CTRL'))
    diff_z = difference_absolute.mean(dim='lon')
    diff_z.plot(label=area, lw=1.5, alpha=1, color=colors[i-1], ls=linestyles[0])
    print('The area is: {}'.format(area))
    print('The {} mean is: {}'.format(area, weighted_mean_1d(diff_z)))
    print('The {} mid-latitude mean is - All: {} ---- NET {} ----- SET {}'
          .format(area, weighted_mean_1d(diff_z.where((diff_z.lat<=-30) | (diff_z.lat >=30))),weighted_mean_1d(diff_z.where(diff_z.lat>=30))
                  ,weighted_mean_1d(diff_z.where(diff_z.lat<=-30))))
    
    i += 1

ax.legend()
ax.axhline(y=0, lw=1, color='black', alpha=0.7)
ax.legend(frameon=False, fontsize=12)
ax.set_xlabel(r'Latitude $(\circ)$', fontsize=11)
ax.set_ylabel(r'$\Delta$ TOA Bias $(Wm^{-2})$ ', fontsize=11) 
sns.despine()

fig.savefig(
    '/home/shofer/mount/coupled_runs/Figures/CERES_Bias_CTRLvsALL.png', dpi=300)
fig.savefig(
    '/home/shofer/mount/coupled_runs/Figures/CERES_Bias_CTRLvsALL.pdf', format='PDF')


# This also works
abs_diff = np.abs(ds['TOA_bias'].sel(constrain='Global')) - np.abs(ds['TOA_bias'].sel(constrain='CTRL'))
abs_diff_weighted = weighted_mean(abs_diff)
# ============ ========= TEST AREA END ===============================


# Getting model values back by adding the ceres values to the biases
model_vals = ds['TOA_bias'] + ceres['toa_net_all_mon']
abs_model = np.abs(model_vals).sel(constrain='Global') - np.abs(model_vals).sel(constrain='CTRL')
abs_model_weighted = weighted_mean(abs_model)

rel_bias_norm = (np.abs(ds['TOA_bias']) / ceres['toa_net_all_mon'])*100

rel_bias = model_vals / ceres['toa_net_all_mon']

sara_bias = ds['TOA_bias'].sel(constrain='Global')/ ds['TOA_bias'].sel(constrain='CTRL')

# Difference in bias between NorESM2 und CERES satellite compared to ctrl run
diff = ds['TOA_bias'] - ds.sel(constrain='CTRL')['TOA_bias']

# Zonal mean Difference in bias between NorESM2 und CERES satellite compared to ctrl run
diff_zonal = (ds['TOA_bias'] - ds.sel(constrain='CTRL')
              ['TOA_bias']).mean(dim='lon')


# def plot_maps(diff, diff_zonal, label=r'$\Delta$ Net Bias $(Wm^{-2})$', figname='NET_bias_CERES'):

label = r'$\Delta$ Net Bias $(Wm^{-2})$'
figname = 'NET_bias_CERES'
# Plotting routines
plt.close('all')
proj = ccrs.PlateCarree()
fig = plt.figure(figsize=(7, 10))
spec2 = gridspec.GridSpec(ncols=1, nrows=4, figure=fig)
ax1 = fig.add_subplot(spec2[0, 0], projection=proj)
ax2 = fig.add_subplot(spec2[1, 0], projection=proj)
ax3 = fig.add_subplot(spec2[2, 0], projection=proj)
ax4 = fig.add_subplot(spec2[3, 0])

# PLOT EVERYTHING

ax = [ax1, ax2, ax3, ax4]
letters = ['A', 'B', 'C', 'D']

area = ['Global', 'NH', 'SH']

i = 0
for area in area:
    data = diff.sel(constrain=area)
    data.plot(ax=ax[i], transform=proj, vmin=-10, vmax=10, cmap='RdBu_r',
              cbar_kwargs={'label': label})
    ax[i].add_feature(feat.COASTLINE.with_scale(
        '50m'), zorder=1, edgecolor='black')
    line = diff_zonal.sel(constrain=area)
    xr.plot.line(line, x='lat', ax=ax[3], label=area)

    ax[i].set_title(area, fontsize=16)
    ax[i].text(-0.05, 0.98, letters[i], fontsize=22, va='center', ha='center',
               transform=ax[i].transAxes, fontdict={'weight': 'bold'})
    i += 1

sns.despine()
fig.tight_layout()
ax[3].text(-0.05, 0.98, letters[3], fontsize=22, va='center', ha='center',
           transform=ax[i].transAxes, fontdict={'weight': 'bold'})
ax[3].set_title('')
ax[3].set_xlabel(r'Latitude $(\circ C)$', fontsize=11)
ax[3].set_ylabel(label, fontsize=11)
ax[3].legend(frameon=False)

fig.savefig(
    '/home/shofer/mount/coupled_runs/Figures/' + figname + '.png', dpi=300)
fig.savefig(
    '/home/shofer/mount/coupled_runs/Figures/' + figname + '.pdf', format='PDF')
