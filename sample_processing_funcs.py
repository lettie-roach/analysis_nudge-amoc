#!/usr/bin/env python
# coding: utf-8
# Functions for processing 

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import xarray as xr
import os
import pandas as pd
import scipy.stats
import sys
sys.path.insert(1, '/glade/u/home/lettier/analysis/')
import master_utils as myf
xr.set_options(keep_attrs=True)

def wrangle_lens (e, myvariables):
    ledir = '/glade/campaign/cesm/collections/cesmLE/CESM-CAM5-BGC-LE/atm/proc/tseries/monthly/'

    ens = str(e)
    if e<10:
        ens = '0'+str(e)
    tmp = []
    print(ens)
    for var in myvariables:
        myfiles = sorted([ledir+var+'/'+f for f in os.listdir(ledir+var) if ('B20TRC5CNBDRD' in f or 'BRCP85C5CNBDRD' in f) and '0'+ens+'.cam.h0.'+var in f ])
        myfiles = [f for f in myfiles if '.192001-199912.nc' not in f and '208101-210012.nc' not in f]
        ds = xr.open_mfdataset(myfiles).isel(lev=-1).sel(time=slice('2004-02','2017-01'))
        ds['time'] = mytime
        tmp.append(ds)
    ds = xr.merge(tmp)[myvariables]
    ds['names'] = 'LENS'+ens
    ds = ds.set_coords('names')

    return ds

def wrangle_nudge (nudge_name, myvariables):
    ds_a = xr.open_mfdataset(eddir+nudge_name+'/atm/hist/*.h0.*')[myvariables]
    ds_b = xr.open_mfdataset(eddir+nudge_name+'_21C/atm/hist/*.h0.*')[myvariables]
    ds = xr.concat([ds_a,ds_b],dim='time').sel(time=slice('2004-02','2017-01'))
    ds['time'] = mytime
    ds['names'] = nudge_name
    ds = ds.set_coords('names')
    
    return ds

def wrangle_proc_nudge (mydir, nudge_name, myvariables):
    
    listds = []
    for var in myvariables:
        listds.append(xr.open_dataset(mydir+nudge_name+'/atm/proc/tseries/month_1/'+nudge_name+'.cam.h0.'+var+'.197901-200512.nc').isel(lev=-1))
    ds_a = xr.merge(listds).sel(time=slice('2004-02','2017-01'))

    listds = []
    for var in myvariables:
        listds.append(xr.open_dataset(mydir+nudge_name+'_21C/atm/proc/tseries/month_1/'+nudge_name+'_21C.cam.h0.'+var+'.200601-201812.nc').isel(lev=-1))
    ds_b = xr.merge(listds).sel(time=slice('2004-02','2017-01'))

    ds = xr.concat([ds_a,ds_b],dim='time')
    ds['time'] = mytime
    ds['names'] = nudge_name
    ds = ds.set_coords('names')
    ds = ds[myvariables]
    
    return ds

def compute_corr (ds):
    # remove monthly climatology
    monclim = ds.groupby('time.month').mean(dim='time')
    monclim = monclim.rename({'month':'time'})
    listds = []
    for y in np.arange(2004,2017,1):
        yds = ds.sel(time=slice(str(y)+'-01',str(y)+'-12'))
        monclim['time'] = yds.time
        yds = yds - monclim
        listds.append(yds)
    ds = xr.concat(listds,dim='time')
    
    name = ds.names.values
    corr, pval = myf.pearson(ds.load(),obds.load(),dim='time')
    
    for var in corr:
        corr = corr.rename({var:var+'_corr'})
        pval = pval.rename({var:var+'_pval'})
    
    corr = xr.merge([corr,pval])
    corr.attrs = ds.attrs
    corr.attrs['decription'] = 'Pearson correlation over 2004-2016 with ERA-Interim at each grid point'
    corr['names'] = name
    corr = corr.set_coords('names')
    corr.attrs['processed_by'] = 'Lettie Roach, May 2022'
    if 'lev' in corr:
        corr = corr.drop('lev')
    corr.to_netcdf(mydir+'processed/corrERAI/UVP_2004-2016/'+str(name)+'.atm_corr_with_ERAI.2004-2016.nc')
  
    
    return


def get_spatial_stuff (ds):
    name = str(ds.names.values)
    units = ds.SST.attrs['units']
    print(name)
    yr_s = str(ds.time.values[0])[:4]
    yr_e = str(ds.time.values[-1])[:4]
 
    ds = ds.groupby('time.year').mean(dim='time')
    slope, intercept, r_value, p_value, std_err = myf.linregress(ds.year,ds.load(),dim='year')

    
    for var in ds:
        slope[var].attrs['units'] = units+'/yr'
        p_value[var] = 100.*p_value[var]
        p_value[var].attrs['units'] = '%'
        slope = slope.rename({var:var+'_trend'})
        p_value = p_value.rename({var:var+'_p_value'})

    ds = xr.merge([slope, p_value])
    ds.attrs['desc'] = 'processed by Lettie Roach, May 2022'
    ds.to_netcdf(mydir+'processed/spatial_mean_trend/ocn/'+name+'.ocn_climtrend.'+yr_s+'-'+yr_e+'.nc')
    
    return ds  



# [rho, alpha, beta] = gsw_rho_alpha_beta(SA,CT,p)
# Calculates in-situ density, the appropiate thermal expansion coefficient
#and the appropriate saline contraction coefficient of seawater from
# Absolute Salinity and Conservative Temperature.
#
#rho    =  in-situ density                                     [ kg/m^3 ]
#alpha  =  thermal expansion coefficient                          [ 1/K ]
#          with respect to Conservative Temperature
#beta   =  saline contraction coefficient                        [ kg/g ]
#          at constant Conservative Temperature#

def rho_alpha_beta(abs_salinity, cthetao, pressure):
    rho, alpha, beta = xr.apply_ufunc(gsw.rho_alpha_beta,
                       abs_salinity.load(), cthetao.load(), pressure.load(),
                       input_core_dims  = [[], [], []], 
                       output_core_dims = [[],[],[]],
                       dask='parallelized')
    
    rho.attrs['units'] = 'kg/m^3'
    alpha.attrs['units'] = '1/K'
    beta.attrs['units'] = 'kg/g'
        
    return rho, alpha, beta



g = 9.81 # m s^-2
rho_0 = 1030 # kg/m^3 density for seawater (not freshwater)
c_p = 3850 # J /kg /K

tarea = xr.open_dataset('/glade/scratch/lettier/archive/anom_nudge_era_60_21C/ice/hist/anom_nudge_era_60_21C.cice.h.2006-01.nc').tarea
tarea = tarea.rename({'ni':'nlon','nj':'nlat'})
tarea = tarea.sel(nlat=slice(300,None))
spg_area = tarea.where(tarea.TLAT>50.).where(tarea.TLAT<=65).where(tarea.TLON<=340).where(tarea.TLON>295)


def compute_buoyancy (ds):
    
    
    pressure = xr.apply_ufunc(gsw.p_from_z, -ds.z_t, ds.TLAT, dask='parallelized', 
                           output_dtypes=[float, ]).rename('pressure')
    pressure.attrs['units'] = 'dbar'

    # absolute salinity from practical salinity
    abs_salinity = xr.apply_ufunc(gsw.SA_from_SP, ds.SALT, pressure,
                                  ds.TLONG, ds.TLAT, dask='parallelized',
                                  output_dtypes=[float,]).rename('abs_salinity')
    abs_salinity.attrs['units'] = 'g/kg'

    cthetao = xr.apply_ufunc(gsw.CT_from_pt, abs_salinity, ds.TEMP, dask='parallelized',
                                    output_dtypes=[float,]).rename('cthetao')
    cthetao.attrs['units'] = 'degC'
    
    rho, alpha, beta = rho_alpha_beta(abs_salinity, cthetao, pressure)
    
    ds['alpha'] = alpha
    ds['beta'] = beta
    ds['rho'] = rho
    ds['cthetao'] = cthetao
    ds['abs_salinity'] = abs_salinity
    
    ds['B_HF'] = (g/rho_0)*ds.alpha*(ds.SHF)/c_p # SHF positive up
    ds['B_FW'] = (g/rho_0)*ds.beta*ds.abs_salinity*(ds.PREC_F-ds.EVAP_F) 
    ds['B'] = ds['B_FW']+ds['B_HF']
    
    ds['Q'] = (rho_0*c_p)/(g*ds.alpha)*ds['B']
    ds['Q_FW'] = (rho_0*c_p)/(g*ds.alpha)*ds['B_FW']
    ds['Q_HF'] = (rho_0*c_p)/(g*ds.alpha)*ds['B_HF']
    
    
    ds['B_HF'].attrs['units'] = 'm^2/s^3'
    ds['B_FW'].attrs['units'] = 'm^2/s^3'
    ds['B'].attrs['units'] = 'm^2/s^3'
    
    ds['Q_HF'].attrs['units'] = 'W/m^2'
    ds['Q_FW'].attrs['units'] = 'W/m^2'
    ds['Q'].attrs['units'] = 'W/m^2'
 
    ds['B_HF'].attrs['long_name'] = 'Heat contribution to buoyancy flux'
    ds['B_FW'].attrs['long_name'] = 'Freshwater contribution to buoyancy flux'
    ds['B'].attrs['long_name'] = 'Buoyancy flux'
    
    ds['Q_HF'].attrs['units'] = 'Heat contribution to heat-equivalent buoyancy flux'
    ds['Q_FW'].attrs['units'] = 'Freshwater contribution to heat-equivalent buoyancy flux'
    ds['Q'].attrs['units'] = 'Heat-equivalent buoyancy flux'
 
    
    
    ds = ds[['B','B_HF','B_FW','Q','Q_HF','Q_FW']]
    ds_avg =  (ds*spg_area).sum(dim=('nlat','nlon'))/(spg_area).sum(dim=('nlat','nlon'))
    ds_avg.attrs['desc'] = 'processed by Lettie Roach, May 2022'
    ds_avg.attrs['region'] = 'Subpolar gyre 50-65N, 295-340E'
    
    yr_s = str(ds.time.values[0])[:4]
    yr_e = str(ds.time.values[-1])[:4]
    
    ds_avg.to_netcdf(mydir+'processed/timeseries/'+str(ds_avg.names.values)+'.buoyancy_timeseries.'+yr_s+'-'+yr_e+'.nc')
   
    
    
    return ds_avg
    