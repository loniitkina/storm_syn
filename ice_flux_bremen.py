#!/usr/bin/python

#This is failed attempt to compute the sea ice area flux from the Uni Bremen algorithm and IFREMER sea ice drift. Here just for future refence and demo of HDF4 file format use.

import numpy as np
from pyhdf.SD import SD, SDC
from netCDF4 import Dataset
from glob import glob
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

inpath='../data/'
outpath='../plots/'
bremenpath = inpath+'UniBremen/AMSR2/'
ifrpath = inpath+'IFREMER/AMSR2/'

#download sea ice concentration data from UNI Bremen
#ftp://seaice.uni-bremen.de/data/amsr2/bootstrap_daygrid/n12500/2015/jan/Arctic/
#ftp_bremen.py

#and sea ice drift data of IFREMER from:
#ftp://ftp.ifremer.fr/ifremer/cersat/products/gridded/psi-drift/data/arctic/amsr2-merged/2-daily/netcdf/2015/
#ftp_ifremer.py

#read the coordinates for the sea ice concentration file
grdfile = bremenpath+'LongitudeLatitudeGrid-n12500-Arctic.hdf'
hdf = SD(grdfile, SDC.READ)

print hdf.datasets()
# Read geolocation dataset.
iclats = hdf.select('Latitudes')[:,:]
iclons = hdf.select('Longitudes')[:,:]
#print lats

##get all the sea ice concentration files
#iceconlist = sorted(glob(bremenpath+'bootstrap-AMSR2-n12500*'))

#get sea ice drift files
ifrlist = sorted(glob(ifrpath+'2015*.nc'))

#read the coordinates for the sea ice drift files
f = Dataset(ifrlist[0])
lats = f.variables['latitude'][:] * f.variables['latitude'].scale_factor
lons = f.variables['longitude'][:] * f.variables['longitude'].scale_factor
f.close()

#get the gate mask
#north
mask1 = (lats < 83.2) * (lats > 83) * (lons>5.) * (lons<70.)
#south
mask2 = (lats < 79.2) * (lats > 79) * (lons>5.) * (lons<70.)
#east
mask3 = (lats < 83.2) * (lats > 79) * (lons>70.) * (lons<70.7)
#west
mask4 = (lats < 83.2) * (lats > 79) * (lons>5.) * (lons<5.7)
#combine  
gates = mask1+mask2+mask3+mask4



for fn in ifrlist:
    print fn
    date = fn.split('-')[-1].split('.')[0]
    if int(date)<20150120:continue
    if int(date)>20150320:continue    
    
    #coarsen the ice concentration to the ice drift grid
    iceconfile = bremenpath+'bootstrap-AMSR2-n12500-'+date+'.hdf'
    hdf = SD(iceconfile, SDC.READ)
    print hdf.datasets()
    icecon = hdf.select('Bootstrap Ice Concentration')[:,:]
    print icecon
    
    #read sea ice drift maps
    #make daily averages
    
    #calculate the area flux
    #sum up for all gates
    #store in time series


 
    #plotting
    fig    = plt.figure(figsize=(10,10))
    cx      = fig.add_subplot(111)

    cx.set_title(date, fontsize=28, loc='left')

    m = Basemap(resolution='h',
                projection='stere', lat_ts=82, lat_0=90., lon_0=0.,
                llcrnrlon=0, llcrnrlat=79,
                urcrnrlon=80, urcrnrlat=80)

    m.drawmapboundary(fill_color='#9999FF')
    m.drawcoastlines()
    m.fillcontinents(color='#ddaa66',lake_color='#9999FF')
    ##Draw parallels and meridians
    m.drawparallels(np.arange(80.,86.,1),labels=[1,0,0,0], fontsize=16,latmax=85.)
    m.drawmeridians(np.arange(-5.,90.,5.),latmax=85.,labels=[0,0,0,1], fontsize=16)
    #m.drawmapscale(17.2, 82.4, 15, 83, 20, barstyle='fancy', units='km', fontsize=16, yoffset=None, labelstyle='simple', fontcolor='k', fillcolor1='w', fillcolor2='k', ax=None, format='%d', zorder=None)

    #plot the ice concentration
    x, y = m(lons, lats)
    im = m.pcolormesh(x, y, icecon, cmap=plt.cm.jet)
    
    #plot the gates
    im = m.pcolor(x, y, gates)
    
    fig.tight_layout()
    outname = 'icecon'+date
    fig.savefig(outpath+outname,bbox_inches='tight')

  
    #fluxm = uec_next*ic*dxC_tm
    exit()
  
  
