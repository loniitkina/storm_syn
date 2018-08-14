#!/usr/bin/python

import numpy as np
from netCDF4 import Dataset
from pyhdf.SD import SD, SDC
from glob import glob
from pyresample import image, geometry, kd_tree
import pyresample as pr
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from matplotlib.patches import Polygon
import matplotlib.dates as mdates

inpath='../data/'
outpath='../plots/annual_cycle/'
icpath = inpath+'OSI-SAF/annual/'

#download sea ice concentration and sea ice drift data from OSI-SAF
#ftp://osisaf.met.no/archive/ice/conc/2015/01/                          #10km gird

#read the coordinates for the sea ice concentration files
iclist = sorted(glob(icpath+'ice_conc_nh_polstere-100_multi_20180708*.nc'))
print iclist
f = Dataset(iclist[0])
lats = f.variables['lat'][:]
lons = f.variables['lon'][:]
xc = f.variables['xc'][:]
yc = f.variables['yc'][:]
f.close()

for fn in iclist:
    print fn
    dt = fn.split('_')[-1].split('.')[0].split('1200')[0]
    print dt
    date = datetime.strptime(dt, "%Y%m%d")
    #if date < datetime(2015,01,20):continue
    #if date > datetime(2015,05,1):continue    
    
    #sea sea ice concentration data
    f = Dataset(fn)
    icecon = f.variables['ice_conc'][0,:,:]                                             #why is this variable masked already???
    f.close()
    
    #fill in missing values
    icecon.mask=False
    icecon = np.where(icecon==-999,0,icecon)
    
    #plot data
    fig    = plt.figure(figsize=(8,8))
    cx     = fig.add_subplot(111)

    cx.set_title(date.strftime('%B %Y'), fontsize=28, loc='left')

    m = Basemap(resolution='l',
            projection='stere', lat_ts=70, lat_0=90., lon_0=-30.,
            llcrnrlon=-70, llcrnrlat=45,
            urcrnrlon=110, urcrnrlat=45)
        
    m.drawmapboundary(fill_color='navy')
    m.drawcoastlines()
    m.fillcontinents(color='#ddaa66',lake_color='darkslateblue')
    ##Draw parallels and meridians
    m.drawparallels(np.arange(40.,90.,10),labels=[0,0,0,0], fontsize=16,latmax=85.)
    m.drawmeridians(np.arange(0.,360.,20.),latmax=85.,labels=[0,0,0,0], fontsize=16)
    #m.drawmapscale(17.2, 82.4, 15, 83, 20, barstyle='fancy', units='km', fontsize=16, yoffset=None, labelstyle='simple', fontcolor='k', fillcolor1='w', fillcolor2='k', ax=None, format='%d', zorder=None)

    ##plot the ice concentration
    x, y = m(lons, lats)
    im = m.pcolormesh(x, y, icecon, cmap=plt.cm.Blues_r, vmin=0, vmax=100)
    cbar = m.colorbar()
    cbar.ax.tick_params(labelsize=24)
    cbar.set_label('Sea Ice Concentration (%)',size=24)
    

    fig.tight_layout()
    outname = 'icecon_year'+dt
    fig.savefig(outpath+outname,bbox_inches='tight')
    plt.clf()

    #exit()

#convert -resize 50% -delay 10 ../plots/annual_cycle/icecon* ../plots/annual_cycle/cycle_anim.gif 
#convert -resize 200x200 -delay 10 ../plots/annual_cycle/icecon* ../plots/annual_cycle/cycle_anim_small.gif