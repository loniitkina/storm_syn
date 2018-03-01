#!/usr/bin/python

import numpy as np
from pyhdf.SD import SD, SDC
from netCDF4 import Dataset
from glob import glob
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

inpath='../data/OSI-SAF/'
outpath='../plots/'
icpath = inpath+'icecon/'
idpath = inpath+'merged/'

#download sea ice concentration and sea ice drift data from OSI-SAF
#ftp://osisaf.met.no/archive/ice/conc/2015/01/                          #10km gird
#ftp://osisaf.met.no/archive/ice/drift_lr/merged/2015/01/               #62.5km grid

#read the coordinates for the sea ice concentration files
iclist = sorted(glob(icpath+'ice_conc_nh_polstere-100_multi_*.nc'))
f = Dataset(iclist[0])
lats = f.variables['lat'][:]
lons = f.variables['lon'][:]
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

#read the coordinates for the sea ice drift files
idfile = idpath+'ice_drift_nh_polstere-625_multi-oi_201501011200-201501031200.nc'
f = Dataset(idfile)
coarse_lats = f.variables['lat'][:]
coarse_lons = f.variables['lon'][:]
f.close()

for fn in iclist:
    print fn
    date = fn.split('_')[-1].split('.')[0]
    print date
    enddate = str(int(date)+20000)
    print enddate
    #if int(date)<20150120:continue
    #if int(date)>20150320:continue    
    
    #reproject the sea ice drift to the ice concentration grid (62.5 > 10km)
    f = Dataset(fn)
    icecon = f.variables['ice_conc'][:]
    f.close()
    
    f = Dataset(idpath+'ice_drift_nh_polstere-625_multi-oi_'+date+'-'+enddate+'.nc')
    dx = f.variables['dX'][:]
    dy = f.variables['dY'][:]
    f.close()
   
    
    
    from pyresample import image, geometry
    
    print coarse_lats.shape
    print lats.shape
    exit()

#>>> area_def = geometry.AreaDefinition('areaD', 'Europe (3km, HRV, VTC)', 'areaD',
#...                                {'a': '6378144.0', 'b': '6356759.0',
#...                                 'lat_0': '50.00', 'lat_ts': '50.00',
#...                                 'lon_0': '8.00', 'proj': 'stere'},
#...                                800, 800,
#...                                [-1370912.72, -909968.64,
#...                                 1029087.28, 1490031.36])
#>>> msg_area = geometry.AreaDefinition('msg_full', 'Full globe MSG image 0 degrees',
#...                                'msg_full',
#...                                {'a': '6378169.0', 'b': '6356584.0',
#...                                 'h': '35785831.0', 'lon_0': '0',
#...                                 'proj': 'geos'},
#...                                3712, 3712,
#...                                [-5568742.4, -5568742.4,
#...                                 5568742.4, 5568742.4])
#>>> data = np.ones((3712, 3712))
#>>> msg_con_quick = image.ImageContainerQuick(data, msg_area)
#>>> area_con_quick = msg_con_quick.resample(area_def)
#>>> result_data_quick = area_con_quick.image_data
#>>> msg_con_nn = image.ImageContainerNearest(data, msg_area, radius_of_influence=50000)
#>>> area_con_nn = msg_con_nn.resample(area_def)
#>>> result_data_nn = area_con_nn.image_data    
    
    
    


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
  
  
