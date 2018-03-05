#!/usr/bin/python

import numpy as np
from netCDF4 import Dataset
from glob import glob
from pyresample import image, geometry
import pyresample as pr
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
xc = f.variables['xc'][:]
yc = f.variables['yc'][:]
f.close()

#get the gate mask
xx, yy = np.meshgrid(xc, yc)
#north
mask1 = (xx < 610) * (xx > 600) * (yy>-500) * (yy<300)
#south
mask2 = (xx < 1010) * (xx > 1000) * (yy>-500) * (yy<300)
#east
mask3 = (xx < 1000) * (xx > 600) * (yy>300) * (yy<310)
#west
mask4 = (xx < 1000) * (xx > 600) * (yy>-510) * (yy<-500)
#combine  
gates = mask1+mask2+mask3+mask4

#read the coordinates for the sea ice drift files
idfile = idpath+'ice_drift_nh_polstere-625_multi-oi_201501011200-201501031200.nc'
f = Dataset(idfile)
coarse_lats = f.variables['lat'][:]
coarse_lons = f.variables['lon'][:]
f.close()

#define both grids (for reprojection later on)
area_def = pr.utils.load_area('area.cfg', 'osi_icecon')
area_def_coarse = pr.utils.load_area('area.cfg', 'osi_drift')

for fn in iclist:
    print fn
    date = fn.split('_')[-1].split('.')[0]
    print date
    enddate = str(int(date)+20000)
    print enddate
    #if int(date)<20150120:continue
    #if int(date)>20150320:continue    
    
    #sea sea ice concentration data
    f = Dataset(fn)
    icecon = f.variables['ice_conc'][0,:,:]
    f.close()
    
    #read sea ice displacement data
    f = Dataset(idpath+'ice_drift_nh_polstere-625_multi-oi_'+date+'-'+enddate+'.nc')
    dx = f.variables['dX'][0,:,:]
    dy = f.variables['dY'][0,:,:]
    f.close()
    
    #calculate drift
    u_coarse = dx/(2*24*60*60)
    v_coarse = dx/(2*24*60*60)
   
    #reproject the sea ice drift to the ice concentration grid (62.5 > 10km)
    tmp = image.ImageContainerNearest(u_coarse, area_def_coarse, radius_of_influence=60)
    u = tmp.resample(area_def).image_data
    tmp = image.ImageContainerNearest(v_coarse, area_def_coarse, radius_of_influence=60)
    v = tmp.resample(area_def).image_data

    #mask all the data
    u_gates = np.ma.array(u,mask=~gates)
    v_gates = np.ma.array(v,mask=~gates)
    ic_gates = np.ma.array(icecon,mask=~gates)
    
    #calculate the area flux
    #calculation has to be separate for all gates - sign will be opposite depending on the gate
    flux_n = v_n * ic_n *100
    flux_s = v_s * ic_s *100
    flux_e = u_e * ic_e *100
    flux_w = u_w * ic_w *100
    
    #sum up for all gates
    flux = -flux_n + flux_s - flux_e + flux_w
    
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
    #im = m.pcolormesh(x, y, icecon, cmap=plt.cm.jet)
    #m.colorbar()
    
    #plot the gates
    im = m.pcolor(x, y, ic_gates)
    m.colorbar()
    
    fig.tight_layout()
    outname = 'icecon'+date
    fig.savefig(outpath+outname,bbox_inches='tight')

  
    #fluxm = uec_next*ic*dxC_tm
    exit()
  
  
