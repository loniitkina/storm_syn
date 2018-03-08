#!/usr/bin/python

import numpy as np
from netCDF4 import Dataset
from glob import glob
from pyresample import image, geometry, kd_tree
import pyresample as pr
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from matplotlib.patches import Polygon
import matplotlib.dates as mdates

inpath='../data/'
outpath='../plots/'
icpath = inpath+'OSI-SAF/icecon/'
idpath = inpath+'OSI-SAF/merged/'

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

##get the gate mask
xx, yy = np.meshgrid(xc, yc)
##north
#mask1 = (xx < 610) * (xx > 600) * (yy>-500) * (yy<300)
##south
#mask2 = (xx < 1010) * (xx > 1000) * (yy>-500) * (yy<300)
##east
#mask3 = (xx < 1000) * (xx > 600) * (yy>300) * (yy<310)
##west
#mask4 = (xx < 1000) * (xx > 600) * (yy>-510) * (yy<-500)
##combine  
#gates = mask1+mask2+mask3+mask4

#read the coordinates for the sea ice drift files
idfile = idpath+'ice_drift_nh_polstere-625_multi-oi_201501011200-201501031200.nc'
f = Dataset(idfile)
coarse_lats = f.variables['lat'][:]
coarse_lons = f.variables['lon'][:]
#xcc = f.variables['xc'][:]
#ycc = f.variables['yc'][:]
f.close()

#define both grids (for reprojection later on)
area_def = pr.utils.load_area('area.cfg', 'osi_icecon')
area_def_coarse = pr.utils.load_area('area.cfg', 'osi_drift')

#sanity check - they are same as in the .nc file!
xp,yp = area_def.get_proj_coords()
print xc[0],xc[-1]
print xp[0,0],xp[-1,-1]
print yc[0],yc[-1]
print yp[0,0],yp[-1,-1]

#make mask for the flux gates
#lower left corner
xmc1=600000; ymc1=-500000
#upper right corner
xmc2=1000000; ymc2=300000
#10km distance
dd=10000
#north
mask1 = (xp < xmc1+dd) * (xp > xmc1) * (yp>ymc1) * (yp<ymc2)
#south
mask2 = (xp < xmc2+dd) * (xp > xmc2) * (yp>ymc1) * (yp<ymc2)
#east
mask3 = (xp < xmc2) * (xp > xmc1) * (yp>ymc2) * (yp<ymc2+dd)
#west
mask4 = (xp < xmc2) * (xp > xmc1) * (yp>ymc1) * (yp<ymc1+dd)
#combine  
gates = mask1+mask2+mask3+mask4

#make mask for the areal average in the box
mask5 = (xp < xmc2) * (xp > xmc1) * (yp>ymc1) * (yp<ymc2)

#still things are very fish here - the corner coordinated whne convereted to lonlat are all mixed up!!!
from pyproj import Proj,transform
osi_proj = Proj("+proj=stere +a=6378273 +b=6356889.44891 +lat_0=90 +lat_ts=70 +lon_0=45")
lonlat = Proj('+init=EPSG:4326')
c1lon,c1lat = transform(osi_proj,lonlat,ymc1,-xmc1)
c2lon,c2lat = transform(osi_proj,lonlat,ymc1,-xmc2)
c3lon,c3lat = transform(osi_proj,lonlat,ymc2,-xmc2)
c4lon,c4lat = transform(osi_proj,lonlat,ymc2,-xmc1)

##alternaitve conversion, leading to even worse results:
#c1x,c1y = area_def.get_xy_from_proj_coords(xmc1,ymc1)
#c1lon,c1lat = area_def.get_lonlat(c1x,c1y)
#c2x,c2y = area_def.get_xy_from_proj_coords(xmc2,ymc1)
#c2lon,c2lat = area_def.get_lonlat(c2x,c2y)
#c3x,c3y = area_def.get_xy_from_proj_coords(xmc2,ymc2)
#c3lon,c3lat = area_def.get_lonlat(c3x,c3y)
#c4x,c4y = area_def.get_xy_from_proj_coords(xmc1,ymc2)
#c4lon,c4lat = area_def.get_lonlat(c4x,c4y)

#import ipdb; ipdb.set_trace()

advection = []
m_icecon = []
dates = []
for fn in iclist:
    #break
    print fn
    dt = fn.split('_')[-1].split('.')[0]
    date = datetime.strptime(dt, "%Y%m%d%H%M")
    if date < datetime(2015,01,20):continue
    if date > datetime(2015,03,20):continue    
    
    #sea sea ice concentration data
    f = Dataset(fn)
    icecon = f.variables['ice_conc'][0,:,:]/100                                             #why is this variable masked already???
    f.close()
        
    #read sea ice displacement data
    dfn = glob(idpath+'ice_drift_nh_polstere-625_multi-oi_'+dt+'*.nc')
    f = Dataset(dfn[0])
    dx = f.variables['dX'][0,:,:]   #displacement in km
    dy = f.variables['dY'][0,:,:]
    f.close()
    
    #calculate drift
    u_coarse = dx*1000/(2*24*60*60) #sea ice velocity in m/s
    v_coarse = dy*1000/(2*24*60*60)
   
    ##reproject the sea ice drift to the ice concentration grid (62.5 > 10km)
    ##tmp = image.ImageContainerNearest(u_coarse, area_def_coarse, radius_of_influence=70000)
    tmp = image.ImageContainerQuick(u_coarse, area_def_coarse)
    u = tmp.resample(area_def).image_data
    ##tmp = image.ImageContainerNearest(v_coarse, area_def_coarse, radius_of_influence=70000)
    tmp = image.ImageContainerQuick(v_coarse, area_def_coarse)
    v = tmp.resample(area_def).image_data
    
    interpolate missing drift values
    u = kd_tree.resample_nearest(area_def_coarse, u_coarse ,area_def, radius_of_influence=100000)
    v = kd_tree.resample_nearest(area_def_coarse, u_coarse ,area_def, radius_of_influence=100000)
    
    
        
    #mask all the data
    #ic_gates = np.ma.array(icecon,mask=~gates)
    v_n = np.ma.array(v,mask=~mask1)
    v_s = np.ma.array(v,mask=~mask2)
    u_e = np.ma.array(u,mask=~mask3)
    u_w = np.ma.array(u,mask=~mask4)
    
    #calculate the area flux
    #calculation has to be separate for all gates - sign will be opposite depending on the gate
    flux_n = v_n * icecon
    flux_s = v_s * icecon
    flux_e = u_e * icecon
    flux_w = u_w * icecon
    
    #sum up for all gates
    flux = -np.sum(flux_n) + np.sum(flux_s) - np.sum(flux_e) + np.sum(flux_w)
    flux = flux/100 #in km2/s   (/1000 to get km/s; *10 to account for 10km wide grid box)
    print flux
    #store in time series
    advection.append(flux)
    
    icecon_subset = np.ma.array(icecon,mask=~mask5)
    m_icecon.append(np.mean(icecon_subset))
    
    dates.append(date)
 
    ##Plotting
    #fig    = plt.figure(figsize=(10,10))
    #cx      = fig.add_subplot(111)

    #cx.set_title(date, fontsize=28, loc='left')

    #m = Basemap(resolution='h',
                #projection='stere', lat_ts=82, lat_0=90., lon_0=0.,
                #llcrnrlon=0, llcrnrlat=79,
                #urcrnrlon=80, urcrnrlat=80)
    
    ##m = pr.plot.area_def2basemap(area_def)
    
    #m.drawmapboundary(fill_color='#9999FF')
    #m.drawcoastlines()
    #m.fillcontinents(color='#ddaa66',lake_color='#9999FF')
    ###Draw parallels and meridians
    #m.drawparallels(np.arange(80.,90.,5),labels=[1,0,0,0], fontsize=16,latmax=85.)
    #m.drawmeridians(np.arange(0.,90.,20.),latmax=85.,labels=[0,0,0,1], fontsize=16)
    ##m.drawmapscale(17.2, 82.4, 15, 83, 20, barstyle='fancy', units='km', fontsize=16, yoffset=None, labelstyle='simple', fontcolor='k', fillcolor1='w', fillcolor2='k', ax=None, format='%d', zorder=None)

    ###plot the ice concentration
    #x, y = m(lons, lats)
    #im = m.pcolormesh(x, y, icecon, cmap=plt.cm.jet)
    #m.colorbar()
    
    ####plot the gates
    ##x, y = m(lons, lats)
    ##im = m.pcolormesh(x, y, gates, cmap=plt.cm.jet)
    #lonbox = [c1lon,c2lon,c3lon,c4lon]
    #latbox = [c1lat,c2lat,c3lat,c4lat]
    #xbox,ybox = m(lonbox,latbox)
    #xybox = zip(xbox,ybox)
    #poly = Polygon(xybox, edgecolor='w', alpha=1, fill=False, linewidth=3)
    #plt.gca().add_patch(poly)

    ###Robs
    ##lonband = [5, 15,25,35,45,55,70,70,55,45,35,25,14,5]
    ##latband = [79,79,79,79,79,79,79,83,83,83,83,83,83,83]
    ##xband,yband = m(lonband,latband)
    ##xyband = zip(xband,yband)
    ##poly_band = Polygon( xyband, edgecolor='k', alpha=1, fill=False, linewidth=3)
    ##plt.gca().add_patch(poly_band)

    ##plot velocities
    #x, y = m(coarse_lons, coarse_lats)
    #Q = plt.quiver(x, y, u_coarse, v_coarse, units='width', scale=8, width=.002)
    #qk = plt.quiverkey(Q, 0.9, 0.1, .1, r'$10 \frac{cm}{s}$', labelpos='E',
                   #coordinates='axes',fontproperties={'size': 16},labelcolor='w',color='w')

    #fig.tight_layout()
    #outname = 'icecon'+dt
    #fig.savefig(outpath+outname,bbox_inches='tight')

    #exit()
  
#save lists 
np.save(inpath+'advection',advection)
np.save(inpath+'m_icecon',m_icecon)
np.save(inpath+'date_flux',dates)

#load lists
advection = np.load(inpath+'advection.npy')/10
m_icecon = np.load(inpath+'m_icecon.npy')
dates = np.load(inpath+'date_flux.npy')

#TIME SERIES PLOT
fig2 = plt.figure(figsize=(8,8))
ax = fig2.add_subplot(211)
ax.set_ylabel(r"Area flux $(km^2/s)$",fontsize=18)
##set the limits for the axis
#ax.set_xlim(start,end)
#ax.set_ylim(-5,5)
ax.grid('on')
ax.set_facecolor('.9')
ax.tick_params(labelsize=18)

ax.plot(dates,advection, label='OSI-SAF')

#volume fluxes (mean thicknes=1.5m)
aax = ax.twinx()
aax.spines['top'].set_visible(False)
aax.spines['bottom'].set_visible(False)
#aax.set_ylabel(r'Volume flux $(km^3/s)$', fontsize=18)
aax.set_ylabel(r'Distributed heat flux $(W/m^2)$', fontsize=18)
#aax.plot(dates,advection*.0015, label='OSI-SAF')
#latent heat flux ~ as if the ice was melting/growing
rhoi=900        #kg/m3
li = 334000     #J/kg    (J=kg*m^2/s^2)
export = advection*.0015 *1000000000              #in m^3/s
area = (xmc1-xmc2)*(ymc1-ymc2)                    #in m^2
fl = rhoi*li*export/area
aax.plot(dates,fl, label='OSI-SAF')               #W/m^2 (W=J/s)


bx = fig2.add_subplot(212)
bx.set_ylabel(r"Sea ice concentration",fontsize=18)
bx.grid('on')
bx.set_facecolor('.9')
bx.tick_params(labelsize=18)

bx.plot(dates,m_icecon, label='mean sea ice concentration')

from matplotlib.dates import MO
days = mdates.DayLocator()   			# every day
mons = mdates.WeekdayLocator(byweekday=MO) 	#every monday

ax.xaxis.set_major_locator(mons)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%b'))
ax.xaxis.set_minor_locator(days)

bx.xaxis.set_major_locator(mons)
bx.xaxis.set_major_formatter(mdates.DateFormatter('%d-%b'))
bx.xaxis.set_minor_locator(days)

fig2.autofmt_xdate()

#highlight storms (all based on Lana's storm table, except the first one which is based on temeprature above -20)
#MAJOR
[whole_plot.axvspan(datetime(2015,1,21,15,0), datetime(2015,1,22,15,0), facecolor='#d8bfd8', alpha=0.4, linewidth=0) for whole_plot in [ax,bx]]
[whole_plot.axvspan(datetime(2015,2,3,11,0), datetime(2015,2,8,21,0), facecolor='#d8bfd8', alpha=0.4, linewidth=0) for whole_plot in [ax,bx]]
[whole_plot.axvspan(datetime(2015,2,15,12,0), datetime(2015,2,16,17,0), facecolor='#d8bfd8', alpha=0.4, linewidth=0) for whole_plot in [ax,bx]]
[whole_plot.axvspan(datetime(2015,2,17,16,0), datetime(2015,2,21,4,0), facecolor='#d8bfd8', alpha=0.4, linewidth=0) for whole_plot in [ax,bx]]

[whole_plot.axvspan(datetime(2015,3,2,10,0), datetime(2015,3,4,1,0), facecolor='#d8bfd8', alpha=0.4, linewidth=0) for whole_plot in [ax,bx]]
[whole_plot.axvspan(datetime(2015,3,7,8,0), datetime(2015,3,8,18,0), facecolor='#d8bfd8', alpha=0.4, linewidth=0) for whole_plot in [ax,bx]]
[whole_plot.axvspan(datetime(2015,3,14,21,0), datetime(2015,3,16,23,0), facecolor='#d8bfd8', alpha=0.4, linewidth=0) for whole_plot in [ax,bx]]

#minor
[whole_plot.axvspan(datetime(2015,2,13,4,0), datetime(2015,2,13,9,0), facecolor='cornflowerblue', alpha=0.2, linewidth=0) for whole_plot in [ax,bx]]
[whole_plot.axvspan(datetime(2015,2,22,8,0), datetime(2015,2,23,1,0), facecolor='cornflowerblue', alpha=0.2, linewidth=0) for whole_plot in [ax,bx]]
[whole_plot.axvspan(datetime(2015,2,25,6,0), datetime(2015,2,25,20,0), facecolor='cornflowerblue', alpha=0.2, linewidth=0) for whole_plot in [ax,bx]]

ax.legend(loc='upper right',prop={'size':16}, fancybox=True, framealpha=.5, numpoints=1)
fig2.tight_layout()
fig2.savefig(outpath+'advection_ts')

#convert -delay 100 ../plots/icecon20150* ../plots/icecon_anim.gif