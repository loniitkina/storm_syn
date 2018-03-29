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
outpath='../plots/'
icpath = inpath+'OSI-SAF/icecon/'
idpath = inpath+'OSI-SAF/merged/'
bremenpath = inpath+'UniBremen/AMSR2/'

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

##Bremen sea ice concentration data
##read the coordinates for the sea ice concentration file
#grdfile = bremenpath+'LongitudeLatitudeGrid-n12500-Arctic.hdf'
#hdf = SD(grdfile, SDC.READ)

#print hdf.datasets()
## Read geolocation dataset.
#blats = hdf.select('Latitudes')[:,:]
#blons = hdf.select('Longitudes')[:,:]

#print blats.shape
#print lats.shape
#exit()

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
xmc2=1000000; ymc2=400000
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

#splitting point
yms=-100000
#split A: Svalbard
mask1_a = (xp < xmc1+dd) * (xp > xmc1) * (yp>ymc1) * (yp<yms)
#south
mask2_a = (xp < xmc2+dd) * (xp > xmc2) * (yp>ymc1) * (yp<yms)
#east
mask3_a = (xp < xmc2) * (xp > xmc1) * (yp>yms) * (yp<yms+dd)
#west
mask4_a = (xp < xmc2) * (xp > xmc1) * (yp>ymc1) * (yp<ymc1+dd)

#plit b: FJL
mask1_b = (xp < xmc1+dd) * (xp > xmc1) * (yp>yms) * (yp<ymc2)
#south
mask2_b = (xp < xmc2+dd) * (xp > xmc2) * (yp>yms) * (yp<ymc2)
#east
mask3_b = (xp < xmc2) * (xp > xmc1) * (yp>ymc2) * (yp<ymc2+dd)
#west
mask4_b = (xp < xmc2) * (xp > xmc1) * (yp>yms) * (yp<yms+dd)


#make mask for the areal average in the box
mask5 = (xp < xmc2) * (xp > xmc1) * (yp>ymc1) * (yp<ymc2)
mask5_a = (xp < xmc2) * (xp > xmc1) * (yp>ymc1) * (yp<yms)
mask5_b = (xp < xmc2) * (xp > xmc1) * (yp>yms) * (yp<ymc2)


#still things are very fish here - the corner coordinated whne convereted to lonlat are all mixed up!!!
from pyproj import Proj,transform
osi_proj = Proj("+proj=stere +a=6378273 +b=6356889.44891 +lat_0=90 +lat_ts=70 +lon_0=45")
lonlat = Proj('+init=EPSG:4326')
c1lon,c1lat = transform(osi_proj,lonlat,ymc1,-xmc1)
c2lon,c2lat = transform(osi_proj,lonlat,ymc1,-xmc2)
c3lon,c3lat = transform(osi_proj,lonlat,ymc2,-xmc2)
c4lon,c4lat = transform(osi_proj,lonlat,ymc2,-xmc1)

#A
c3lon_a,c3lat_a = transform(osi_proj,lonlat,yms,-xmc2)
c4lon_a,c4lat_a = transform(osi_proj,lonlat,yms,-xmc1)


#print 'lons: ',c1lon,c2lon,c3lon_a,c4lon_a
#print 'lats: ',c1lat,c2lat,c3lat_a,c4lat_a
#exit()


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
advection_a = []
advection_b = []
m_icecon = []
m_icecon_a = []
m_icecon_b = []
dates = []
for fn in iclist:
    #break
    print fn
    dt = fn.split('_')[-1].split('.')[0]
    date = datetime.strptime(dt, "%Y%m%d%H%M")
    if date < datetime(2015,01,20):continue
    if date > datetime(2015,03,21):continue    
    
    #sea sea ice concentration data
    f = Dataset(fn)
    icecon = f.variables['ice_conc'][0,:,:]/100                                             #why is this variable masked already???
    f.close()
    
    ##Bremen sea ice concentration
    #dtsh = dt.split('1200')[0]
    #iceconfile = bremenpath+'bootstrap-AMSR2-n12500-'+dtsh+'.hdf'
    #hdf = SD(iceconfile, SDC.READ)
    ##print hdf.datasets()
    #bicecon = hdf.select('Bootstrap Ice Concentration')[:,:]

        
    #read sea ice displacement data for that day from 2 consequitive files
    dfn = glob(idpath+'ice_drift_nh_polstere-625_multi-oi_'+dt+'*.nc')
    f = Dataset(dfn[0])
    dx = f.variables['dX'][0,:,:]   #displacement in km
    dy = f.variables['dY'][0,:,:]
    f.close()
    
    #Dy SIGN REVERSED IN VERSION 1.3!!!!
    #calculate drift from first file
    u_coarse1 = dx*1000/(2*24*60*60) #sea ice velocity in m/s
    v_coarse1 = -dy*1000/(2*24*60*60)
    
    #one file before
    dt2 = datetime.strftime(date-timedelta(days=1),"%Y%m%d%H%M")
    dfn = glob(idpath+'ice_drift_nh_polstere-625_multi-oi_'+dt2+'*.nc')
    f = Dataset(dfn[0])
    dx = f.variables['dX'][0,:,:]   #displacement in km
    dy = f.variables['dY'][0,:,:]
    f.close()
    
    #Dy SIGN REVERSED IN VERSION 1.3!!!!
    #calculate drift from second file
    u_coarse2 = dx*1000/(2*24*60*60) #sea ice velocity in m/s
    v_coarse2 = -dy*1000/(2*24*60*60)
   
    #make average over the two files
    u_coarse = (u_coarse1+u_coarse2)/2
    v_coarse = (v_coarse1+v_coarse2)/2
    
    #take values just form the first file
    u_coarse = u_coarse2
    v_coarse = v_coarse2
    
    
    ##reproject the sea ice drift to the ice concentration grid (62.5 > 10km)
    ##tmp = image.ImageContainerNearest(u_coarse, area_def_coarse, radius_of_influence=70000)
    tmp = image.ImageContainerQuick(u_coarse, area_def_coarse)
    u = tmp.resample(area_def).image_data
    ##tmp = image.ImageContainerNearest(v_coarse, area_def_coarse, radius_of_influence=70000)
    tmp = image.ImageContainerQuick(v_coarse, area_def_coarse)
    v = tmp.resample(area_def).image_data
    
    #interpolate missing drift values
    #u = kd_tree.resample_nearest(area_def_coarse, u_coarse ,area_def, radius_of_influence=100000)
    #v = kd_tree.resample_nearest(area_def_coarse, u_coarse ,area_def, radius_of_influence=100000)
    
    
        
    #mask all the data
    #ic_gates = np.ma.array(icecon,mask=~gates)
    u_n = np.ma.array(u,mask=~mask1)
    u_s = np.ma.array(u,mask=~mask2)
    v_e = np.ma.array(v,mask=~mask3)
    v_w = np.ma.array(v,mask=~mask4)
    
    #A
    u_n_a = np.ma.array(u,mask=~mask1_a)
    u_s_a = np.ma.array(u,mask=~mask2_a)
    v_e_a = np.ma.array(v,mask=~mask3_a)
    v_w_a = np.ma.array(v,mask=~mask4_a)
    
    #B
    u_n_b = np.ma.array(u,mask=~mask1_b)
    u_s_b = np.ma.array(u,mask=~mask2_b)
    v_e_b = np.ma.array(v,mask=~mask3_b)
    v_w_b = np.ma.array(v,mask=~mask4_b)
    
    
    
    
    #calculate the area flux
    #calculation has to be separate for all gates - sign will be opposite depending on the gate
    #positive sign = import, negative sign = export
    flux_n = u_n * icecon
    flux_s = u_s * icecon
    flux_e = v_e * icecon
    flux_w = v_w * icecon
    
    #A
    flux_n_a = u_n_a * icecon
    flux_s_a = u_s_a * icecon
    flux_e_a = v_e_a * icecon
    flux_w_a = v_w_a * icecon
    
    
    #B
    flux_n_b = u_n_b * icecon
    flux_s_b = u_s_b * icecon
    flux_e_b = v_e_b * icecon
    flux_w_b = v_w_b * icecon
    
    
    
    #sum up for all gates
    flux = np.sum(flux_n) - np.sum(flux_s) - np.sum(flux_e) + np.sum(flux_w)
    #southeren gate has often no data
    #fluxes there are typically low, assume 0 in such cases
    if (flux_s.all() is np.ma.masked): flux = -np.sum(flux_n) - np.sum(flux_e) + np.sum(flux_w)
    flux = flux/100 #in km2/s   (/1000 to get km/s; *10 to account for 10km wide grid box)
    print flux
    #print -np.sum(flux_n), np.sum(flux_s), -np.sum(flux_e), np.sum(flux_w)
    
    #A
    flux_a = np.sum(flux_n_a) - np.sum(flux_s_a) - np.sum(flux_e_a) + np.sum(flux_w_a)
    if (flux_s_a.all() is np.ma.masked): flux_a = -np.sum(flux_n_a) - np.sum(flux_e_a) + np.sum(flux_w_a)
    flux_a = flux_a/100
    
    #B
    flux_b = np.sum(flux_n_b) - np.sum(flux_s_b) - np.sum(flux_e_b) + np.sum(flux_w_b)
    if (flux_s_b.all() is np.ma.masked): flux = -np.sum(flux_n_b) - np.sum(flux_e_b) + np.sum(flux_w_b)
    flux_b = flux_b/100
    
    #store in time series
    advection.append(flux)
    advection_a.append(flux_a)
    advection_b.append(flux_b)
    
    #mask the ice concentration    
    icecon_subset = np.ma.array(icecon,mask=~mask5)
    m_icecon.append(np.mean(icecon_subset))
    
    #A
    icecon_subset_a = np.ma.array(icecon,mask=~mask5_a)
    m_icecon_a.append(np.mean(icecon_subset_a))
    
    #B
    icecon_subset_b = np.ma.array(icecon,mask=~mask5_b)
    m_icecon_b.append(np.mean(icecon_subset_b))
    
    dates.append(date)
 
    #Plotting
    fig    = plt.figure(figsize=(10,10))
    cx      = fig.add_subplot(111)

    cx.set_title(date, fontsize=28, loc='left')

    m = Basemap(resolution='h',
                projection='stere', lat_ts=82, lat_0=90., lon_0=0.,
                llcrnrlon=0, llcrnrlat=79,
                urcrnrlon=80, urcrnrlat=80)
    
    #m = pr.plot.area_def2basemap(area_def)
    
    m.drawmapboundary(fill_color='#9999FF')
    m.drawcoastlines()
    m.fillcontinents(color='#ddaa66',lake_color='#9999FF')
    ##Draw parallels and meridians
    m.drawparallels(np.arange(80.,90.,5),labels=[1,0,0,0], fontsize=16,latmax=85.)
    m.drawmeridians(np.arange(0.,90.,20.),latmax=85.,labels=[0,0,0,1], fontsize=16)
    #m.drawmapscale(17.2, 82.4, 15, 83, 20, barstyle='fancy', units='km', fontsize=16, yoffset=None, labelstyle='simple', fontcolor='k', fillcolor1='w', fillcolor2='k', ax=None, format='%d', zorder=None)

    ##plot the ice concentration
    x, y = m(lons, lats)
    im = m.pcolormesh(x, y, icecon, cmap=plt.cm.jet)
    m.colorbar()
    
    ###plot the gates
    #x, y = m(lons, lats)
    #im = m.pcolormesh(x, y, gates, cmap=plt.cm.jet)
    lonbox = [c1lon,c2lon,c3lon,c4lon]
    latbox = [c1lat,c2lat,c3lat,c4lat]
    xbox,ybox = m(lonbox,latbox)
    xybox = zip(xbox,ybox)
    poly = Polygon(xybox, edgecolor='w', alpha=1, fill=False, linewidth=3)
    plt.gca().add_patch(poly)
    
    #A
    lonbox = [c1lon,c2lon,c3lon_a,c4lon_a]
    latbox = [c1lat,c2lat,c3lat_a,c4lat_a]
    xbox,ybox = m(lonbox,latbox)
    xybox = zip(xbox,ybox)
    poly = Polygon(xybox, edgecolor='w', alpha=1, fill=False, linewidth=3, linestyle='--')
    plt.gca().add_patch(poly)
    

    ##Robs
    #lonband = [5, 15,25,35,45,55,70,70,55,45,35,25,14,5]
    #latband = [79,79,79,79,79,79,79,83,83,83,83,83,83,83]
    #xband,yband = m(lonband,latband)
    #xyband = zip(xband,yband)
    #poly_band = Polygon( xyband, edgecolor='k', alpha=1, fill=False, linewidth=3)
    #plt.gca().add_patch(poly_band)

    #plot velocities
    x, y = m(coarse_lons, coarse_lats)
    ur,vr = Basemap.rotate_vector(m,u_coarse,v_coarse,coarse_lons, coarse_lats)
    Q = plt.quiver(x, y, ur, vr, units='width', scale=6, width=.002)
    qk = plt.quiverkey(Q, 0.9, 0.1, .1, r'$10 \frac{cm}{s}$', labelpos='E',
                   coordinates='axes',fontproperties={'size': 16},labelcolor='w',color='w')

    fig.tight_layout()
    outname = 'icecon'+dt
    fig.savefig(outpath+outname,bbox_inches='tight')

    #exit()
  
#save lists 
np.save(inpath+'advection',advection)
np.save(inpath+'advection_a',advection_a)
np.save(inpath+'advection_b',advection_b)

np.save(inpath+'m_icecon',m_icecon)
np.save(inpath+'m_icecon_a',m_icecon_a)
np.save(inpath+'m_icecon_b',m_icecon_b)

np.save(inpath+'date_flux',dates)

#load lists
advection = np.load(inpath+'advection.npy')
advection_a = np.load(inpath+'advection_a.npy')
advection_b = np.load(inpath+'advection_b.npy')

m_icecon = np.load(inpath+'m_icecon.npy')
m_icecon_a = np.load(inpath+'m_icecon_a.npy')
m_icecon_b = np.load(inpath+'m_icecon_b.npy')

dates = np.load(inpath+'date_flux.npy')

#TIME SERIES PLOT
fig2 = plt.figure(figsize=(8,8))
ax = fig2.add_subplot(312)
ax.set_ylabel(r"Area flux $(1e3km^2/day)$",fontsize=12)
##set the limits for the axis
#ax.set_xlim(start,end)
#ax.set_ylim(-5,5)
ax.grid('on')
ax.set_facecolor('.9')
ax.tick_params(labelsize=18)

#ax.plot(dates,advection)
ax.plot(dates,advection_a*24*60*60/1e3, c='orange')
#ax.plot(dates,advection_b, c='g')

#volume fluxes (mean thicknes=1.5m)
aax = ax.twinx()
aax.spines['top'].set_visible(False)
aax.spines['bottom'].set_visible(False)
#aax.set_ylabel(r'Volume flux $(km^3/s)$', fontsize=18)
aax.set_ylabel(r'Distributed heat flux $(W/m^2)$', fontsize=12)
#aax.plot(dates,advection*.0015, label='OSI-SAF')
#latent heat flux ~ as if the ice was melting/growing
rhoi=900        #kg/m3
li = 334000     #J/kg    (J=kg*m^2/s^2)
export = advection_a *1e6                    #in m^2/s
area = (xmc1-xmc2)*(ymc1-yms)                    #in m^2
fl_max = rhoi*li*export/area*1.5
fl_min = rhoi*li*export/area*1.
fl_med = rhoi*li*export/area*1.25


aax.plot(dates,fl_max, c='0.5',label='max=1.5m')               #W/m^2 (W=J/s)
aax.plot(dates,fl_min, c='0.5',label='min=1m')
aax.plot(dates,fl_med, c='orange',label='mean=1.25m')



#export = advection_b*.0015 *1000000000              #in m^3/s
#area = (xmc1-xmc2)*(yms-ymc2)                    #in m^2
#fl = rhoi*li*export/area
#aax.plot(dates,fl, c='g',ls='--')               #W/m^2 (W=J/s)

#aax.set_ylim(-380,250)



bx = fig2.add_subplot(311)
bx.set_ylabel(r"Sea ice concentration",fontsize=12)
bx.grid('on')
bx.set_facecolor('.9')
bx.tick_params(labelsize=18)

#bx.plot(dates,m_icecon, c='b', label='whole box')
bx.plot(dates,m_icecon_a, c='orange')
#bx.plot(dates,m_icecon_b, c='g', label='FJL part')

bbx = bx.twinx()
bbx.spines['top'].set_visible(False)
bbx.spines['bottom'].set_visible(False)
bbx.set_ylabel(r'Sea ice area $(1e3km^2)$', fontsize=12)
ic_area_all = area/1e6*m_icecon_a
bbx.plot(dates,ic_area_all/1e3, c='orange')


cx = fig2.add_subplot(313)
cx.set_ylabel(r"Area change $(1e3km^2/day)$",fontsize=12)
cx.grid('on')
cx.set_facecolor('.9')
cx.tick_params(labelsize=18)

ic_area_change = np.zeros_like(ic_area_all)
ic_area_change[1:] = ic_area_all[1:]- ic_area_all[:-1]      #if next time step is greater=positive value=increase=sea ice growth in leads&import
ic_area_export = advection_a *60*60*24                      #km^2/s to km^2/day
ic_area_melt = ic_area_change + ic_area_export              #km^2/day               import=increase=positive value

print ic_area_change
print ic_area_export

cx.plot(dates,ic_area_melt/1e3, c='orange')


ccx = cx.twinx()
ccx.spines['top'].set_visible(False)
ccx.spines['bottom'].set_visible(False)
ccx.set_ylabel(r'Heat flux $(W/m^2)$', fontsize=12)
fl_max = rhoi*li*ic_area_melt*1e6/(60*60*24)*1.5  /area          #flux from km^2/day to m^2/s
fl_min = rhoi*li*ic_area_melt*1e6/(60*60*24)*1./area
fl_med = rhoi*li*ic_area_melt*1e6/(60*60*24)*1.25/area
ccx.plot(dates,fl_max, c='0.5')
ccx.plot(dates,fl_min, c='0.5')
ccx.plot(dates,fl_med, c='orange')



from matplotlib.dates import MO
days = mdates.DayLocator()   			# every day
mons = mdates.WeekdayLocator(byweekday=MO) 	#every monday

#ax.xaxis.set_major_locator(mons)
#ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%b'))
ax.xaxis.set_minor_locator(days)

#bx.xaxis.set_major_locator(mons)
#bx.xaxis.set_major_formatter(mdates.DateFormatter('%d-%b'))
bx.xaxis.set_minor_locator(days)

cx.xaxis.set_major_locator(mons)
cx.xaxis.set_major_formatter(mdates.DateFormatter('%d-%b'))
cx.xaxis.set_minor_locator(days)


fig2.autofmt_xdate()

#highlight storms (all based on Lana's storm table, except the first one which is based on temeprature above -20)
#MAJOR
[whole_plot.axvspan(datetime(2015,1,21,15,0), datetime(2015,1,22,15,0), facecolor='#d8bfd8', alpha=0.4, linewidth=0) for whole_plot in [ax,bx,cx]]
[whole_plot.axvspan(datetime(2015,2,3,11,0), datetime(2015,2,8,21,0), facecolor='#d8bfd8', alpha=0.4, linewidth=0) for whole_plot in [ax,bx,cx]]
[whole_plot.axvspan(datetime(2015,2,15,12,0), datetime(2015,2,16,17,0), facecolor='#d8bfd8', alpha=0.4, linewidth=0) for whole_plot in [ax,bx,cx]]
[whole_plot.axvspan(datetime(2015,2,17,16,0), datetime(2015,2,21,4,0), facecolor='#d8bfd8', alpha=0.4, linewidth=0) for whole_plot in [ax,bx,cx]]

[whole_plot.axvspan(datetime(2015,3,2,10,0), datetime(2015,3,4,1,0), facecolor='#d8bfd8', alpha=0.4, linewidth=0) for whole_plot in [ax,bx,cx]]
[whole_plot.axvspan(datetime(2015,3,7,8,0), datetime(2015,3,8,18,0), facecolor='#d8bfd8', alpha=0.4, linewidth=0) for whole_plot in [ax,bx,cx]]
[whole_plot.axvspan(datetime(2015,3,14,21,0), datetime(2015,3,16,23,0), facecolor='#d8bfd8', alpha=0.4, linewidth=0) for whole_plot in [ax,bx,cx]]

#minor
[whole_plot.axvspan(datetime(2015,2,13,4,0), datetime(2015,2,13,9,0), facecolor='cornflowerblue', alpha=0.2, linewidth=0) for whole_plot in [ax,bx,cx]]
[whole_plot.axvspan(datetime(2015,2,22,8,0), datetime(2015,2,23,1,0), facecolor='cornflowerblue', alpha=0.2, linewidth=0) for whole_plot in [ax,bx,cx]]
[whole_plot.axvspan(datetime(2015,2,25,6,0), datetime(2015,2,25,20,0), facecolor='cornflowerblue', alpha=0.2, linewidth=0) for whole_plot in [ax,bx,cx]]

#bx.legend(loc='lower right',prop={'size':16}, fancybox=True, framealpha=.5, numpoints=1)
#ax.legend(loc='lower right',prop={'size':16}, fancybox=True, framealpha=.5, numpoints=1)
aax.legend(loc='upper right',prop={'size':10}, fancybox=True, framealpha=.5, ncol=3)
fig2.tight_layout()
fig2.savefig(outpath+'advection_ts')

#convert -delay 100 ../plots/icecon20150* ../plots/icecon_anim.gif

#scatter plots for A and B
#TIME SERIES PLOT
fig3 = plt.figure(figsize=(12,6))
ax = fig3.add_subplot(121)
ax.set_title('Svalbard part')
ax.scatter(advection_a[:-2],m_icecon_a[2:],marker='o')

bx = fig3.add_subplot(122)
bx.set_title('FJL part')
bx.scatter(advection_b[:-2],m_icecon_b[2:],marker='o')

fig3.savefig(outpath+'advection_cc')