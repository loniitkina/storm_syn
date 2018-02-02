from glob import glob
from netCDF4 import Dataset
import numpy as np
from mpl_toolkits.basemap import Basemap
import  matplotlib.pyplot as plt
from matplotlib import colors
from datetime import date, timedelta, datetime
from matplotlib.patches import Polygon
from matplotlib.path import Path
import matplotlib.dates as mdates
from storm_func import *

path_in = '../data/ArcLeads_2015/'
path_out='../plots/'

#netCDF4 file that can be read by scipy netcdf package
#downloaded from: http://hs.pangaea.de/Maps/arcleads/latlonmap.zip
fn = '../data/ArcLeads_2015/latlonmap.nc'
f = Dataset(fn)
lats = f.variables['latitude'][:]
lons = f.variables['longitude'][:]
f.close()

#data downloaded from: http://hs.pangaea.de/Maps/arcleads/ArcLeads_2015.zip
#use: unzip '*.zip'
flist= sorted(glob(path_in+'ArcLeads_*.nc'))

metfile = '../data/10minute_nounits.csv'

leadfra = []; leaddates = []; qf = []; leadfra_band = []; qf_band = []

for fn in flist:
    date = fn.split('_')[-1].split('.')[0]
    outname='leads'+date
    #break

    print date
    if int(date)<20150120:continue
    if int(date)>20150320:continue

    f = Dataset(fn)
    leads = f.variables['leadMap'][:]
    f.close()

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

    #plot the leads
    x, y = m(lons, lats)

    #discrete colorbar
    cmap=plt.cm.jet
    norm = colors.BoundaryNorm(np.arange(0,6,1), cmap.N)

    im = m.pcolormesh(x, y, leads, cmap=cmap, norm=norm)
    #0: land mask, 1: no data (cloud or open ocean), 2: sea ice, 3: artifact (high uncertainty) and 4: lead.
    cbar = m.colorbar(im, ticks=[0, 1, 2, 3, 4],location='right')
    cbar.set_ticklabels(['land', '   no data', '   sea ice', '   artifact', '   lead'])
    cbar.ax.tick_params(labelrotation=90,labelright=True,labelsize='large')
    #print help(cbar.ax.tick_params)

    #Lance postion (from Lance's met system)
    dt = datetime.strptime(date, "%Y%m%d")
    mettime = getColumn(metfile,0)
    dtb = [ datetime.strptime(mettime[i], "%Y-%m-%d %H:%M:%S") for i in range(len(mettime)) ]
    if dtb[0]>dt: continue
    if dtb[-1]<dt: continue
    mi = np.argmin(abs(np.asarray(dtb)-dt))
    lon = np.asarray(getColumn(metfile,2),dtype=float)[mi]
    lat = np.asarray(getColumn(metfile,1),dtype=float)[mi]
    if np.isnan(lon): continue
    #plot the buoy
    xl,yl = m(lon,lat)
    m.plot(xl, yl, '*', color='purple', markersize=20, markeredgewidth=2, label='Lance')

    #get 100km box around Lance
    xbox = [xl-50000, xl+50000, xl+50000, xl-50000]
    ybox = [yl-50000, yl-50000, yl+50000, yl+50000]
    xybox = zip(xbox,ybox)
    poly = Polygon( xybox, edgecolor='blue', alpha=1, fill=False, linewidth=3)
    plt.gca().add_patch(poly)
   
    #large box-band
    lonband = [5, 15,25,35,45,55,60,60,55,45,35,25,14,5]
    latband = [80,80,80,80,80,80,80,85,85,85,85,85,85,85]
    xband,yband = m(lonband,latband)
    xyband = zip(xband,yband)
    poly_band = Polygon( xyband, edgecolor='red', alpha=1, fill=False, linewidth=3)
    plt.gca().add_patch(poly_band)
       
    #legend
    lg = cx.legend(loc='upper right',prop={'size':16}, fancybox=True, framealpha=.5, numpoints=1)
    lg.get_frame().set_facecolor('white')

    fig.tight_layout()
    fig.savefig(path_out+outname,bbox_inches='tight')
    
    #cut out the area and calculate lead fraction - append to list
    path = Path(xybox)
    xall, yall = x.flatten(), y.flatten()
    xyall = np.vstack((xall,yall)).T

    #box
    mask = np.ones_like(leads, dtype=bool)
    mask = False  
    grid = path.contains_points(xyall)
    grid = grid.reshape((x.shape[0],x.shape[1]))
    mask = np.logical_or(mask,grid)
    leads_cut = np.ma.array(leads,mask=~mask)
    
    tmp = leads_cut.compressed()
    tmp1 = np.ma.array(tmp,mask=tmp<4)
    lead_grids =  np.ma.count(tmp1)
    nolead_grids = np.ma.count_masked(tmp1)
    print lead_grids, nolead_grids
    lead_fraction = float(lead_grids)/float(lead_grids+nolead_grids)
    print lead_fraction
    leadfra.append(lead_fraction)

    #flag bad data (if no data takes more than 10%)
    tmp2 = np.ma.array(tmp,mask=tmp==1)
    nodata_grids = np.ma.count_masked(tmp2)
    if float(nodata_grids)/float(lead_grids+nolead_grids)<0.25:
        qf.append(1)
    else:
        qf.append(0)

    #band
    path = Path(xyband)
    mask = np.ones_like(leads, dtype=bool)
    mask = False  
    grid = path.contains_points(xyall)
    grid = grid.reshape((x.shape[0],x.shape[1]))
    mask = np.logical_or(mask,grid)
    leads_cut = np.ma.array(leads,mask=~mask)
    
    tmp = leads_cut.compressed()
    tmp1 = np.ma.array(tmp,mask=tmp<4)
    lead_grids =  np.ma.count(tmp1)
    nolead_grids = np.ma.count_masked(tmp1)
    print lead_grids, nolead_grids
    lead_fraction = float(lead_grids)/float(lead_grids+nolead_grids)
    print lead_fraction
    leadfra_band.append(lead_fraction)

    #flag bad data (if no data takes more than 10%)
    tmp2 = np.ma.array(tmp,mask=tmp==1)
    nodata_grids = np.ma.count_masked(tmp2)
    if float(nodata_grids)/float(lead_grids+nolead_grids)<0.25:
        qf_band.append(1)
    else:
        qf_band.append(0)
    

    leaddates.append(dt)

    plt.close("all")
    del leads, leads_cut, mask
 

#save lists 
np.save(path_in+'leadfra',leadfra)
np.save(path_in+'leaddates',leaddates)
np.save(path_in+'qf',qf)

np.save(path_in+'leadfra_band',leadfra_band)
np.save(path_in+'qf_band',qf_band)

#load lists
leadfra = np.load(path_in+'leadfra.npy')
leaddates = np.load(path_in+'leaddates.npy')
qf = np.load(path_in+'qf.npy')

leadfra_band = np.load(path_in+'leadfra_band.npy')
qf_band = np.load(path_in+'qf_band.npy')

leadfra_qc = np.ma.array(leadfra,mask=qf==0)
leadfra = leadfra*100
leadfra_qc = leadfra_qc*100

leadfra_band_qc = np.ma.array(leadfra_band,mask=qf_band==0)
leadfra_band = leadfra_band*100
leadfra_band_qc = leadfra_band_qc*100

#TIME SERIES PLOT
fig2 = plt.figure(figsize=(8,8))
ax = fig2.add_subplot(111)
ax.set_ylabel(r"Lead fraction (%)",fontsize=18)
##set the limits for the axis
#ax.set_xlim(start,end)
#ax.set_ylim(-5,5)
ax.grid('on')
ax.set_facecolor('.9')
ax.tick_params(labelsize=18)

ax.plot(leaddates,leadfra, label='all data box')
ax.plot(leaddates,leadfra_qc, label='quality checked box')

ax.plot(leaddates,leadfra_band, label='all data band')
ax.plot(leaddates,leadfra_band_qc, label='quality checked band')




from matplotlib.dates import MO
days = mdates.DayLocator()   			# every day
mons = mdates.WeekdayLocator(byweekday=MO) 	#every monday

ax.xaxis.set_major_locator(mons)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%b'))
ax.xaxis.set_minor_locator(days)

fig2.autofmt_xdate()

#highlight storms (all based on Lana's storm table, except the first one which is based on temeprature above -20)
#MAJOR
[whole_plot.axvspan(datetime(2015,1,21,15,0), datetime(2015,1,22,15,0), facecolor='#d8bfd8', alpha=0.4, linewidth=0) for whole_plot in [ax]]
[whole_plot.axvspan(datetime(2015,2,3,11,0), datetime(2015,2,8,21,0), facecolor='#d8bfd8', alpha=0.4, linewidth=0) for whole_plot in [ax]]
[whole_plot.axvspan(datetime(2015,2,15,12,0), datetime(2015,2,16,17,0), facecolor='#d8bfd8', alpha=0.4, linewidth=0) for whole_plot in [ax]]
[whole_plot.axvspan(datetime(2015,2,17,16,0), datetime(2015,2,21,4,0), facecolor='#d8bfd8', alpha=0.4, linewidth=0) for whole_plot in [ax]]

[whole_plot.axvspan(datetime(2015,3,2,10,0), datetime(2015,3,4,1,0), facecolor='#d8bfd8', alpha=0.4, linewidth=0) for whole_plot in [ax]]
[whole_plot.axvspan(datetime(2015,3,7,8,0), datetime(2015,3,8,18,0), facecolor='#d8bfd8', alpha=0.4, linewidth=0) for whole_plot in [ax]]
[whole_plot.axvspan(datetime(2015,3,14,21,0), datetime(2015,3,16,23,0), facecolor='#d8bfd8', alpha=0.4, linewidth=0) for whole_plot in [ax]]

#minor
[whole_plot.axvspan(datetime(2015,2,13,4,0), datetime(2015,2,13,9,0), facecolor='cornflowerblue', alpha=0.2, linewidth=0) for whole_plot in [ax]]
[whole_plot.axvspan(datetime(2015,2,22,8,0), datetime(2015,2,23,1,0), facecolor='cornflowerblue', alpha=0.2, linewidth=0) for whole_plot in [ax]]
[whole_plot.axvspan(datetime(2015,2,25,6,0), datetime(2015,2,25,20,0), facecolor='cornflowerblue', alpha=0.2, linewidth=0) for whole_plot in [ax]]


ax.legend(loc='upper right',prop={'size':16}, fancybox=True, framealpha=.5, numpoints=1)
fig2.tight_layout()
fig2.savefig(path_out+'leads_ts')
