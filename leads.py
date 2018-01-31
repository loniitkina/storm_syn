from netCDF4 import Dataset
import numpy as np
from mpl_toolkits.basemap import Basemap
import  matplotlib.pyplot as plt
from matplotlib import colors

path = '../data/'
path_out='../plots/'
outname='leads'


#downloaded from: http://hs.pangaea.de/Maps/arcleads/ArcLeads_2015.zip
fn = '../data/ArcLeads_2015/ArcLeads_20150101.nc/ArcLeads_20150101.nc'      #netCDF4 file that can be read by scipy netcdf package
f = Dataset(fn)
leads = f.variables['leadMap'][:]
f.close()

#downloaded from: http://hs.pangaea.de/Maps/arcleads/latlonmap.zip
fn = '../data/ArcLeads_2015/latlonmap/latlonmap.nc'
f = Dataset(fn)
lats = f.variables['latitude'][:]
lons = f.variables['longitude'][:]
f.close()

#plotting
fig    = plt.figure(figsize=(10,10))
cx      = fig.add_subplot(111)

m = Basemap(resolution='h',
	    projection='stere', lat_ts=82, lat_0=90., lon_0=0.,
	    llcrnrlon=-5, llcrnrlat=79,
	    urcrnrlon=70, urcrnrlat=83)

m.drawmapboundary(fill_color='#9999FF')
m.drawcoastlines()
m.fillcontinents(color='#ddaa66',lake_color='#9999FF')
##Draw parallels and meridians
m.drawparallels(np.arange(80.,86.,1),labels=[1,0,0,0], fontsize=16,latmax=85.)
m.drawmeridians(np.arange(-5.,70.,5.),latmax=85.,labels=[0,0,0,1], fontsize=16)
m.drawmapscale(17.2, 82.4, 15, 83, 20, barstyle='fancy', units='km', fontsize=16, yoffset=None, labelstyle='simple', fontcolor='k', fillcolor1='w', fillcolor2='k', ax=None, format='%d', zorder=None)

#plot the leads
x, y = m(lons, lats)

#discrete colorbar
cmap=plt.cm.jet
norm = colors.BoundaryNorm(np.arange(0,6,1), cmap.N)

im = m.pcolormesh(x, y, leads, cmap=cmap, norm=norm)
#0: land mask, 1: no data (cloud or open ocean), 2: sea ice, 3: artifact (high uncertainty) and 4: lead.
cbar = m.colorbar(im, ticks=[0, 1, 2, 3, 4],location='right')
cbar.set_ticklabels(['land', 'no data', 'sea ice', 'artifact', 'lead'])
#cbar.set_label(r'Leads',size=16)


##plot Lance track
#data = np.loadtxt(lance, dtype=np.str,delimiter=',', skiprows=1)
#latitude = np.ma.array(data[:,1],mask=data[:,1]=='0')
#longitude = np.ma.array(data[:,2],mask=data[:,2]=='0')
#date = data[:,0]
#x, y = m(longitude, latitude)
#ax.plot(x,y,'-',linewidth=4, label='R/V Lance drift track',color='purple')


#Plot buoy positons
#data = np.loadtxt(b1, dtype=np.str,delimiter=',', skiprows=1)
#latitude = np.ma.array(data[:,2])
#longitude = np.ma.array(data[:,3])
#x, y = m(longitude, latitude)
#ax.plot(x,y,'o',markersize=10,markeredgewidth=1, label='GPS drifters',color='orchid')

#cnt = 0
#data = np.loadtxt(b2, dtype=np.str,delimiter=',', skiprows=1)
#latitude = np.ma.array(data[:,2])
#longitude = np.ma.array(data[:,3])
#x, y = m(longitude, latitude)
#if cnt==0:
  #ax.plot(x,y,'o',markersize=10,markeredgewidth=1,color='hotpink',label='GPS drifters 24/04/2015'); cnt=cnt+1
#else:
  #ax.plot(x,y,'o',markersize=10,markeredgewidth=1,color='hotpink')


#lg = ax.legend(loc='upper right',prop={'size':16}, fancybox=True, framealpha=.5, numpoints=1)
#lg.get_frame().set_facecolor('white')

fig.tight_layout()
fig.savefig(path_out+outname,bbox_inches='tight')
