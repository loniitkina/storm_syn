#! /usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import scipy.io
from pyresample import image, geometry, kd_tree
import pyresample as pr
import matplotlib.pyplot as plt
from shapely.geometry import Polygon as shpol
from matplotlib.path import Path

from storm_func import *

matfile = 'Segmented_image_20150128.mat'
fname = 'tsx20150128'
leadval = [9,9,9,9,12]
polyid = 1

#matfile = 'Segmented_image_20150207.mat'
#fname = 'tsx20150207'
#leadval = [6,6,7,8,15]
#polyid = 0

#matfile = 'Segmented_image_20150212.mat'
#fname = 'tsx20150212'
#leadval = [5,8,12,13,14]
#polyid = 1


path = '../data/'
reg=fname
title =fname
polyfile=path+'ice_poly'
path_out='../plots/'

mat = scipy.io.loadmat(path+matfile)['Segmented_image']
lats = mat[:,:,1]
lons = mat[:,:,2]
vals = mat[:,:,0]

#define projection and interpolate to regular grid
area_def = pr.utils.load_area('area.cfg', reg)
swath_def = geometry.SwathDefinition(lons=lons, lats=lats)
val_map = kd_tree.resample_nearest(swath_def, vals,area_def, radius_of_influence=100)

#get only leads
mask_lead = (val_map == leadval[0]) | (val_map == leadval[1]) | (val_map == leadval[2]) | (val_map == leadval[3])| (val_map == leadval[4])
#assign all unmasked areas same value
leads = np.ones_like(val_map)
leads = np.ma.array(leads,mask=~mask_lead)

leads_tiff = np.where(mask_lead,1,0)

#plotting
fig    = plt.figure(figsize=(10,10))
cx      = fig.add_subplot(111)

bmap = pr.plot.area_def2basemap(area_def)

lon,lat = area_def.get_lonlats()
x,y = bmap(lon,lat)

#col = bmap.pcolormesh(x, y, val_map)
#bmap.colorbar()
bmap.pcolormesh(x, y, leads)

#matched area polygon
bmap.readshapefile(polyfile,'poly', drawbounds=True)

fig.savefig(path_out+fname, bbox_inches='tight')


#extract a mask
path = Path(bmap.poly[1])
xall, yall = x.flatten(), y.flatten()
xyall = np.vstack((xall,yall)).T

#polygon cut
mask = np.ones_like(leads, dtype=bool)
mask = False  
grid = path.contains_points(xyall)
grid = grid.reshape((x.shape[0],x.shape[1]))
mask = np.logical_or(mask,grid)
val_cut = np.ma.array(val_map,mask=~mask)

#bmap.pcolormesh(x, y, val_cut)
#fig.savefig(path_out+fname, bbox_inches='tight')

#estimate lead fraction
tmp = val_cut.compressed()
mask_lead = (tmp == leadval[0]) | (tmp == leadval[1]) | (tmp == leadval[2]) | (tmp == leadval[3])| (tmp == leadval[4])
tmp1 = np.ma.array(tmp,mask=~mask_lead)
lead_grids =  np.ma.count(tmp1)
nolead_grids = np.ma.count_masked(tmp1)
#print lead_grids, nolead_grids
lead_fraction = float(lead_grids)/float(lead_grids+nolead_grids)
print 'lead fraction: ',lead_fraction

#estimate volume
def PolyArea(x,y):
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

poly = shpol(bmap.poly[1])
px,py = poly.exterior.coords.xy
area = PolyArea(px,py)

old_ice_vol_max = area*(1-lead_fraction)*1.5
old_ice_vol_min = area*(1-lead_fraction)*1
new_ice_vol = area*lead_fraction*.22

new_ice_fraction_max = new_ice_vol/(old_ice_vol_max+new_ice_vol)
new_ice_fraction_min = new_ice_vol/(old_ice_vol_min+new_ice_vol)

print 'new ice volume at 1.5m old ice', new_ice_fraction_max
print 'new ice volume at 1m old ice', new_ice_fraction_min

exit()

#save TIF 
geotiff_file = path_out+fname+'leads.tif'
save_geotiff(leads_tiff, area_def, geotiff_file)
