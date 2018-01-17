#! /usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import scipy.io
from pyresample import image, geometry, kd_tree
import pyresample as pr
import matplotlib.pyplot as plt

from storm_func import *

matfile = 'Segmented_image_20150128.mat'
fname = 'tsx20150128'

matfile = 'Segmented_image_20150207.mat'
fname = 'tsx20150207'

matfile = 'Segmented_image_20150212.mat'
fname = 'tsx20150212'




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

#plotting
#time
fig    = plt.figure(figsize=(10,10))
cx      = fig.add_subplot(111)

bmap = pr.plot.area_def2basemap(area_def)
#x, y = bmap(lons, lats)
#bmap.pcolor(x,y,val_map, origin='upper')
col = bmap.imshow(val_map, origin='upper')
bmap.colorbar()

#matched area polygon
bmap.readshapefile(polyfile,'poly', drawbounds=True)

fig.savefig(path_out+fname, bbox_inches='tight')

##save TIF 
#geotiff_file = path_out+fname+'.tif'
#save_geotiff(val_map, area_def, geotiff_file)
