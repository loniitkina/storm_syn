#! /usr/bin/python

import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from storm_func import *

#winter
start = datetime(2015, 1, 15, 0, 0)
end = datetime(2015, 3, 31, 0, 0)

path_out='../plots/'

#meteorological data (from Lance's met system)
metfile = '../data/10minute_nounits.csv'
mettime = getColumn(metfile,0)
metdates = [ datetime.strptime(mettime[x], "%Y-%m-%d %H:%M:%S") for x in range(len(mettime)) ]
mettemp = np.asarray(getColumn(metfile,6),dtype=float)
mettemp = np.ma.masked_invalid(mettemp)

#average to daily
n = 6*24
col = len(mettemp)/n
tmp = mettemp.reshape(col,n)
atmp = np.mean(tmp,axis=1)

#get the hourly dates
dtmp = metdates[::n]
wsi = np.argmin(abs(np.asarray(dtmp)-start))
wei = np.argmin(abs(np.asarray(dtmp)-end))

wtemp = atmp[wsi:wei]

t0= -1.9

#cumulative sum
fdd1 = np.cumsum(t0-wtemp)

#constant mean temperature
print 'mean winter temperature:'
print np.mean(wtemp)

wtemp=np.ones_like(wtemp)*np.mean(wtemp)
fdd2 = np.cumsum(t0-wtemp)

#constant cold=-30
wtemp=np.ones_like(wtemp)*-30
fdd3 = np.cumsum(t0-wtemp)




#FDD model (results in cm), (Lebedev, 1938)
#no initial thickness
ic1 = 1.33 * fdd1 ** 0.58
ic2 = 1.33 * fdd2 ** 0.58
ic3 = 1.33 * fdd3 ** 0.58

#plotting
fig1    = plt.figure(figsize=(6,6))
ax      = fig1.add_subplot(111)
ax.plot(ic1,c='b', label='FDD with met data')
ax.plot(ic2,c='r', label='FDD at const Tmean')
ax.plot(ic3,c='g', label='FDD at const -30')
ax.legend()
ax.set_xlabel('Day after 15th Jan')
ax.set_ylabel('Ice thickness (cm)')
fig1.savefig(path_out+'fdd')