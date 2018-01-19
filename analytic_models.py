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




#FDD model (results in cm), (Lebedev, 1938), empirial formula based on Stefan's law
#no initial thickness
ic1 = 1.33 * fdd1 ** 0.58
ic2 = 1.33 * fdd2 ** 0.58
ic3 = 1.33 * fdd3 ** 0.58

#Stefan's law
h0=0        #initial thickness [cm]
S=5;T=-15
ki=2.03 + .117 * S/T #thermal conductivity of ice at -20C [W/mK], [Maykut and Untersteiner, 1971] for S and T
ki=ki* 60*60*24 /100      #convert to days and cm, W=J/s
rhoi=.9     #ice density [g/cm**3]
L=335       #latent heat of freezing [J/g]
a =np.sqrt((2*ki)/(rhoi*L))
h_1=np.sqrt(h0**2+a**2*fdd1)

#Stefan's law with correction for atmospheric coupling - works for cases with no snow on top
kia=10  #ratio ki/ka, where ka is a heat exchange coefficient between ice and air
h_2=np.sqrt(h0**2+a**2*fdd1+kia**2)-kia

#Stefan's law with snow depth proportional to ice thickness
#thermal conductivity of snow is constant
fs = .3 #snow/ice thickness ratio; 0.3 is the flooding limit (based on Archimedes principle): (rhow-rhoi)/rhos
ks=.3* 60*60*24 /100     #convert to days and cm, W=J/s
h_3=np.sqrt(2*ki*fdd1/(rhoi*L*(1+fs*ki/ks)))

fs = .1
h_4=np.sqrt(2*ki*fdd1/(rhoi*L*(1+fs*ki/ks)))

#plotting
fig1    = plt.figure(figsize=(6,6))
ax      = fig1.add_subplot(111)
ax.plot(ic1,c='b', label='FDD with met data')
ax.plot(ic2,c='r', label='FDD at const Tmean')
ax.plot(ic3,c='g', label='FDD at const -30')

ax.plot(h_1,c='purple', label='Stefan Law')
ax.plot(h_2,c='m', label='SL, bare ice')
ax.plot(h_3,c='y', label='SL, snow=0.3*it')
ax.plot(h_4,ls='--',c='y', label='SL, snow=0.1*it')

ax.legend()
ax.set_xlabel('Day after 15th Jan')
ax.set_ylabel('Ice thickness (cm)')
fig1.savefig(path_out+'fdd')