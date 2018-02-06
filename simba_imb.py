#! /usr/bin/python
# -*- coding: utf-8 -*-

from datetime import date, timedelta, datetime
import csv
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from glob import glob
import os
import pandas as pd
from storm_func import *

##get all the files
path = '../data/SIMBA/'

#some default values
octmplim = -3
cr1 = -1.9375  	#value of the reference isotherm (freezing conditions) - formation stage
cr2 = -1.8750	#melting stage (both are empirical values, only 1 step appart in the sensor resolution)
#cr2 = -1.8375
#number of thermisters to average the snow treshhold (determining the chain icing)
nta = 5
#treshhold to desnow-ice interface
cr3 = 1.1
#max abrupt jump in the interface position (10 thermistors=20cm)
jump = 10
jump_air = 8
#smoothing window
swin=6

melt = datetime(2015,5,26,0,0,0)
################################################PICK A BUOY
##FMI14
#start = datetime(2015,1,25,18,0,0)
#end = datetime(2015,2,26,12,0,0)
#melt = datetime(2015,2,20,0,0,0)
#simbaID=300234060695050;cruiseID='N-ICE'; IMBunit='FMI_14';buoyID='SIMBA_2015e';
#iair = 1; air_snow_no = 45; snow_ice_no = 45; ice_sea_no = 107; franc_bord=46 # normal chain (2cm apart)
#cr4 = 1.6
#octmplim = -10

##FMI19
#start = datetime(2015,1,24,22,0,0)
#end = datetime(2015,2,19,17,0,0)
#simbaID=300234060669770;cruiseID='N-ICE'; IMBunit='FMI_19';buoyID='SIMBA_2015f';
#iair = 1; air_snow_no = 34; snow_ice_no = 55; ice_sea_no = 100; franc_bord=54# epaisseur glace90; neige 43; franc-bord -1 a confirmer (2cm apart)
##ssfn = path+'hot_wires/'+'snowstake_floe1_1.txt'
##hwfn = path+'hot_wires/'+'hw_floe1_1.txt'
#cr4= 1.5
#jump = 20

##FMI20
#start = datetime(2015,1,29,17,0,0)
#end = datetime(2015,3,4,0,0,0)
#simbaID=300234060666760;cruiseID='N-ICE'; IMBunit='FMI_20';buoyID='SIMBA_2015g';
#iair = 1; air_snow_no = 33; snow_ice_no = 44; ice_sea_no = 109; franc_bord=56#% epaisseur glace130; neige 22; franc-bord 24 a confirmer (2cm apart) 5m chain
#cr4 = 1.7

#NPOL01
start = datetime(2015,1,15,4,0,0)
end = datetime(2015,3,16,9,59,0)
simbaID=300234060000001;cruiseID='2015N-ICE'; IMBunit='NPOL_01';buoyID='SIMBA_2015a';
iair = 1; air_snow_no = 29; snow_ice_no = 52; ice_sea_no = 118; franc_bord=53#% epaisseur glacexx; neige xx; franc-bord xx a confirmer (2cm apart)
cr4 = 1.6

##NPOL03
#start = datetime(2015,1,15,23,0,0)
#end = datetime(2015,2,16,8,59,0)
#simbaID=300234060000002;cruiseID='2015N-ICE'; IMBunit='NPOL_03';buoyID='SIMBA_2015b';
#iair = 1; air_snow_no = 37; snow_ice_no = 52; ice_sea_no = 119; franc_bord=54#% epaisseur glacexx; neige xx; franc-bord xx a confirmer (2cm apart)
#cr4 = 1.9

##NPOL05
#start = datetime(2015,3,7,16,0,0)
#end = datetime(2015,4,25,23,59,0)
#simbaID=300234061762880;cruiseID='2015N-ICE'; IMBunit='NPOL_05';buoyID='SIMBA_2015d';
#iair = 1; air_snow_no = 52; snow_ice_no = 64; ice_sea_no = 129; franc_bord=68#% epaisseur glacexx; neige xx; franc-bord xx a confirmer (2cm apart)
##it seems like all the initial interfaces have an offset of at least 4 thermistors (bad readings or bad conversion???)
#iair = 1; air_snow_no = 40; snow_ice_no = 58; ice_sea_no = 119; franc_bord=68
#cr4 = 1.5

##NPOL04
#start = datetime(2015,4,23,13,0,0)
#end = datetime(2015,6,9,10,0,0)
#melt = datetime(2015,5,26,0,0,0)
##NPOL04:
#start = datetime(2015,4,24,0,0,0)
#end = datetime(2015,6,8,0,0,0)
#simbaID=300234061762880;cruiseID='2015N-ICE'; IMBunit='NPOL_04';buoyID='SIMBA_2015c';
#iair = 1; air_snow_no = 16; snow_ice_no = 47; ice_sea_no = 138; franc_bord=48#% epaisseur glacexx; neige xx; franc-bord xx a confirmer (2cm apart)
#ssfn = path+'hot_wires/'+'snowstake_floe3_1.txt'
#hwfn = path+'hot_wires/'+'hw_floe3_1.txt'
#cr4 = 1.9
#nta=1
#swin=6
#cr1 = -1.8750
#jump_air = 5

##IAOOS4
#start = datetime(2015,1,23,14,45,0)
#end = datetime(2015,3,16,23,50,0)
#simbaID=300234061260820;cruiseID='2015N-ICE'; IMBunit='Locean_09';buoyID='SIMBA_2015h';
#iair = 1; air_snow_no = 26; snow_ice_no = 53; ice_sea_no = 130; franc_bord=54#% epaisseur glace 155; neige 55; franc-bord 2
#ssfn = path+'hot_wires/'+'snowstake_floe1_3.txt'
#hwfn = path+'hot_wires/'+'hw_floe1_4.txt'
#cr4 = 1.42
#nta=7

##IAOOS6
#start = datetime(2015,2,1,17,0,0)
#end = datetime(2015,2,21,22,0,0)
#simbaID=300234061267810;cruiseID='2015N-ICE'; IMBunit='Locean_04';buoyID='SIMBA_2015i';
#iair = 1; air_snow_no = 9; snow_ice_no = 43; ice_sea_no = 102; franc_bord=41#% pas de mesure car simple echange de valise avec la LOCEAN06!
#cr4 = 1.45
#jump = 13

##NPOL041
#start = datetime(2015,9,15,13,0,0)
#end = datetime(2015,12,30,0,0,0)
##melt = datetime(2015,5,26,0,0,0)
#simbaID=300234061762880;cruiseID='FS-2015'; IMBunit='NPOL_41';buoyID='SIMBA_2015j';
#iair = 1; air_snow_no = 16; snow_ice_no = 47; ice_sea_no = 138; franc_bord=48#% epaisseur glacexx; neige xx; franc-bord xx a confirmer (2cm apart)


#test data Bremerhaven IMB workshop


################################################END

title = buoyID
asi = air_snow_no
sii = snow_ice_no
ioi = ice_sea_no 
filename = path+IMBunit+'_td.txt'

#get the dates
time = getColumn_nh(filename,11)
dt = [ datetime.strptime(time[x], "%Y-%m-%d %H:%M:%S") for x in range(len(time)) ]

#get just the predefined date/time interval
si = np.argmin(abs(np.asarray(dt)-start))
ei = np.argmin(abs(np.asarray(dt)-end))
dates = dt[si:ei+1]

#get the data
#data from column 17:
results = csv.reader(open(filename))
data = [result[21:] for result in results][si:ei+1]
rec = getColumn_nh(filename,15)
rec = [int(i) for i in rec][si:ei+1]

#the lenght of the data is 4 temperature profiles per day (passive mode) and 1 heated cycle (active mode)
#print len(data)
#print len(rec)
#print len(dates)
#print dates[0]
#print data[0]
#print rec[0]

#test = data[0][:rec[0]]
#test = [float(i) for i in test]
#print test

#the temperature profiles are separated in two lines: one with 150 entires and 1 with 90 entires. they are both followed by 'garabage data'==-99.9
#the profiles need to be 'glued' together into 1 line and the garbage data thrown away
#the date of the both lines is identical!
#the heated cycle data are  240 entires long, there are 2 heated cycles measured every 24 hours: first one 30s aftwer heating sd the second 


tc = []
hc1 = []
hc2 = []
date_tc = []
date_hc = []
n = 0
m = 0
for i in range(0,len(data)):
  if dates[i]!= dates[-1] and dates[i]==dates[i+1]:
    print dates[i]
    fp = data[i][:rec[i]]
    sp = data[i+1][:rec[i+1]]
    #print fp
    #check if data is not rubbish (every buoy has specific failures)
    if IMBunit=='NPOL_04':
      if float(fp[0]) > 0 and float(fp[0]) != 1 and float(sp[-1])>-3:
	tc.append(fp+sp)
      else:
	tc.append(['0']*240)
    else:
      #tc.append(fp+sp)
      if float(fp[0]) < 0: #and float(fp[0]) != 1 and float(sp[-1])>-3:
	tc.append(fp+sp)
      else:
	tc.append(['0']*240)
    date_tc.append(dates[i])
    n = n+1
  elif rec[i]==240:
    if dates[i]!= dates[-1] and rec[i+1]==240:
      #print 'first measurement'
      hc1.append(data[i][:rec[i]])
      date_hc.append(dates[i])
      m = m + 1
    elif rec[i-1]==240:
      #print 'second measurement'
      hc2.append(data[i][:rec[i]])
    else:
      #add missing value
      hc1.append(data[i][:rec[i]])
      date_hc.append(dates[i])
      hc2.append(np.zeros(240))
      #print 'missing value!'
      m = m + 1
      #date_hc2.append(dates[i]+deltatime(seconds=90)

##for NPOL_03 the last measurement in the active mode is missing
#if IMBunit=='NPOL_03':
  #hc2.append(np.zeros(240))
      
#array of the temperature cycle - passive mode (now both transmissions glued together in 1 profile)  
#some other buoys might have 150+91
tc = np.array(tc,dtype=np.float).reshape(n,150+90)

#separate first and second heating cycle (active mode)
hc1 = np.array(hc1,dtype=np.float).reshape(m,240)
hc2 = np.array(hc2,dtype=np.float).reshape(m,240)

#mask out the situation where the whole chain failed (marked by 0 in the rewriting loop) and replace by average value before and after the failure
#this only works if the gap is just 1 record wide
mask = (tc==0)
tc = np.where(mask,np.nan,tc)
#tc = np.ma.array(tc,mask=mask)
tcp = np.zeros_like(tc)
tca = np.zeros_like(tc)
tcp[1:,:] = tc[:-1,:]
tca[:-1,:] = tc[1:,:]
tc[mask] = (tcp[mask]+tca[mask])/2.
tc = np.nan_to_num(tc)

#mask out the abrupt jumps in data
tcp[1:,:] = tc[:-1,:]
diff = np.zeros_like(tc)
diff[:,sii+3:] = np.abs(tcp[:,sii+3:]-tc[:,sii+3:])
diff = np.where(tcp==0,0,diff)

tca[:-1,:] = tc[1:,:]
diffa = np.zeros_like(tc)
diffa[:,sii:] = np.abs(tca[:,sii:]-tc[:,sii:])
diffa = np.where(tca==0,0,diffa)

#difference in vertical
tcu = np.zeros_like(tc)
tcu[:,1:] = tc[:,:-1]
diffv = np.zeros_like(tc)
diffv[:,sii:] = np.abs(tcu[:,sii:]-tc[:,sii:])
diffv = np.where(tcu==0,0,diffv)

asi = air_snow_no
sii = snow_ice_no
ioi = ice_sea_no 

tc_oc=np.zeros_like(tc)
tc_oc[:,ioi:] = tc[:,ioi:]
#ocean cooler than -3
mask =  (tc==0) | (tc_oc<octmplim) | (diff>3.3) | (diffv>3)
tc = np.ma.array(tc, mask=mask, fill_value=-999)

#use pandas to interpolate missing values
ts_tc = pd.Series(tc, index=date_tc)


###interpolate the masked values by averaging over neighbouring pixels
#tc[mask] = (tcp[mask]+tca[mask])/2.

#np.set_printoptions(threshold=np.nan)
#print tc[:,0] 
#exit()

#hc1 = np.ma.array(hc1, mask=hc1==0)
#hc2 = np.ma.array(hc2, mask=hc2==0)

hcr = hc2/hc1
#print hcr[:,5]
#exit()



#searching for interfaces
y = np.arange(0,tc.shape[1])
#ICE-OCEAN INTERFACE
tf = -1.8821	#freezing temperature
#cr1 = -1.9375  	#value of the reference isotherm (freezing conditions) - formation stage
#cr2 = -1.8750	#melting stage (both are empirical values, only 1 step appart in the sensor resolution)
##cr2 = -1.8375
#cr1 = -1.8750
h_oc_ice = [ioi]
for i in range(1,tc.shape[0]):			#for every profile
    #check the date (melt onset) to decide which reference isotherm to use
    if date_tc[i] < melt:   
      idx = np.argmax(tc[i,90:]>=cr1)+90
    else:
      idx = np.argmax(tc[i,90:]>=cr2)+90
    #smoothing (finding the intersection of freezing temperature and the linear fit through the the sea ice temperature profile)
    #take 5 points above the index from the previous step
    fit = np.polyfit(y[idx-5:idx],tc[i,idx-5:idx],1)
    
    #filter out high jumps
    inter = (tf-fit[1])/fit[0]
    if np.abs(inter-h_oc_ice[-1]) > 15:
      print 'high jump'
      h_oc_ice.append(h_oc_ice[-1])
    else:
      h_oc_ice.append(inter)

#print h_oc_ice
    
#if the jump is unrealistic (due to bad data that cant be filtered automatically): interpolate
h_oc_ice = np.array(h_oc_ice)
op = np.zeros_like(h_oc_ice)
oa = np.zeros_like(h_oc_ice)
#compare between 1 step before and 3 later (for large windows with errors)
op[1:] = h_oc_ice[:-1]
oa[:-3] = h_oc_ice[3:]
diff1 = np.abs(op-h_oc_ice)
diff2 = np.abs(oa-op)
mask = diff2<diff1
h_oc_ice[mask] = (op[mask]+oa[mask])/2.
#plt.plot(date_tc,h_oc_ice)

h_oc_ice = smooth(h_oc_ice,swin,window='flat')

#SNOW-ICE INTERFACE
#first value in thermal resistivity proxy that is higher than treshold
#attention: this is snow-ice interface in the hole where the IMB is deployed. If not filled up and frozen to the initial iterface, the representative interface that should be used for the freeboard calculation has a certain offset - initially the freeboard.

h_ice_sn = [sii]
idx=sii

for i in range(1,hcr.shape[0]):			#for every profile start searching at the initial first thermistor in snow 
    #check if the cycle is normal (the values inside the snow needs to be reasonable)
    #print hcr[i,asi]
    if hcr[i,asi] < cr3:
      #append idx from previous step
      h_ice_sn.append(idx)
    else:
      idx = np.argmin(hcr[i,asi:]>=cr3)+asi
      #check its not too high jump (jumps higher than 10*2cm are not realistic in 24h)
      if np.abs(idx-h_ice_sn[-1]) > jump:
	print 'high jump snow-ice'
	h_ice_sn.append(h_ice_sn[-1])
      else:
	h_ice_sn.append(idx)

#if the jump is unrealistic (due to bad data that cant be filtered automatically): interpolate
h_ice_sn = np.array(h_ice_sn)
op = np.zeros_like(h_ice_sn)
oa = np.zeros_like(h_ice_sn)
#compare between 1 step before and 3 later (for large windows with errors)
op[1:] = h_ice_sn[:-1]
oa[:-3] = h_ice_sn[3:]
diff1 = np.abs(op-h_ice_sn)
diff2 = np.abs(oa-op)
mask = diff2<diff1
h_ice_sn[mask] = (op[mask]+oa[mask])/2.
h_ice_sn = smooth(h_ice_sn,5,window='flat')
 
#interpolate to datetime of temperature profiles
ts_is = pd.Series(h_ice_sn, index=date_hc)
#here the interpolation sometimes does not work unless the interval is set very low - 1S. There mustbe some trick with the start/end data
#need to work on this later on!!!
ttmp = ts_is.resample(rule='6H',fill_method='bfill')
ts_oi = pd.Series(h_oc_ice, index=date_tc)
ts_is_6h = ttmp.reindex(ts_oi.index, method='bfill')
h_ice_sn_6h = ts_is_6h.values

#AIR-SNOW INTERFACE
#under development: snow has higher thermal resistivity than air, when there is no jump sharp difference between the snow and air, the chain is iced and value should be interpolated

h_sn_air = [asi]
idx=asi

for i in range(1,hcr.shape[0]):			#for every profile start searching at the initial first thermistor in snow 
    #check if the cycle is normal (the values inside the air needs to be reasonable, if similar to snow = icing)
    #also dont use very low values (bad cycle or missing value)
    if np.mean(hcr[i,:nta]) > cr4 or hcr[i,0] < .6:
      #append idx from previous step
      h_sn_air.append(idx)
    else:
      idx = np.argmax(hcr[i,:]>=cr4)
      #check its not too high jump (jumps higher than 10*2cm are not realistic in 24h)
      if np.abs(idx-h_sn_air[-1]) > jump_air:
	print 'high jump'
	h_sn_air.append(h_sn_air[-1])
      else:
	h_sn_air.append(idx)

#if the jump is unrealistic (due to bad data that cant be filtered automatically): interpolate
h_sn_air = np.array(h_sn_air)
op = np.zeros_like(h_sn_air)
oa = np.zeros_like(h_sn_air)
#compare between 1 step before and 3 later (for large windows with errors)
op[1:] = h_sn_air[:-1]
oa[:-3] = h_sn_air[3:]
diff1 = np.abs(op-h_sn_air)
diff2 = np.abs(oa-op)
mask = diff2<diff1
h_sn_air[mask] = (op[mask]+oa[mask])/2.
h_sn_air = smooth(h_sn_air,5,window='flat')

#interpolate to datetime of temperature profiles
ts_sn = pd.Series(h_sn_air, index=date_hc)
#same tricky time step as above!!!
ttmp = ts_sn.resample(rule='6H',fill_method='bfill')
ts_sn_6h = ttmp.reindex(ts_oi.index, method='bfill')

h_sn_air_6h = ts_sn_6h.values



##end of interfaces##################################################################################3

##snow depth
#sd = (h_ice_sn_6h - h_sn_air_6h) *2
##ice thickness 
#it = (h_oc_ice - h_ice_sn_6h) *2 #2cm spacing og the thermistor chain

##average temperatures/interface temperatures
#astemp = []
#stemp = []
#stemp_fr = []
#sitemp = []
#itemp = []
#itemp_fr = []
#iotemp = []
#otemp = []
#for i in range(0,tc.shape[0]-9):
  ###air
  ##atemp = np.mean(tc[:,:asi],axis=1)
  ###air-snow
  #astemp.append(tc[i,h_sn_air_6h[i]])
  ###snow
  #ub = int(h_sn_air_6h[i])
  #lb = int(h_ice_sn_6h[i])
  #stemp.append(np.mean(tc[i,ub:lb]))
  #frtmp = tc[i,ub:lb].filled(fill_value=-999)
  #fr_str = np.array_repr(frtmp).replace('\n', '')
  #stemp_fr.append(fr_str)
  ###snow-ice
  #sitemp.append(tc[i,h_ice_sn_6h[i]])
  ###ice
  #ub = int(h_ice_sn_6h[i])
  #lb = int(h_oc_ice[i])
  #itemp.append(np.mean(tc[i,ub:lb]))
  #frtmp = tc[i,ub:lb].filled(fill_value=-999)
  #fr_str = np.array_repr(frtmp).replace('\n', '')
  #itemp_fr.append(fr_str)
  ##ocean
  #iotemp.append(tc[i,h_oc_ice[i]])
  #otemp.append(tc[i,h_oc_ice[i]+5])

##write out timeseries
#tt = [date_tc, it, sd, astemp, stemp, stemp_fr, sitemp, itemp, itemp_fr, iotemp, otemp]
#table = zip(*tt)
##print table[0]

#outname = path + 'ice_thickness_'+buoyID+'.out'
#with open(outname, 'wb') as f:
  ##header
  #f.write(b'time, ice thickness, snow depth, air-snow temperature, snow mean temperature, snow temperature - full resolution (every 2cm), snow-ice temperature, ice mean temperature, ice temperature - full resolution (every 2cm), ice-ocean interface temperature, ocean temperature 10cm below ice\n')
  ##f.write(b'time; ice thickness; snow depth; air-snow temperature; snow mean temperature; snow temperature - full resolution (every 2cm); snow-ice temperature; ice mean temperature; ice temperature - full resolution (every 2cm); ice-ocean interface temperature; ocean temperature 10cm below ice\n')
  #np.savetxt(f, table, fmt="%s", delimiter=",")

##write out timeseries for simple use
#tt = [date_tc, it, sd, astemp, stemp, sitemp, itemp, iotemp, otemp]
#table = zip(*tt)
##print table[0]

#outname = path + 'ice_thickness_basic'+buoyID+'.out'
#with open(outname, 'wb') as f:
  ##header
  #f.write(b'time, ice thickness, snow depth, air-snow temperature, snow mean temperature, snow-ice temperature, ice mean temperature, ice-ocean interface temperature, ocean temperature 10cm below ice\n')
  ##f.write(b'time; ice thickness; snow depth; air-snow temperature; snow mean temperature; snow temperature - full resolution (every 2cm); snow-ice temperature; ice mean temperature; ice temperature - full resolution (every 2cm); ice-ocean interface temperature; ocean temperature 10cm below ice\n')
  #np.savetxt(f, table, fmt="%s", delimiter=",")


##quality checked temperature data
#tmp=np.ma.filled(tc)
#tt = [date_tc, tmp.tolist()]
#table = zip(*tt)

##print table
##exit()

#outname = path + 'temp'+buoyID+'.out'
#with open(outname, 'wb') as f:
  ##header
  #f.write(b'time, temperature - full resolution (every 2cm)\n')
  ##f.write(b'time; ice thickness; snow depth; air-snow temperature; snow mean temperature; snow temperature - full resolution (every 2cm); snow-ice temperature; ice mean temperature; ice temperature - full resolution (every 2cm); ice-ocean interface temperature; ocean temperature 10cm below ice\n')
  #np.savetxt(f, table, fmt="%s", delimiter=",")



##snow stakes and hot wire readings
#sdsd=[]
#hwsd=[]
#sdd=[]
#hwd=[]
#sd=[]
#hw=[]

#if 'ssfn' in locals():
  ##snow stakes
  #tmp = np.loadtxt(ssfn, dtype=np.str, skiprows=2)
  #tmpt = tmp[:,0]
  #sdd = [ datetime.strptime(tmpt[x], "%d.%m.%Y") for x in range(len(tmpt)) ]
  #tmps = tmp[:,1]
  #sd = [ np.float(tmps[x]) for x in range(len(tmps)) ]
  #sd = np.array(sd)/-2. + snow_ice_no
  #tmps = tmp[:,2]
  #sdsd = [ np.float(tmps[x]) for x in range(len(tmps)) ]
  #sdsd = np.array(sdsd)/-2.
#if 'hwfn' in locals():
  ##hot wires
  #tmp = np.loadtxt(hwfn, dtype=np.str, skiprows=2)
  #tmpt = tmp[:,0]
  #hwd = [ datetime.strptime(tmpt[x], "%d.%m.%Y") for x in range(len(tmpt)) ]
  #tmps = tmp[:,1]
  #hw = [ np.float(tmps[x]) for x in range(len(tmps)) ]
  #hw = np.array(hw)/2. + snow_ice_no
  #tmps = tmp[:,2]
  #hwsd = [ np.float(tmps[x]) for x in range(len(tmps)) ]
  #hwsd = np.array(hwsd)/2.

#first preview plots
#plt.imshow(tc.T)
#plt.show()
#plt.imshow(hc.T)
#plt.show()


#fancy plots
fig1 = plt.figure(figsize=(15,11))
ax = fig1.add_subplot(211)
#ax.set_title(title,fontsize=20)
ax.set_ylabel(r"Distance (m)",size=18)
#set the limits for the axis
ax.set_xlim(start,end)
ax.grid('on')
ax.set_axis_bgcolor('.9')
y = np.arange(0,tc.shape[1])
#turn around the y axis
ax.set_ylim(y[-1],y[0])
#convert thermister number to distance in m
yt = np.arange(0,tc.shape[1]*0.02,.5)
plt.yticks(y[::25], yt)
x = mdates.date2num(date_tc)
#define winter and summer color scale
if start > datetime(2015,4,1,0,0,0):
  a = -10;b=2
  a = -3;b=2
else:
  a = -30;b=-1
plt.pcolor(x,y,tc.T, cmap=plt.cm.RdYlBu_r, vmin=a, vmax=b)
cbar = plt.colorbar()
cbar.ax.set_ylabel(r'Temperature ($^\circ$C)',size=18)

#add initial interface depths
ax.axhline(asi,color='.2',linestyle='--',linewidth=3)
ax.axhline(sii,color='.2',linestyle='--',linewidth=3)
ax.axhline(ioi,color='.2',linestyle='--',linewidth=3)

#plot ice-ocean interface
ax.plot(date_tc,h_oc_ice,'w',linewidth=3)

#plot snow-ice interface
if IMBunit=='NPOL_04':
  ax.plot(date_tc[-20:],h_ice_sn_6h[-20:],'--w',linewidth=3)
  ax.axhline(sii+2,color='w',linestyle='-',linewidth=3)
else:
  ax.plot(date_tc,h_ice_sn_6h,'w',linewidth=3)

#plot air-snow interface
ax.plot(date_tc,h_sn_air_6h,'w',linewidth=3)

#dont plot dates on ax
ax.tick_params(
axis='x',          # changes apply to the x-axis
which='both',      # both major and minor ticks are affected
bottom='on',      # ticks along the bottom edge are off
top='on',         # ticks along the top edge are off
labelbottom='off')

##plot snow stake and hotwire data
#if 'ssfn' in locals():
  #ax.errorbar(sdd, sd, sdsd, linestyle='None', marker='o', color='m',label='snow stakes')
  #ax.errorbar(hwd, hw, hwsd, linestyle='None', marker='o',label='hot wires')
  #ax.legend(loc='lower left',prop={'size':15}, fancybox=True, framealpha=0.5)

bx = fig1.add_subplot(212)
bx.set_ylabel(r"Distance (m)",size=18)
#set the limits for the axis
bx.set_xlim(start,end)
bx.grid('on')
bx.set_axis_bgcolor('.9')
#turn around the y axis
bx.set_ylim(y[-1],y[0])
#convert thermister number to distance in m
plt.yticks(y[::25], yt)
x = mdates.date2num(date_hc)
if IMBunit=='NPOL_04':
  idx = np.array([2,8,10,30,31,32,33,34,35])
  mask = np.zeros_like(hcr)
  mask[idx,:] = 1
  hcr = np.ma.array(hcr,mask=mask)

plt.pcolor(x,y,hcr.T, cmap=plt.cm.RdYlBu_r, vmin=.5, vmax=2)
cbar = plt.colorbar()
cbar.ax.set_ylabel(r'Thermal resistivity proxy',size=18)

#add initial interface depths
bx.axhline(asi,color='.2',linestyle='--',linewidth=3)
bx.axhline(sii,color='.2',linestyle='--',linewidth=3)
bx.axhline(ioi,color='.2',linestyle='--',linewidth=3)

#plot ice-ocean interface
bx.plot(date_tc,h_oc_ice,'w',linewidth=3)

#plot snow-ice interface
if IMBunit=='NPOL_04':
  bx.plot(date_tc[-20:],h_ice_sn_6h[-20:],'--w',linewidth=3)
  bx.axhline(sii+1,color='w',linestyle='-',linewidth=3)
else:
  bx.plot(date_tc,h_ice_sn_6h,'w',linewidth=3)

#plot air-snow interface
bx.plot(date_tc,h_sn_air_6h,'w',linewidth=3)

fig1.tight_layout()
#ax.axes.get_xaxis().set_ticks([])
#plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=7))
#plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d-%b'))
fig1.autofmt_xdate()
fig1.savefig('../plots/simba'+buoyID, bbox_inches='tight')
