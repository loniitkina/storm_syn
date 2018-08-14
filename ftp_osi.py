#! /usr/bin/python
# -*- coding: utf-8 -*-

#notes: 
#20110217 is missing! >> replaced by 20110218
#20130322-25 are missing >> replaced by 20130321

#data on FTP server only available to December 2009

##send all files to ftp
from ftplib import FTP
from glob import glob
import os

ftp = FTP('osisaf.met.no')
ftp.login('anonymous','polona@npolar.no')
ftp.cwd('archive/ice/drift_lr/merged/2009/12/')
os.chdir('../data/OSI-SAF/merged')
filenames = ftp.nlst() # get filenames within the directory
#print filenames

for filename in filenames:
    if filename.split('.')[-1]!='nc':continue           #download just nc files
    if filename.split('_')[2]!='nh':continue            #download just Northern H. files
    local_filename = filename
    file = open(local_filename, 'wb')
    ftp.retrbinary('RETR '+ filename, file.write)

    file.close()

ftp.quit()


#bzip2 -d *