#! /usr/bin/python
# -*- coding: utf-8 -*-

##send all files to ftp
from ftplib import FTP
from glob import glob
import os

ftp = FTP('osisaf.met.no')
ftp.login('anonymous','polona@npolar.no')
ftp.cwd('archive/ice/conc/2015/03/')
os.chdir('../data/OSI-SAF/icecon')
filenames = ftp.nlst() # get filenames within the directory
#print filenames

for filename in filenames:
    local_filename = filename
    file = open(local_filename, 'wb')
    ftp.retrbinary('RETR '+ filename, file.write)

    file.close()

ftp.quit()


#bzip2 -d *