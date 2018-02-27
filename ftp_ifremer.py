#! /usr/bin/python
# -*- coding: utf-8 -*-

##send all files to ftp
from ftplib import FTP
from glob import glob
import os

ftp = FTP('ftp.ifremer.fr')
ftp.login('anonymous','polona@npolar.no')
ftp.cwd('ifremer/cersat/products/gridded/psi-drift/data/arctic/amsr2-merged/2-daily/netcdf/2015')
os.chdir('../data/IFREMER/AMSR2')
filenames = ftp.nlst() # get filenames within the directory
#print filenames

for filename in filenames:
    local_filename = filename
    file = open(local_filename, 'wb')
    ftp.retrbinary('RETR '+ filename, file.write)

    file.close()

ftp.quit()


#bzip2 -d *