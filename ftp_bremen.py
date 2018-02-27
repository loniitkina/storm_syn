#! /usr/bin/python
# -*- coding: utf-8 -*-

##send all files to ftp
from ftplib import FTP
from glob import glob
import os

ftp = FTP('seaice.uni-bremen.de')
ftp.set_pasv(False)
ftp.login('anonymous')
ftp.cwd('amsr2/bootstrap_daygrid/n12500/2015/jan/Arctic/')
os.chdir('../data/UniBremen/AMSR2')
filenames = ftp.nlst() # get filenames within the directory
#print filenames

for filename in filenames:
    local_filename = filename
    file = open(local_filename, 'wb')
    ftp.retrbinary('RETR '+ filename, file.write)

    file.close()

ftp.quit()


#bzip2 -d *