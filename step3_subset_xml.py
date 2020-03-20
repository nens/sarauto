#!/usr/bin/env python
# coding: utf-8

import os, time
import glob
import xml.etree.ElementTree as etree
from snappy import ProductIO
import datetime
import config, imp
import logging

logger = logging.getLogger()

home = os.getcwd()
xml_file = glob.glob(home+config.xmlpathsubset+"*.XML")
tree = etree.parse(xml_file[0])
elem = tree.findall(".//node")
while not os.path.exists(os.getcwd()+"/update_config/update_praprocess.txt"):
    time.sleep(0.5)
else:
    imp.reload(config)
isFile = home + config.new_praprocessresult
indx = 1


## Get SAR from selected Date
logger.info('Get SAR from selected Date')
date1 = str(config.new_start_date)
date2 = str(config.new_end_date)
start = datetime.datetime.strptime(date1, '%Y%m%d')
end = datetime.datetime.strptime(date2, '%Y%m%d')
step = datetime.timedelta(days=12)
list_date = []
while start <= end:
    list_date.append(start.strftime('%Y%m%d'))
    start += step
file_list = os.listdir(isFile)
selected_file = []
for ss in list_date:
    if any(ss in s for s in file_list):
        rr = [s for s in file_list if ss in s and s.endswith(".dim")]
        selected_file.append(rr[0])

##checkdir
xml_dir = home+config.xmlprocesspathsubset
if not os.path.exists(xml_dir):
    os.makedirs(xml_dir)

# Subset and Stack start...
dim_dir = home + config.xmlpraprocessresultsubset
if not os.path.exists(dim_dir):
    os.makedirs(dim_dir)

logger.info('Start Praprocessing GPF using GPT and XML..')
logger.info('GPF Operator: Subset-Stack')
for d_file in selected_file:
    (sarfileshortname, extension) = os.path.splitext(d_file)
    read_data = isFile + d_file
    write_data = dim_dir+sarfileshortname+'_subset'+'.dim'
    for entry in elem:
        try:
            if (entry.attrib["id"]=="Read"):
                entry[2][0].text = read_data
            if (entry.attrib["id"]=="Write"):
                entry[2][0].text = write_data
        except:
            continue
    tree.write(xml_dir + 'sar_subset_' + str(indx) + '.xml')
    indx = indx + 1
logger.info('Make XML file for subset Graphic Processing Framework')
