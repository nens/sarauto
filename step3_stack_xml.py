#!/usr/bin/env python
# coding: utf-8

import os, imp, time
import glob
import xml.etree.ElementTree as etree
from snappy import ProductIO
import datetime
import config
import logging

logger = logging.getLogger()

if config.classification_mode==1:
    target_name = config.target_name_dtw
else:
    target_name = config.target_name_rf

# Subset and Stack start...
while not os.path.exists(os.getcwd()+"/update_config/update_praprocess.txt"):
    time.sleep(0.5)
else:
    imp.reload(config)
home = os.getcwd()
dim_dir = home + config.xmlpraprocessresultstack
if not os.path.exists(dim_dir):
    os.makedirs(dim_dir)

xml_subset_exist_file = glob.glob(os.getcwd()+config.xmlpathsubset+"*.XML")
if xml_subset_exist_file:
    isFile = home + config.xmlpraprocessresultsubset
    write_data = dim_dir + target_name + '_subset_stack.dim'  # Lampung_S1A_timeseries_2018Anual_Medium
else:
    isFile = home + config.new_praprocessresult
    write_data = dim_dir + target_name + '_stack.dim'  # Lampung_S1A_timeseries_2018Anual_Medium

xml_file = glob.glob(home+config.xmlpathstack+"*.XML")
tree = etree.parse(xml_file[0])
elem = tree.findall(".//node")
indx = 1

## Get SAR from selected Date
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
xml_dir = home+config.xmlprocesspathstack
if not os.path.exists(xml_dir):
    os.makedirs(xml_dir)

list_files = []
for d_file in selected_file:
    (sarfileshortname, extension)  = os.path.splitext(d_file)
    if(d_file.endswith(".dim")):
        list_files.append(isFile+d_file)

read_data=','.join(list_files)
for entry in elem:
    try:
        if (entry.attrib["id"]=="ProductSet-Reader"):
            entry[2][0].text = read_data
        if (entry.attrib["id"]=="Write"):
            entry[2][0].text = write_data
    except:
        continue
tree.write(xml_dir+'sar_stack_.xml')
indx = indx + 1
logger.info('Make XML file for stack Graphic Processing Framework')

### Write update folder parproces to text file
f = open(os.getcwd() + "/update_config/update_subset_stack.txt", 'w')
f.write(write_data.replace('.dim',''))
f.close()