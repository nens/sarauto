#!/usr/bin/env python
# coding: utf-8

import ast
import os, time
import glob
import xml.etree.ElementTree as etree
from sentinelsat import SentinelAPI, read_geojson, geojson_to_wkt
from snappy import ProductIO
import datetime
import config, imp
import logging

logger = logging.getLogger()

cwd = os.getcwd()
xml_file = glob.glob(cwd+config.xmlpathsubset+"*.XML")
tree = etree.parse(xml_file[0])
elem = tree.findall(".//node")
while not os.path.exists(os.getcwd()+"/update_config/update_preprocess.txt"):
    time.sleep(0.5)
else:
    imp.reload(config)
preprocess_result_path = cwd + config.new_preprocessresult
indx = 1


# read geojson and related files
geojson_in = read_geojson(cwd + config.geojson)
files_raw = geojson_in["features"][0]["properties"]["files"]
files_u_strings = ast.literal_eval(files_raw)
geojson_files = [str(f) for f in files_u_strings]
logger.info("geojson_files")
logger.info(geojson_files)

## Get SAR from selected Date
logger.info("Get SAR from selected Date")
date1 = str(config.new_start_date)
date2 = str(config.new_end_date)
start = datetime.datetime.strptime(date1, "%Y%m%d")
end = datetime.datetime.strptime(date2, "%Y%m%d")
step = datetime.timedelta(days=12)
potential_file_dates = []
while start <= end:
    potential_file_dates.append(start.strftime("%Y%m%d"))
    start += step
files = os.listdir(preprocess_result_path)
dim_files = []

logger.info("Selected SAR files:")
logger.info(potential_file_dates)
logger.info("files")
logger.info(files)
for file_date in potential_file_dates:
    for file in files:
        if file_date in file:
            if file.endswith(".dim"):
                for geojson_file in geojson_files:
                        if geojson_file.replace(".zip", "") in file:
                            dim_files.append(file)

##checkdir
xml_dir = cwd+config.xmlprocesspathsubset
if not os.path.exists(xml_dir):
    os.makedirs(xml_dir)

# Align area of subset to area of geojson
dst_polygon = geojson_to_wkt(geojson_in)
if "MULTIPOLYGON" in dst_polygon:
    # XML subset expects POLYGON, so we select the first POLYGON
    logger.info("MULTIPOLYGON")
    dst_polygon = dst_polygon.replace("MULTIPOLYGON (((", "POLYGON ((")
    dst_polygon = dst_polygon.replace("MULTIPOLYGON(((", "POLYGON ((")
    dst_polygon = dst_polygon.replace("(((", "((")
    dst_polygon = dst_polygon.replace(")))", "))")
    dst_polygon = dst_polygon.split("))", 1)[0] + "))"
src_polygon = "POLYGON ((104.70 -5.045, 105.40 -5.045, 105.40 -5.412, 104.70 -5.412, 104.70 -5.045))"


# Subset and Stack start...
dim_dir = cwd + config.xmlpreprocessresultsubset
if not os.path.exists(dim_dir):
    os.makedirs(dim_dir)

logger.info("Start Preprocessing GPF using GPT and XML..")
logger.info("GPF Operator: Subset-Stack")
for dim_file in dim_files:
    (sarfileshortname, extension) = os.path.splitext(dim_file)
    read_data = preprocess_result_path + dim_file
    write_data = dim_dir+sarfileshortname+"_subset"+".dim"
    for entry in elem:
        try:
            if (entry.attrib["id"]=="Subset"):
                # region = entry[2][1]
                entry[2][2].text = dst_polygon
                # elem.remove(region)
            if (entry.attrib["id"]=="Read"):
                entry[2][0].text = read_data
            if (entry.attrib["id"]=="Write"):
                entry[2][0].text = write_data
        except:
            continue
    tree.write(xml_dir + "sar_subset_" + str(indx) + ".xml")
    indx = indx + 1
logger.info("Make XML file for subset Graphic Processing Framework")
