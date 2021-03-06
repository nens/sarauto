# -*- coding: utf-8 -*-

import ast
import config
import geopandas as gpd
import json
import logging
import numpy as np
import os
import pandas as pd
import re
import time
import xmltodict

from sentinelsat import SentinelAPI, read_geojson, geojson_to_wkt
from datetime import date, datetime, timedelta
from glob import glob
from shapely import wkt
from zipfile import ZipFile

logger = logging.getLogger()

# Here for convert date
def format_date(in_date):
    """Format date or datetime input or a YYYYMMDD string input to
    YYYY-MM-DDThh:mm:ssZ string format. In case you pass an
    """
    if type(in_date) == datetime or type(in_date) == date:
        return in_date.strftime("%Y-%m-%dT%H:%M:%SZ")
    else:
        try:
            return datetime.strptime(in_date, "%Y%m%d").strftime("%Y-%m-%dT%H:%M:%SZ")
        except ValueError:
            return in_date


def footprint_sar_image(image_file_path):
    """get bbox coordinates from the manifest of sentinel-1 data"""
    image_zipfile = ZipFile(image_file_path, "r")
    image_safe = image_file_path.replace(".zip", ".SAFE")
    image_safe_short = image_safe.split("\\")[-1]
    image_manifest = image_zipfile.read("{}/manifest.safe".format(image_safe_short))
    manifest_dict = xmltodict.parse(image_manifest)
    metadata_objects = manifest_dict["xfdu:XFDU"]["metadataSection"]["metadataObject"]  # [13]

    for metadata in metadata_objects:
        if "metadataWrap" in metadata:
            xmldata = metadata["metadataWrap"]["xmlData"]
            if "safe:frameSet" in xmldata:
                footprint = xmldata["safe:frameSet"]["safe:frame"]["safe:footPrint"]
                footprint_sar_image = footprint["gml:coordinates"]
                return footprint_sar_image


def rounded_coordinates(coordinates_str, decimal=1):
    coordinates_rounded = []
    coordinates = [coordinate.split(",") for coordinate in coordinates_str.split(" ")]
    for coordinate in coordinates:
        x, y = coordinate
        x_long = round(float(x) * 10 ** decimal) / 10 ** decimal
        y_lat = round(float(y) * 10 ** decimal) / 10 ** decimal
        coordinate = [y_lat,x_long]
        coordinates_rounded.append(coordinate)
    return coordinates_rounded

# Start
# Define initial date, end date, and parameter for download SAR
cwd = os.getcwd()
# read geojson and related files
input_geojson = read_geojson(cwd + config.geojson)

# logger.info(footprints[0])
type_sar = config.producttype  # can cange to SLC
orbit = config.orbitdirection

# Set Date for DTW Classifier or RF Classifier
if config.classification_mode == 1:
    start_date = format_date(config.start_date)
    end_date = format_date(config.end_date)
    print(("Step 1: Download Sentinel SAR Product in " + config.name_of_area +
           " area with select Dates between " + start_date + " and " + end_date))
    logger.info("Download Sentinel SAR Product in " + config.name_of_area +
                " area with select Dates between " + start_date + " and " + end_date)
else:
    select_date = datetime.strptime(config.rf_date, "%Y%m%d")  # type: datetime
    step = timedelta(days=12)
    select_date2 = select_date - step
    print("Step 1: Download Sentinel SAR Product in " + config.name_of_area +
          " area with select Dates between " + format_date(select_date2) + " and " + format_date(select_date))
    logger.info("Download Sentinel SAR Product in " + config.name_of_area +
                " area with select Dates between " + format_date(select_date2) + " and " + format_date(select_date))
    end_date = format_date(select_date + timedelta(days=1))
    start_date = format_date(select_date2)

url = config.url
username = config.username  # ask ITC for the username and password
password = config.password

# # Get info product
api = SentinelAPI(username, password)  # fill with SMARTSeeds user and password

footprint = geojson_to_wkt(input_geojson)
products = api.query(footprint,
                     producttype=type_sar,
                     orbitdirection=orbit,
                     date="[{0} TO {1}]".format(start_date, end_date)
                     )
dirpath = cwd + config.sentineldirpath

if not os.path.exists(dirpath):
    os.makedirs(dirpath)
api.download_all(products, directory_path=dirpath, checksum=True)

zipfiles = glob("{}*.zip".format(dirpath))

polygons = []
fid = 1
file_coordinates, file_coordinates_str = {}, {}
for image_file_path in zipfiles:
    coordinates_str = footprint_sar_image(image_file_path)
    coordinates = rounded_coordinates(coordinates_str, 1)
    date = re.findall(r"\d{8}", image_file_path)[0]
    dt_date = datetime.strptime(date, "%Y%m%d")
    date = dt_date.strftime("%Y-%m-%d")
    file_path, file_name = os.path.split(image_file_path)
    polygon = {"type": "Feature",
               "geometry": {"type": "Polygon", "coordinates": [coordinates]},
               "properties": {"fid": fid,
                              "coordinates": coordinates_str,
                              "file": str(file_name),
                              "date": date}
              }
    fid = fid + 1
    polygons.append(polygon)

    geojson = {"type": "FeatureCollection",
               "features": polygons}

# write geojson
with open("zipfiles.geojson", "w") as outfile:
    json.dump(geojson, outfile)

tiles = gpd.read_file("zipfiles.geojson")
tiles["polygon"] = tiles.geometry.astype(str)
polygon_dates = pd.DataFrame(tiles.groupby("polygon")["file"].apply(lambda x: sorted(x.values.tolist()))).reset_index()

polygon_dates["polygon_geom"] = polygon_dates["polygon"].apply(wkt.loads)
tile_dates = gpd.GeoDataFrame(polygon_dates, geometry="polygon_geom")
tile_dates["geometry_str"] = tile_dates.polygon_geom.astype(str)
tile_dates = tile_dates.rename(columns={"file":"files"})

# read geojson and related files
project_area = gpd.read_file(cwd + config.project_area)
overlay_tile_dates = gpd.sjoin(tile_dates, project_area, lsuffix="", rsuffix="_pa")
project_area.geometry = project_area.buffer(0.1)

overlay_tile_dates.files = overlay_tile_dates.files.astype(str)
project_tile_dates = overlay_tile_dates.iloc[np.r_[1:5, 6:10, 11:15], :]
project_tile_dates.to_file("project_tile_dates.geojson", driver="GeoJSON")
# Clip tile to (buffered) project area and write to tile_file
for i in range(len(project_tile_dates)):
    tile_file = cwd + config.smart_tiles + "project_tile_{}.geojson".format(i)
    tile_gdf = gpd.GeoDataFrame(project_tile_dates.iloc[[i]], geometry="polygon_geom")    
    clipped_tile_gdf = gpd.overlay(tile_gdf, project_area, how='intersection')
    clipped_tile_gdf.to_file(tile_file, driver="GeoJSON")

# list sentinel_files per geojson
# reload input geojson and get sentinel file_dates
input_geojson = read_geojson(cwd + config.geojson)
files_raw = input_geojson["features"][0]["properties"]["files"]
files_u_strings = ast.literal_eval(files_raw)
sentinel_files = [str(f) for f in files_u_strings]
sentinel_file_dates = sorted([re.findall(r"\d{8}", f)[0] for f in sentinel_files])
logger.info("sentinel_file_dates")
logger.info(sentinel_file_dates)

# Write update date to text file
update_dir = os.getcwd() + "/update_config/"
if not os.path.exists(update_dir):
    os.makedirs(update_dir)
f = open(os.getcwd() + "/update_config/update_date.txt", "w")
f.write(sentinel_file_dates[0] + "\n")
f.write(sentinel_file_dates[-1] + "\n")
f.close()

# delete all incomplete file
for item in os.listdir(dirpath):
    if item.endswith(".incomplete"):
        os.remove(os.path.join(dirpath, item))
