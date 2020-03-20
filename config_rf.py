import snappy, os, datetime
from datetime import datetime, timedelta, date
from snappy import (ProductIO, GPF, jpy)
from scipy.io import loadmat, savemat
import numpy as np
import pandas as pd
import time, imp
from pathlib import Path
import config
import logging

logger = logging.getLogger()

logger.warning("All Input Data must Exist!")

home = os.getcwd()
name_of_area = config.name_of_area
while not os.path.exists(home+"/update_config/update_subset_stack.txt"):
    time.sleep(0.5)
else:
    imp.reload(config)

def readProd(file_sar):
    product = ProductIO.readProduct(file_sar)
    class sar:
        if 'snap.core.datamodel.Product' in str(type(product)):
            prod = product
            width = product.getSceneRasterWidth()
            height = product.getSceneRasterHeight()
            name = product.getName()
            crs_code = product.getSceneCRS()
        else:
            print('errors.. file not found!')
    return sar

def readlatlonProd(prod, pix1, pix2):
    geoPosType = jpy.get_type('org.esa.snap.core.datamodel.GeoPos')
    geocoding = prod.getSceneGeoCoding()
    geo_pos = geocoding.getGeoPos(snappy.PixelPos(pix1, pix2), geoPosType())
    return (geo_pos.lat, geo_pos.lon)

### Read Information of Sentinel Product
product_location =  (config.new_sub_stack_result +'.dim')
Product_SAR = readProd(product_location)

### get lat and lon of product
lat1, lon1 = readlatlonProd(Product_SAR.prod, 0, 0)
lat2, lon2 = readlatlonProd(Product_SAR.prod, Product_SAR.height, 0)
lat3, lon3 = readlatlonProd(Product_SAR.prod, 0, Product_SAR.width)
lat4, lon4 = readlatlonProd(Product_SAR.prod, Product_SAR.height, Product_SAR.width)
lat = [lat1, lat4] # Lampung lat = [-5.045, -5.747]
lon = [lon1, lon4] # Lampung lot = [104.700, 105.066]

### get CRS product
listcrs = str(Product_SAR.crs_code).split('\r\n')
epsg_code = listcrs[len(listcrs)-1]

# Parameterss
dateStr = datetime.strptime(config.rf_date, '%Y%m%d').strftime('%d%b%Y')
ext1 = 'img'
ext2 = 'hdr'
splitStr = '_'
daysInterval = 12
time_formatStr = '%d%b%Y'
bandName = ['Sigma0_VH','Sigma0_VV']
ext = ['vh', 'vv']
train_path = home + config.train_path
folder = config.new_sub_stack_result+'.data'
numfiles = len(filter(lambda x: "Sigma0_" in x, os.listdir(folder)))/4
epsg = int(filter(str.isdigit, epsg_code))
chunk_s = config.chunk_size
n_est = config.n_estimators
rnd_state = config.random_state
max_dpth = config.max_depth