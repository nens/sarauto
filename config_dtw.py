import snappy
from snappy import (ProductIO, GPF, jpy)
import os, re
import glob
import imp,sys
import math
from scipy.io import loadmat
import config
import time
import warnings
import datetime
import logging
warnings.filterwarnings("ignore")

logger = logging.getLogger()

home = os.getcwd()
name_of_area = config.name_of_area
while not os.path.exists(home+"/update_config/update_subset_stack.txt"):
    time.sleep(0.5)
else:
    imp.reload(config)

def readProd(file_sar):
    product = ProductIO.readProduct(file_sar)
    class sar:
        if "snap.core.datamodel.Product" in str(type(product)):
            prod = product
            width = product.getSceneRasterWidth()
            height = product.getSceneRasterHeight()
            name = product.getName()
            band1 = str(product.getBandNames()[0]).split("_db")[0]
            band2 = str(product.getBandNames()[1]).split("_db")[0]            
            crs_code = product.getSceneCRS()
        else:
            print("errors.. file not found!")
    return sar

def readlatlonProd(prod, pix1, pix2):
    geoPosType = jpy.get_type("org.esa.snap.core.datamodel.GeoPos")
    geocoding = prod.getSceneGeoCoding()
    geo_pos = geocoding.getGeoPos(snappy.PixelPos(pix1, pix2), geoPosType())
    return (geo_pos.lat, geo_pos.lon)

### Read Information of Sentinel Product
product_location =  ( config.new_sub_stack_result +".dim")
Product_SAR = readProd(product_location)

### get lat and lon of product
lat1, lon1 = readlatlonProd(Product_SAR.prod, 0, 0)
lat2, lon2 = readlatlonProd(Product_SAR.prod, Product_SAR.height, 0)
lat3, lon3 = readlatlonProd(Product_SAR.prod, 0, Product_SAR.width)
lat4, lon4 = readlatlonProd(Product_SAR.prod, Product_SAR.height, Product_SAR.width)
lat = [lat1, lat4] # Lampung lat = [-5.045, -5.747]
lon = [lon1, lon4] # Lampung lot = [104.700, 105.066]

### get CRS product
def epsg_from_crs(listcrs):
    for i in listcrs:
        if "central_meridian" in i:
            central_meridian = int(re.findall(r"\d+", i)[0])
            zone = int(math.floor((central_meridian + 180) / 6.0) + 1)
            if "South" in listcrs[0]:
                epsg_code = 32700 + zone
            else:
                epsg_code = 32600 + zone
    return epsg_code


listcrs = str(Product_SAR.crs_code).split("\r\n")
if "UTM Zone" in listcrs[0]:
    epsg_code = epsg_from_crs(listcrs)
else:
    epsg_code = listcrs[len(listcrs)-1]
epsg_code = 4326 # epsg_code # 4326
logger.info(epsg_code)

"""
1. Locate to "Vegetable_classification" folder, and load matlab files
   regarding the crop pattern of vegetable types
"""
logger.warning("All Input Data must Exist regarding the crop pattern of vegetable types!")
folder_mat_file = home + config.mat_files
folder_mask_file = home + config.mask_files

# MAT Files
try:
    train_mat = loadmat(glob.glob(folder_mat_file + "*.mat")[0])
    par_DTW = train_mat["par_DTW"][0]
    tsTrain = train_mat["tsTrain"][0]
    clsNames = train_mat["clsNames"][0]
except Exception:
    print("Selected *.MAT file is wrong or No File Exist in {} directory".format(folder_mat_file))
    logger.error("Selected *.MAT file is wrong or No File Exist in {} directory".format(folder_mat_file))
    raise

# Mask Files
try:
    cropROI_path = glob.glob(folder_mask_file + "*.tif")[0]
except:
    cropROI_path = ""

"""
2. Specify parameters for the time series classification algorithm
"""
logger.warning("Please specify parameters for the time series classification algorithm!")
ts_stack_foler = config.new_sub_stack_result+".data"
cropROI_path = cropROI_path
time_formatStr= config.time_formatDTW
startDate_ts = datetime.datetime.strptime(config.new_start_date, "%Y%m%d").strftime(time_formatStr) # get from update time series sar product
endDate_ts =datetime.datetime.strptime(config.new_end_date,  "%Y%m%d").strftime(time_formatStr)# get from update time series sar product
selectDates = config.select_date
daysInterval = config.days_of_intrvl
splitStr = config.splt_Str
R = Product_SAR.height # get from sar product
C = Product_SAR.width # get from sar product
bandName = [Product_SAR.band1, Product_SAR.band2] # get from sar product
block_size = config.block_size
doFilter = config.doFilter
folder= ts_stack_foler
K_value = config.K_val
epsg = epsg_code #4326 #int(filter(str.isdigit, epsg_code))
