import os, sys, datetime
from datetime import timedelta, date, datetime
from scipy.io import loadmat, savemat
import numpy as np
import pandas as pd
import time
from pathlib import Path
from PIL import Image
from spectral import *
import tifffile as tiff
from tqdm import tqdm
import pandas as pd
import config
from osgeo import gdal, osr
from random_forest_function import *
from config_rf import *
import logging

logger = logging.getLogger()

home = os.getcwd()
# Make Dictionary of processing image
def main():
    # Check target file
    check_tif_file = os.getcwd() + config.rf_tif_save_dir + name_of_area + '_' + config.rf_save_name + '.tif'
    if os.path.exists(check_tif_file):
        logger.info(name_of_area + '_' + config.rf_save_name + '.tif is already in folder ' + os.getcwd() + config.rf_tif_save_dir)
        print(name_of_area + '_' + config.rf_save_name + '.tif is already in folder ' + os.getcwd() + config.rf_tif_save_dir)
        return None

    print("creating dictionoary files....")
    img = {
        'vh': [f for f in os.listdir(folder) if f.endswith('.' + 'img') and "_VH" in f],
        'vv': [f for f in os.listdir(folder) if f.endswith('.' + 'img') and "_VV" in f]}
    hdr = {
        'vh': [f for f in os.listdir(folder) if f.endswith('.' + 'hdr') and "_VH" in f],
        'vv': [f for f in os.listdir(folder) if f.endswith('.' + 'hdr') and "_VV" in f]}

    date_img = {
        'vh': [get_list_date(i) for i in img[ext[0]]],
        'vv': [get_list_date(i) for i in img[ext[1]]] }

    date_hdr = {
        'vh': [get_list_date(i) for i in hdr[ext[0]]],
        'vv': [get_list_date(i) for i in hdr[ext[1]]] }

    img_file = [find_files(date_img, img, ext, j,
                           dateStr, daysInterval) for j in range(len(date_img))]
    hdr_file = [find_files(date_hdr, hdr, ext, j,
                           dateStr, daysInterval) for j in range(len(date_hdr))]

    img_current = get_img_file(img_file, hdr_file, 0) ## 0 for current, 1, previous
    img_previous = get_img_file(img_file, hdr_file, 1) ## 0 for current, 1, previous

    print("the files are....")
    print(img_file)

    #Create features dimension
    img_features = np.zeros((img_current.shape[0], img_current.shape[1], 6))

    #Extract features
    logger.info("Extracting features")
    print("Extracting features....")

    np.seterr(divide='ignore', invalid='ignore')
    img_features[:,:,0] = img_current[:,:,0]
    img_features[:,:,1] = img_current[:,:,1]
    img_features[:,:,2] = img_current[:,:,0]/img_current[:,:,1]
    img_features[:,:,3] = img_current[:,:,0]-img_previous[:,:,0]
    img_features[:,:,4] = img_current[:,:,1]-img_previous[:,:,1]
    img_features[:,:,5] = img_current[:,:,0]/img_current[:,:,1]-\
                          img_previous[:,:,0]/img_previous[:,:,1]

    [d1,d2,d3] = img_features.shape
    logger.info("Extracting features complete")
    print("Extracting features complete....")

    #reshape and normalize features values
    print("Reshape features and normalize....")
    test_features = np.reshape(img_features,[d1*d2,d3])
    del(img_features)
    #test_features = normalize_rows(test_features)
    print("Reshape features and normalize complete....")

    #read data traning from matlab
    logger.info("Reading traning features from "+ train_path)
    print("Reading traning features from "+ train_path )
    mat = loadmat(train_path)  # load mat-file
    mdata = mat['train_mat']  # variable in mat file
    mtype = mdata.dtype
    ndata = {n: mdata[n][0,0] for n in mtype.names}
    data_headline = ndata['feature_name']
    headline = data_headline[0]


    #load data training into dataframe
    logger.info("Loading traning features into dataframe")
    print("Loading traning features into dataframe... ")
    data_raw = ndata['train_data']
    data_df = np.reshape(data_raw,[250,6]);
    #data_df = normalize_rows(data_df);
    data_df = pd.DataFrame(data_raw)
    data_test = pd.DataFrame(ndata['train_label'])


    #initiate train features and label features
    logger.info("Initiate train features and label features")
    print("Initiate train features and label features... ")
    train_features = data_df
    train_labels = data_test
    logger.info("Creating traning features is complete")
    print("Creating traning features is complete....")

    #create RF Classifier
    from sklearn.ensemble import RandomForestClassifier

    logger.info("Create RF Classifier with {} estimators".format(n_est))
    print("Create RF Classifier with {} estimators... ".format(n_est))
    rfc =  RandomForestClassifier(n_estimators = n_est, random_state = rnd_state, max_depth=max_dpth)

    rfc.fit(train_features, train_labels.values.ravel())
    rfc.score(train_features, train_labels.values.ravel())


    # Use the forest's classifier predict method on the test data
    logger.info("Predict the data using the model")
    test_features=np.nan_to_num(test_features)
    print("Size of test feature: {}".format(test_features.shape))
    datatest_chunk, index_arr = split_Arr(test_features, chunk_s)
    yc_pred = np.zeros([len(test_features)])
    print("Predict the data using the model with size per chunk: {}".format(chunk_s))
    for i in tqdm(range(0, len(datatest_chunk))):
        if i == 0:
            yc_pred[0:index_arr[i]]=rfc.predict(datatest_chunk[i])
        elif i == len(datatest_chunk)-1:
            yc_pred[index_arr[i-1]: len(yc_pred)] = rfc.predict(datatest_chunk[i])
        else:
            yc_pred[index_arr[i - 1]: index_arr[i]] = rfc.predict(datatest_chunk[i])
    # yc_pred = rfc.predict(test_features)
    arr_pred = yc_pred.reshape((d1, d2))
    ###----------- End of Classification-----------###

    ###----------- Save to MAT_FILES --------------###
    # Check Directory
    print("Save to MAT File...")
    try:
        mat_dir = home + config.rf_mat_save_dir
        if not os.path.exists(mat_dir):
            os.makedirs(mat_dir)
        # Create a dictionary for save MAT-Files
        mat_save_file = {}
        mat_save_file['growthMap'] = arr_pred
        # #
        # Saving...
        savemat(mat_dir + name_of_area +'_'+config.rf_save_name+'.mat', mat_save_file)
        logger.info("Saved successfully to MAT File")
    except:
        logger.warning("Data is too large, can not save to file *.mat!")
        logger.warning("maxsize for one variable: {}".format(sys.maxsize))
        pass
    # # ###################################################
    #
    ###----------- Save to TIF_FILES --------------###
    # Check Directory
    print("Save to GEOTIFF File...")
    tif_dir = home + config.rf_tif_save_dir
    if not os.path.exists(tif_dir):
        os.makedirs(tif_dir)
    nx = arr_pred.shape[0]
    ny = arr_pred.shape[1]
    # Set geotransform
    xmin, ymin, xmax, ymax = [min(lon), min(lat), max(lon), max(lat)]
    xres = (xmax - xmin) / float(nx)
    yres = (ymax - ymin) / float(ny)
    geotransform = (xmin, xres, 0, ymax, 0, -yres)
    # create the 1-band raster file
    dst_ds = gdal.GetDriverByName('GTiff').Create(tif_dir + name_of_area +'_'+config.rf_save_name + '.geotiff',
                                                  ny, nx, 1,
                                                  gdal.GDT_UInt16,
                                                  options = ['COMPRESS=DEFLATE'])
    dst_ds.SetGeoTransform(geotransform)  # specify coords
    srs = osr.SpatialReference()  # establish encoding
    srs.ImportFromEPSG(epsg)
    dst_ds.SetProjection(srs.ExportToWkt())  # export coords to file
    dst_ds.GetRasterBand(1).WriteArray(arr_pred)
    dst_ds.GetRasterBand(1).SetNoDataValue(9999)
    # Saving...
    dst_ds.FlushCache()  # write to disk
    dst_ds = None
    logger.info("Saved successfully to GEOTIFF File...")
    ###################################################
    return

if __name__ == "__main__":
    main()