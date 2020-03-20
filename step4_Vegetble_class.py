#!/usr/bin/env python
# coding: utf-8
from __future__ import print_function
from datetime import datetime
from scipy.io import savemat
from math import floor as floor
from pathlib import Path
from spectral import *
import tifffile as tiff
from osgeo import gdal, osr
import statsmodels.api as sm
from time_series_function import *
from config_dtw import *
import warnings
import logging

logger = logging.getLogger()

warnings.filterwarnings('ignore')
home = os.getcwd()
name_of_area= config.name_of_area

###---------------- Main Function ------------------------###
def main():
    # Check target file
    check_tif_file = os.getcwd() + config.dtw_tif_save_dir + name_of_area +'_'+ config.dtw_save_name + '.geotiff'
    if os.path.exists(check_tif_file):
        logger.info(name_of_area +'_'+config.dtw_save_name + '.geotiff is already in folder '+os.getcwd() + config.dtw_tif_save_dir)
        print(name_of_area +'_'+config.dtw_save_name + '.geotiff is already in folder '+os.getcwd() + config.dtw_tif_save_dir)
        return
    logger.info('Start Classification of Time Series Data using Dynamic Time Warping')
    print('Start Classification of Time Series Data using Dynamic Time Warping')
    ###################################################
    #  Validate parameters
    ###################################################
    logger.warning("The input time series have to be converted into dB, otherwise the computation will be wrong!")
    #warnings.warn("The input time series have to be converted into dB, otherwise the computation will be wrong!")

    try:
        assert(bandName[0]=='Sigma0_VH' or bandName[1]=='Sigma0_VV')
    except:
        logger.error("The first band name should be [Sigma0_VH], and the second band name should be [Sigma0_VV]")
        print("The first band name should be [Sigma0_VH], and the second band name should be [Sigma0_VV]")
        return
    logger.info('Check Landcover mask')
    print('Check Landcover mask')
    if os.path.isfile(cropROI_path):
        print('Landcover mask Exist')
        logger.info('Landcover mask File is Exist')
        im = tiff.imread(cropROI_path)
        cropROI  = np.array(im)
        # print(cropROI.shape)
        cropROI = cropROI[:R,:C] ####### if dimension of img file smaller than mask file(fix different dimension from mat files)
        cropROI[np.where( cropROI != 1 )] = 0
        if cropROI.shape[0] != R and cropROI.shape[1] != C:
            logger.error('The size of cropROI does not match the size of input time series images!')
            print('The size of cropROI does not match the size of input time series images!')
            # need system break or not?
            return
    else:
        print('Landcover mask not Exist')
        logger.info('Landcover mask File is not Exist')
        cropROI = np.array([])

    # 2.1 Categorize files---------
    logger.info('Categorize files')
    print('Categorize files')
    fC_img = filesCategorizeByDate(folder, 'img', bandName, splitStr, startDate_ts, daysInterval, endDate_ts, time_formatStr)
    fC_hdr = filesCategorizeByDate(folder, 'hdr', bandName, splitStr, startDate_ts, daysInterval, endDate_ts, time_formatStr)

    # 1.2 Extract a subset of time series images
    logger.info('Extract a subset of time series images')
    print('Extract a subset of time series images')
    tsList = []
    ts_listAll = fC_img['datetime'][0:]

    if len(selectDates) != 0:
        d1 = nearest_date(ts_listAll, datetime.datetime.strptime(str(selectDates[0]), time_formatStr))
        d2 = nearest_date(ts_listAll, datetime.datetime.strptime(str(selectDates[1]), time_formatStr))
        #find index
        low_idx = fC_img['datetime'].index(d1)
        upp_idx = fC_img['datetime'].index(d2)
        tsList = fC_img['datetime'][low_idx:upp_idx+1]
        fC_img = {'datetime': tsList,
                  bandName[0]:fC_img[bandName[0]][low_idx:upp_idx+1],
                  bandName[1]:fC_img[bandName[1]][low_idx:upp_idx+1]}
        fC_hdr = {'datetime': tsList,
                  bandName[0]:fC_hdr[bandName[0]][low_idx:upp_idx+1],
                  bandName[1]:fC_hdr[bandName[1]][low_idx:upp_idx+1]}
    else:
        tsList = ts_listAll

    tsLen = len(tsList)
    if tsLen < 2:
        logger.error('The time series folder does not contain multi-temporal images!')
        print('The time series folder does not contain multi-temporal images!')
        return

    # 1.3 Check consistence of time series images
    logger.info('Check consistence of time series images')
    print('Check consistence of time series images')
    tsList_valid = fC_hdr['datetime']
    if  tsList != tsList_valid:
        print('The time series images are NOT consistent in time line!')
        logger.warning('The time series images are NOT consistent in time line!')
    elif  len(tsList_valid) < len(tsList):
        print("The time series stack contains - %d - empty files" % (len(tsList)-len(tsList_valid)))
        logger.warning("The time series stack contains - %d - empty files" % (len(tsList)-len(tsList_valid)))
    else:
        print('The time series images are consistent in time line!')
        logger.info('The time series images are consistent in time line!')
    #
    # [3] Read and classify pixel values of time series images----------------
    logger.info('Read and classify pixel values of time series images, always conduct block-based processing')
    # 3.1 Always conduct block-based processing
    # block_size | R | C
    print('Read and classify pixel values of time series images, always conduct block-based processing')
    [block_row_range, block_col_range, num_r2, num_c2] = blocking_im(block_size, R, C)
    rowcol = (R, C)
    cropMap = np.zeros(rowcol)
    minDist = np.zeros(rowcol)

    #.3 Read image features for each blocks -----------------------------Progressbar using tqdm
    logger.info('Read image features for each blocks')
    print('Read image features for each blocks')
    for i in range(0, num_r2):
        for j in range(0, num_c2):
            logger.info('Obtain subset image for block row:{}-column:{}'.format(i+1, j+1))
            logger.info('   Start classification in time series subset image')
            print('Obtain subset image for block row:{}-column:{}'.format(i+1, j+1))
            print('Start classification in time series subset image')
            tstart = time.time()
            Row_range = block_row_range[i, j]
            Col_range = block_col_range[i, j]
            fea_im_ij = obtain_fea_im_subset(Row_range, Col_range, tsList, ts_stack_foler, fC_hdr, fC_img, bandName)
            if doFilter > 0:  # applaying savgol filter
                logger.info('   applying temporal filter using savgol filter')
                fea_im_ij = temporal_filter_jit(fea_im_ij)
            if cropROI.any():
                cropROI_ij = cropROI[(Row_range[0] - 1):Row_range[1], (Col_range[0] - 1):Col_range[1]]
            else:
                cropROI_ij = np.ones((Row_range[1] - Row_range[0] + 1, Col_range[1] - Col_range[0] + 1))

            # 4.2 Classification
            #logger.info('Start classification for each blocks')
            [clsIm_ij, distIm_ij] = twDTW_spring_subset_wise(fea_im_ij, tsList, cropROI_ij, tsTrain, par_DTW, daysInterval, K_value)
            cropMap[Row_range[0] - 1:Row_range[1], Col_range[0] - 1:Col_range[1]] = clsIm_ij
            minDist[Row_range[0] - 1:Row_range[1], Col_range[0] - 1:Col_range[1]] = distIm_ij
            cropMap = cropMap.astype(int)
            logger.info("   Classification for blocks with row:{}, and column:{} is done!".format(i+1,j+1))
            logger.info("   The elapsed time is {0:.2f} minutes".format((time.time() - tstart)/60))
            print("   Classification for blocks with row:{}, and column:{} is done!".format(i+1,j+1))
            print("   The elapsed time is {0:.2f} minutes".format((time.time() - tstart) / 60))
    logger.info('End of Classification')
    ###----------- End of Classification-----------###

    ###----------- Save to MAT_FILES --------------###
    print("Save to MAT File...")
    # Check Directory
    mat_dir = home + config.dtw_mat_save_dir
    if not os.path.exists(mat_dir):
        os.makedirs(mat_dir)
    # Create a dictionary for save MAT-Files
    mat_save_file = {}
    mat_save_file['cropMap'] = cropMap
    mat_save_file['minDist'] = minDist
    # Saving...
    savemat(mat_dir + name_of_area +'_'+config.dtw_save_name+'.mat', mat_save_file)
    logger.info("Saved successfully to MAT File")
    ###################################################

    ###----------- Save to TIF_FILES --------------###
    print("Save to GEOTIFF File...")
    # Check Directory
    tif_dir = home + config.dtw_tif_save_dir
    if not os.path.exists(tif_dir):
        os.makedirs(tif_dir)
    nx = cropMap.shape[0]
    ny = cropMap.shape[1]
    # Set geotransform
    xmin, ymin, xmax, ymax = [min(lon), min(lat), max(lon), max(lat)]
    xres = (xmax - xmin) / float(nx)
    yres = (ymax - ymin) / float(ny)
    geotransform = (xmin, xres, 0, ymax, 0, -yres)
    # create the 1-band raster file
    dst_ds = gdal.GetDriverByName('GTiff').Create(tif_dir +name_of_area +'_'+ config.dtw_save_name+'.geotiff',
                                                  ny, nx, 1,
                                                  gdal.GDT_UInt16,
                                                  options = ['COMPRESS=DEFLATE'])
    dst_ds.SetGeoTransform(geotransform)  # specify coords
    srs = osr.SpatialReference()  # establish encoding
    srs.ImportFromEPSG(epsg)
    dst_ds.SetProjection(srs.ExportToWkt())  # export coords to file
    dst_ds.GetRasterBand(1).WriteArray(cropMap)
    dst_ds.GetRasterBand(1).SetNoDataValue(9999)
    # Saving...
    dst_ds.FlushCache()  # write to disk
    dst_ds = None
    logger.info("Saved successfully to GEOTIFF File...")
    ###################################################
    return

### ------------Built in function -------------####

def filesCategorizeByDate(folder, extension, headStr, splitStr, startTSstr, daysInterval, endTSstr, time_formatStr):
    numC = len(headStr)
    files = [f for f in os.listdir(folder) if f.endswith('.' + extension)]
    num_f = len(files)

    # We first get all the dates
    startTS = datetime.datetime.strptime(startTSstr, time_formatStr)  # .strftime(time_formatStr)
    endTS = datetime.datetime.strptime(endTSstr, time_formatStr)  # .strftime(time_formatStr)

    if (((endTS - startTS).days % daysInterval) != 0):
        print("'Wrong endTS or startTS!'")
    else:
        ts_num = (endTS - startTS).days / daysInterval + 1
    fC = {
        'datetime': [None] * (num_f / 2),
        headStr[0]: [None] * (num_f / 2),
        headStr[1]: [None] * (num_f / 2)}
    for i in range(0, num_f):
        temp = files[i].split('.')[0].split(splitStr)
        dt_i = datetime.datetime.strptime(temp[len(temp) - 1], '%d%b%Y')  # temp[len(temp)-1]
        tsIX = (dt_i - startTS).days / daysInterval
        fC['datetime'][tsIX] = dt_i
        for j in range(0, numC):
            band_i = temp[0] + '_' + temp[1]
            if band_i == headStr[j]:
                fC[headStr[j]][tsIX] = files[i]
    return (fC)

def nearest_date(items, pivot):
    return min(items, key=lambda x: abs(x - pivot))

def blocking_im(block_size, R, C):  # block_size | R | C
    if not (block_size):
        block_size = [R, 50]
        logger.warning('The block size uses the default, %d-by-%d, which might not be the best choice!' % (
        block_size[0], block_size[1]))
        print('The block size uses the default, %d-by-%d, which might not be the best choice!' % (
        block_size[0], block_size[1]))
    row = int(block_size[0])
    col = int(block_size[1])
    num_r = int(floor(R / row))
    num_c = int(floor(C / col))
    rem_r = int(R % row)
    rem_c = int(C % col)
    # 4.2 Decide the row,col margins, which have size smaller than Block_size
    if rem_r > 0:
        num_r2 = num_r + 1
    else:
        num_r2 = num_r
    if rem_c > 0:
        num_c2 = num_c + 1
    else:
        num_c2 = num_c;
    block_row_range = np.empty(shape=(int(num_r2), int(num_c2)),
                               dtype=list)  # np.zeros(shape=(int(num_r2),int(num_c2)))
    block_col_range = np.empty(shape=(int(num_r2), int(num_c2)),
                               dtype=list)  # np.zeros(shape=(int(num_r2),int(num_c2)))
    for m in range(0, num_r2):
        for n in range(0, num_c2):
            if m + 1 <= num_r and n + 1 <= num_c:
                Row_range = [(m) * row + 1, (m + 1) * row]
                Col_range = [(n) * col + 1, (n + 1) * col]
            elif m + 1 <= num_r and n + 1 > num_c:
                Row_range = [(m) * row + 1, (m + 1) * row]
                Col_range = [(n) * col + 1, C]
            elif m + 1 > num_r and n + 1 <= num_c:
                Row_range = [(m) * row + 1, R]
                Col_range = [(n) * col + 1, (n + 1) * col]
            else:
                Row_range = [(m) * row + 1, R]
                Col_range = [(n) * col + 1, C]
            block_row_range[m, n] = Row_range
            block_col_range[m, n] = Col_range
    return block_row_range, block_col_range, num_r2, num_c2

def obtain_fea_im_subset(Row_range, Col_range, tsList, ts_stack_foler, fC_hdr, fC_img, bandName):
    """
        fea_im_subset = eng.zeros(int(tsLen),int(num_fea),int(d2),int(d1)) # this is matlab.double type
        to convert matlab.double to ndarray
        For one-dimensional arrays, access only the "_data" property of the Matlab array.
        For multi-dimensional arrays you need to reshape the array afterwards.
        np.array(x._data).reshape(x.size[::-1]).T
    """
    d1 = Row_range[1] - Row_range[0] + 1
    d2 = Col_range[1] - Col_range[0] + 1
    num_band = len(bandName)
    num_fea = num_band + 1
    tsLen = len(tsList)
    dim_fea = (d1, d2, num_fea, tsLen)
    fea_im_subset = np.zeros(dim_fea)
    FILL = 0
    for t in range(0, tsLen):
        for i in range(0, num_band):
            hdrPath_t_i = ts_stack_foler + '/' + fC_hdr[bandName[i]][t]
            imgPath_t_i = ts_stack_foler + '/' + fC_img[bandName[i]][t]
            # print(imgPath_t_i)
            # --------------------Check file existance--------------------------
            if Path(hdrPath_t_i).is_file():
                info = envi.read_envi_header(hdrPath_t_i)
                img = envi.open(hdrPath_t_i)
                img_open = img.open_memmap(writeable=True)
                im_t_i = img_open[Row_range[0] - 1:Row_range[1], Col_range[0] - 1:Col_range[1], 0]
                # print(im_t_i.shape)
                # im_t_i = np.copy(img_open[:Row_range[1]+1,:Col_range[1],0])
            else:
                print("The -%d-th TS is empty!!!" % t)
                im_nan = np.zeros((d1, d2))
                im_nan[im_nan == 0] = np.nan
                im_t_i = im_nan
                FILL = 1
            # fea_im_subset[t][i][:][:]
            fea_im_subset[:, :, i, t] = im_t_i
    if FILL == 1:
        nans, x = np.isnan(fea_im_subset), lambda z: z.nonzero()[0]
        fea_im_subset[nans] = np.interp(x(nans), x(~nans),
                                        fea_im_subset[~nans])  # linear Interpolation, in Matlab: Cubic spline
        # fea_im_subset = eng.fillmissing(fea_im_subset,'spline');
    for t in range(0, tsLen):
        # Calculate additional features ---------------------------------------
        fea_im_subset[:, :, num_band, t] = fea_im_subset[:, :, 1, t] - fea_im_subset[:, :, 0, t]
    return fea_im_subset

def temporal_filter(im_mat):
    [d1, d2, num_fea, tsLen] = im_mat.shape
    im_mat_filtered = im_mat
    for f in range(0, num_fea):
        im_mat_f = np.squeeze(im_mat[:, :, f, :])
        im_mat_f_res = im_mat_f.reshape(d1 * d2, tsLen)
        for i in range(0, (d1 * d2)):  # haven't implemented parallelization
            im_mat_f_res[i, :] = savgol_filter(im_mat_f_res[i, :], 5, 2)
        im_mat_filtered[:, :, f, :] = im_mat_f_res.reshape(d1, d2, tsLen)  # reshape(im_mat_f,[d1,d2,1,tsLen])
    return im_mat_filtered
temporal_filter_jit = numba.jit(temporal_filter)

if __name__ == "__main__":
    main()