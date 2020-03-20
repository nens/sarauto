import os
from datetime import timedelta, date
from scipy.io import loadmat, savemat
import pandas as pd
import time
from pathlib import Path
from PIL import Image
from spectral import *
import tifffile as tiff
from tqdm import tqdm
import pandas as pd
import config
import numpy as np
from config_rf import *

# Built in Function
def nearest_date(items, pivot):
    return min(items, key=lambda x: abs(x - pivot))

def get_list_date(files):
    temp = files.split('.')[0].split('_')
    return datetime.strptime(temp[len(temp)-1], '%d%b%Y')

def find_files(list_date, files, ext, j, dateStr, daysInterval):
    date_curr = nearest_date(list_date[ext[j]], datetime.strptime(str(dateStr), '%d%b%Y'))
    date_prev = date_curr - timedelta(daysInterval)
    idx_curr = list_date[ext[j]].index(date_curr)
    idx_prev = list_date[ext[j]].index(date_prev)
    return files[ext[j]][idx_curr], files[ext[j]][idx_prev]

def get_img_file(img_file, hdr_file, val):
    hdr_vh = folder +'/'+hdr_file[0][val]
    img_vh = folder +'/'+img_file[0][val]
    hdr_vv = folder +'/'+hdr_file[1][val]
    img_vv = folder +'/'+img_file[1][val]
    l = [[hdr_vh, img_vh], [hdr_vv, img_vv]]
    if Path(hdr_vh).is_file() and Path(img_vh).is_file():
        if Path(hdr_vv).is_file() and Path(img_vv).is_file():
            get_info = envi.read_envi_header(hdr_vh)
            d1 = int(get_info['lines'])
            d2 = int(get_info['samples'])
            dim = (d1,d2,2)
            im_subset = np.zeros((dim))
    for ix in tqdm(range(len(l))):
        get_img = envi.open(l[ix][0])
        img_open = get_img.open_memmap(writeable = True)
        im = img_open[:d1,:d2,0]
        im_subset[:,:,ix] = im
    return im_subset

def normalize_rows(x):
    """
    function that normalizes each row of the matrix x to have unit length.

    Args:
     ``x``: A numpy matrix of shape (n, m)

    Returns:
     ``x``: The normalized (by row) numpy matrix.
    """
    return x/np.linalg.norm(x, ord=2, axis=1, keepdims=True)

def index_marks(nrows, chunk_size):
    return range(1 * chunk_size, (nrows // chunk_size + 1) * chunk_size, chunk_size)

def split_Arr(Arr, chunk_size):
    indices = index_marks(Arr.shape[0], chunk_size)
    return np.split(Arr, indices), indices