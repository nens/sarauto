#!/usr/bin/env python
# coding: utf-8
import os, gc, fnmatch, zipfile, glob, shutil, sys, time
from snappy import (ProductIO, GPF, jpy)
from snappy_operator import *
import xml.etree.ElementTree as etree
import config
import datetime, imp
from tqdm import tqdm
import logging

logger = logging.getLogger()


def readProd(file_sar):
    product = ProductIO.readProduct(file_sar)
    class sar:
        if 'snap.core.datamodel.Product' in str(type(product)):
            prod = product
            width = product.getSceneRasterWidth()
            height = product.getSceneRasterHeight()
            name = product.getName()
            description = product.getDescription()
        else:
            print('errors.. file not found!')
    return sar

home = os.getcwd()
isFile = home + config.sentineldirpath
OperName = home + config.snappy_ops
the_operators = open(OperName, 'r')
Operators_Result = the_operators.readlines()
for i, name in enumerate(Operators_Result):
    op = name.strip()  # type: str
    opname = op[:3]
    if i == 0:
        nameOP = opname
    else:
        nameOP = nameOP + '_' + opname

dim_dir = home + config.preprocess_result + nameOP + '/'
if not os.path.exists(dim_dir):
    os.makedirs(dim_dir)

## Get SAR from selected Date
while not os.path.isfile(home+"/update_config/update_date.txt"):
    time.sleep(0.5)
else:
    imp.reload(config)
date1 = str(config.new_start_date)
date2 = str(config.new_end_date)
start = datetime.datetime.strptime(date1, '%Y%m%d')
end = datetime.datetime.strptime(date2, '%Y%m%d')
logger.info('Get Sentinel SAR Data from selected Date ' +start.strftime('%d%b%Y') +' to '+end.strftime('%d%b%Y'))
step = datetime.timedelta(days=12)
list_date = []
list_date.append(start.strftime('%Y%m%d'))
while start <= end:
    start += step
    list_date.append(start.strftime('%Y%m%d'))

file_list = os.listdir(isFile)
selected_file = []
for ss in list_date:
    if any(ss in s for s in file_list):
        rr = [s for s in file_list if ss in s]
        selected_file.append(rr[0])

# Preprocessing Start
logger.info('Start Preprocessing GPF using snappy..')
logger.info('GPF Operator: '+' - '.join(Operators_Result).replace('\n', ''))

select_file = [entry for entry in selected_file if entry.endswith(".zip") and 'S1A' in entry]

for xi in range(0, len(select_file)):
    folder = selected_file[xi]
    logger.info('Reading ' + folder)
    print('\nReading ' + folder)
    (sarfname, extension) = os.path.splitext(folder)
    trg_path = dim_dir + sarfname + '.dim'
    # check if file is exist or not
    if os.path.exists(trg_path):
        logger.info(folder + ' already processed')
        print (folder + ' already processed')
    else:
        sar = readProd(isFile + folder)
        # create product of SNAP
        for i, name_of_operator in enumerate(Operators_Result):
            op = name_of_operator.strip()  # type: str
            if i == 0:
                logger.info('create product of SNAP for operator: '+ op)
                print('create product of SNAP for operator: '+ op)
                run_op = operatorSNAP(op).execute(sar.prod)
                continue
            run_op = operatorSNAP(op).execute(run_op.trg_data)
            logger.info('create product of SNAP for operator: ' + op)
            print('create product of SNAP for operator: ' + op)
            # run_op.trg_data.dispose()
            sar.prod.closeIO()

        # write product
        WriteProd().write_product(run_op.trg_data, trg_path)
        logger.info('Write Product: ' + str(trg_path))
        print('Write Product:' + str(trg_path))
        System = jpy.get_type('java.lang.System')
        System.gc()

### Write update folder parproces to text file
f = open(os.getcwd() + "/update_config/update_preprocess.txt", 'w')
f.write(nameOP)
f.close()