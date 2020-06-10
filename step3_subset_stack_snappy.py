#!/usr/bin/env python
# coding: utf-8
import logging, os, gc, fnmatch, zipfile, glob, shutil, sys, time
from snappy import (ProductIO, GPF, jpy)
from snappy_operator import *
import xml.etree.ElementTree as etree
import config
import datetime

if config.classification_mode == 1:
    target_name = config.target_name_dtw
else:
    target_name = config.target_name_rf

def readProd(file_sar):
    product = ProductIO.readProduct(file_sar)
    class sar:
        if "snap.core.datamodel.Product" in str(type(product)):
            prod = product
            width = product.getSceneRasterWidth()
            height = product.getSceneRasterHeight()
            name = product.getName()
            description = product.getDescription()
        else:
            print("errors.. see the log file!")
    return sar

home = os.getcwd()
while not os.path.exists(os.getcwd()+"/update_config/update_preprocess.txt"):
    time.sleep(0.5)
else:
    imp.reload(config)
isFile = home + config.new_preprocessresult
OperName = home + config.snappy_ops2
the_operators = open(OperName, "r")
Operators_Result = the_operators.readlines()

## Get SAR from selected Date
date1 = str(config.new_start_date)
date2 = str(config.new_end_date)
start = datetime.datetime.strptime(date1, "%Y%m%d")
end = datetime.datetime.strptime(date2, "%Y%m%d")
logger.info("Get Preprocessed SAR Data from selected Date " +start.strftime("%d%b%Y") +" to "+end.strftime("%d%b%Y"))
step = datetime.timedelta(days=12)
list_date = []
while start <= end:
    list_date.append(start.strftime("%Y%m%d"))
    start += step
file_list = os.listdir(isFile)
selected_file = []
for ss in list_date:
    if any(ss in s for s in file_list):
        rr = [s for s in file_list if ss in s and s.endswith(".dim")]
        selected_file.append(rr[0])
# print(selected_file)

# Subset and Stack start...
logger.info("Start Preprocessing GPF using snappy..")
logger.info("GPF Operator: "+" - ".join(Operators_Result).replace("\n", ""))

dim_dir = home + config.subset_stack_result
if not os.path.exists(dim_dir):
    os.makedirs(dim_dir)

if len(Operators_Result)==1 and ("CreateStack" in Operators_Result):
    pass
    op=Operators_Result[0].strip()
    stack_product=[]
    for file in selected_file:
        if file.endswith(".dim"):
            (sarfname, extension) = os.path.splitext(file)
            # read sar
            sar = readProd(isFile + file)
            stack_product.append(sar.prod)
            System = jpy.get_type("java.lang.System")
            System.gc()
    run_op = operatorSNAP(op).execute(stack_product)
    # write product
    trg_path = dim_dir + target_name+"_stack.dim" #Lampung_S1A_timeseries_2018Anual_Medium
    # check if file is exist or not
    if os.path.exists(trg_path):
        logger.info(target_name+"_stack.dim already processed")
        print (target_name+"_stack.dim already processed")
    else:
        # Download file and read
        try:
            logger.info("Write Product: " + str(trg_path))
            print("Write Product:" + str(trg_path))
            WriteProd().write_product(run_op.trg_data, trg_path)
        except:
            logger.info("Failed to Processed GPF using snappy python API!")
            print("Failed to Processed GPF using snappy python API!")
else:
    subset_product=[]
    for ops in Operators_Result:
        op = ops.strip()
        if op == "CreateStack":
            logger.info("create product of SNAP for operator: " + op)
            print("create product of SNAP for operator: "+ op)
            run_op = operatorSNAP(op).execute(subset_product)
        else:
            logger.info("create product of SNAP for operator: " + op)
            print("create product of SNAP for operator: " + op)
            for file in selected_file:
                #print(file)
                if file.endswith(".dim"):
                    (sarfname, extension) = os.path.splitext(file)
                    # read sar
                    sar = readProd(isFile + file)
                    run_op = operatorSNAP(op).execute(sar.prod)
                    subset_product.append(run_op.trg_data)
                    System = jpy.get_type("java.lang.System")
                    System.gc()
    #print(subset_product)
    # write product
    trg_path = dim_dir + target_name+"_subset_stack.dim" #Lampung_S1A_timeseries_2018Anual_Medium
    # check if file is exist or not
    if os.path.exists(trg_path):
        logger.info(target_name+"_subset_stack.dim already processed")
        print(target_name+"_subset_stack.dim already processed")
    else:
        # Download file and read
        try:
            logger.info("Write Product: " + str(trg_path))
            print("Write Product:" + str(trg_path))
            WriteProd().write_product(run_op.trg_data, trg_path)
        except:
            logger.info("Failed to Processed GPF using snappy python API!")
            print("Failed to Processed GPF using snappy python API!")

### Write update folder parproces to text file
f = open(os.getcwd() + "/update_config/update_subset_stack.txt", "w")
f.write(trg_path.replace(".dim", ""))
f.close()