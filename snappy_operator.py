# class of operators
'''
----------------------------------------------------------------------------------------------------------------------------------------
Import required libraries from python
----------------------------------------------------------------------------------------------------------------------------------------
'''
import os
import snappy
import config
import logging

logger = logging.getLogger()

# The ProductIO class provides several utility methods concerning data I/O for remote sensing data products.
from snappy import ProductIO

# SNAP's Graph Processing Framework GPF used for developing and executing raster data operators and graphs of such operators.
from snappy import GPF

# Java - Python bridge
from snappy import jpy

# to read file text and convert it to dictionary
def read_param(filename):
    import json
    commands = {}
    fh = open(filename, 'r')
    lines = list(fh)
    for line in lines:
        li = line.strip()
        if not li.startswith("#"):
            if li:
                line = line.replace('\t', ' ')
                command, description = line.strip().split(' ', 1)
                commands[command] = description.strip()
    return (commands)


'''
----------------------------------------------------------------------------------------------------------------------------------------
Class of each Operators
----------------------------------------------------------------------------------------------------------------------------------------
'''

class operatorSNAP(object):

    def __init__(self, name_op):
        # HashMap
        #           Key-Value pairs.
        #           https://docs.oracle.com/javase/7/docs/api/java/util/HashMap.html
        self.HashMap = jpy.get_type('java.util.HashMap')

        # Get snappy Operators
        GPF.getDefaultInstance().getOperatorSpiRegistry().loadOperatorSpis()

        # Get operator Name
        self.name = name_op

        # Parameter File
        homedir = os.getcwd() + config.snappy_params
        filename = homedir + '/' + name_op + '.txt'

        # Read Parameters file from txt files and convert it to dictionary variables
        self.params = read_param(filename)

    def execute(self, src_data):
        para_dict = self.params
        self.func_name = self.name
        func_code = self.name

        parameters = self.HashMap()

        # Set parameters.
        logger.info('{}: Set parameters'.format(self.func_name))
        print('{}: Set parameters'.format(self.func_name))
        for para_name in para_dict:
            parameters.put(para_name, para_dict[para_name])

        # Create product.
        logger.info('{}: Create product...'.format(self.func_name))
        print('{}: Create product...'.format(self.func_name))
        self.trg_data = GPF.createProduct(func_code, parameters, src_data)
        return self


'''
----------------------------------------------------------------------------------------------------------------------------------------
Class Write Product 
----------------------------------------------------------------------------------------------------------------------------------------
'''


class WriteProd(object):

    def __init__(self):
        # HashMap
        #           Key-Value pairs.
        #           https://docs.oracle.com/javase/7/docs/api/java/util/HashMap.html
        self.HashMap = jpy.get_type('java.util.HashMap')

        # Get snappy Operators
        GPF.getDefaultInstance().getOperatorSpiRegistry().loadOperatorSpis()

    def write_product(self, src_data, trg_file_path, trg_file_format='BEAM-DIMAP'):
        # Parameter: - src_data <SAR Object>  : An object ('Product') of Sentinel SAR dataset
        #            - trg_file_path <string> : A string containing the path to the output Sentinel SAR dataset and it's name file.
        #            - trg_file_format <supported-format>: BEAM-DIMAP, GeoTIFF, NetCDF, ...
        # ------------------------------------------------------------------------------------------------------------------------------
        # ATTENTION, in case of:
        #                         RuntimeError: java.lang.OutOfMemoryError: Java heap space
        # ------------------------------------------------------------------------------------------------------------------------------
        # SOLUTION 1:
        # (http://forum.step.esa.int/t/snappy-error-productio-writeproduct/1102)
        #
        # 1. CHANGE <snappy>/jpyconfig.py:
        #                         jvm_maxmem = None    ---->      jvm_maxmem = '6G'
        #                                                         Increase RAM
        #
        # 2. CHANGE <snappy>/snappy.ini:
        #                         # java_max_mem: 4G   ---->      java_max_mem: 6G
        #                                                         Remove '#' and increase RAM
        # ------------------------------------------------------------------------------------------------------------------------------
        # SOLUTION 2:
        # If you swapped Latitude/Longitude in POLYGON(...) there is also a out-of-memory-error
        #
        ProductIO.writeProduct(src_data, trg_file_path, trg_file_format)