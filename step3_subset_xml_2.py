import os, glob
import xml.etree.ElementTree as etree
import config
import logging

logger = logging.getLogger()

isFile = os.getcwd()+config.xmlprocesspathsubset
listOfFiles=os.listdir(isFile)

# Function to check File Dim exist or not in directory
def get_xml_name(elem):
    for entry in elem:
        try:
            if (entry.attrib["id"]=="Write"):
                return(entry[2][0].text)
        except:
            pass

for d_file in listOfFiles:
    filepath = isFile + d_file
    tree = etree.parse(filepath)
    elem = tree.findall(".//node")
    dim_name = get_xml_name(elem)
    if not os.path.isfile(filepath):
        logger.error("File " + filepath + " not founds")
        print("File "+filepath+" not founds")
        sys.exit(0)
    else:
        if os.path.exists(dim_name):
            logger.info(os.path.basename(dim_name) + ' already processed')
            print (os.path.basename(dim_name) + ' already processed')
            os.remove(filepath)
        else:
            logger.info("processing GPT for " + os.path.basename(dim_name))
            print("processing GPT for " + os.path.basename(dim_name))
            try:
                os.system("gpt "+filepath)
                logger.info("GPT Subset for " + os.path.basename(dim_name) + ' is complete')
                print("GPT Subset for " + os.path.basename(dim_name) + ' is complete')
                os.remove(filepath)
            except:
                print('Failed to complete GPT Subset for ' + os.path.basename(dim_name))
                logger.error('Failed to complete Subset GPT for ' + os.path.basename(dim_name))
                sys.exit(0)