import os, time
import glob
import xml.etree.ElementTree as etree
from snappy import ProductIO
import datetime
import config
import logging
import imp

logger = logging.getLogger()

home = os.getcwd()
xml_file = glob.glob(home+config.xmlpath+"*.XML")
tree = etree.parse(xml_file[0])
elem = tree.findall(".//node")
sentinel_path = home + config.sentineldirpath
indx = 1

## Get SAR from selected Date
while not os.path.isfile(home+"/update_config/update_date.txt"):
    time.sleep(0.5)
else:
    imp.reload(config)

date1 = str(config.new_start_date)
date2 = str(config.new_end_date)
start = datetime.datetime.strptime(date1, '%Y%m%d')
end = datetime.datetime.strptime(date2, '%Y%m%d')
logger.info('Get Sentinel SAR Data from selected Date from ' +start.strftime('%d%b%Y') +' to '+end.strftime('%d%b%Y'))
step = datetime.timedelta(days=12)
sentinel_dates = []
sentinel_dates.append(start.strftime('%Y%m%d'))
while start <= end:
    start += step
    sentinel_dates.append(start.strftime('%Y%m%d'))

file_list = os.listdir(sentinel_path)
selected_files = []
for sentinel_date in sentinel_dates:
    if any(sentinel_date in file for file in file_list):
        selected_file = [file for file in file_list if sentinel_date in file]
        selected_files.append(selected_file[0])
selected_files = os.listdir(sentinel_path)
logger.info("selected_files")
logger.info(selected_files)

name=[]
GPF_ops = []
for prnt in elem:
    if (prnt.attrib["id"]!="Read"):
        if (prnt.attrib["id"]=="Write"):
            break
        else:
            GPF_ops.append(prnt.attrib['id'])
            name.append(prnt.attrib['id'][:3])
nameOP = '_'.join(name)

##checkdir
xml_dir = home+config.xmlprocesspath
if not os.path.exists(xml_dir):
    os.makedirs(xml_dir)

dim_dir = home + config.preprocess_result + nameOP + '/'
if not os.path.exists(dim_dir):
    os.makedirs(dim_dir)

logger.info('Start Preprocessing GPF using GPT and XML..')
logger.info('GPF Operator: '+' - '.join(GPF_ops))

for selected_file in selected_files:
    (sarfileshortname, extension)  = os.path.splitext(selected_file)
    read_data = sentinel_path + selected_file
    write_data = dim_dir + sarfileshortname+'.dim'
    for entry in elem:
        try:
            if (entry.attrib["id"]=="Read"):
                entry[2][0].text = read_data
            if (entry.attrib["id"]=="Write"):
                entry[2][0].text = write_data
        except:
            continue
    tree.write(xml_dir+'sar_preprocess_'+str(indx)+'.xml')
    indx = indx + 1
logger.info('Make XML file for all Graphic Processing Framework')
### Write update folder parproces to text file
f = open(os.getcwd() + "/update_config/update_preprocess.txt", 'w')
f.write(nameOP)
f.close()