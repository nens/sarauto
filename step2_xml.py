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
isFile = home + config.sentineldirpath
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

dim_dir = home + config.praprocess_result + nameOP + '/'
if not os.path.exists(dim_dir):
    os.makedirs(dim_dir)

logger.info('Start Praprocessing GPF using GPT and XML..')
logger.info('GPF Operator: '+' - '.join(GPF_ops))
for d_file in selected_file:
    (sarfileshortname, extension)  = os.path.splitext(d_file)
    read_data = isFile+d_file
    write_data = dim_dir+sarfileshortname+'.dim'
    for entry in elem:
        try:
            if (entry.attrib["id"]=="Read"):
                entry[2][0].text = read_data
            if (entry.attrib["id"]=="Write"):
                entry[2][0].text = write_data
        except:
            continue
    tree.write(xml_dir+'sar_praproses_'+str(indx)+'.xml')
    indx = indx + 1
logger.info('Make XML file for all Graphic Processing Framework')
### Write update folder parproces to text file
f = open(os.getcwd() + "/update_config/update_praprocess.txt", 'w')
f.write(nameOP)
f.close()