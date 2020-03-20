# -*- coding: utf-8 -*-

from sentinelsat import SentinelAPI, read_geojson, geojson_to_wkt
from datetime import date, datetime, timedelta
import  os, time
import config, logging

logger = logging.getLogger()

# Here for convert date
def format_date(in_date):
    """Format date or datetime input or a YYYYMMDD string input to
    YYYY-MM-DDThh:mm:ssZ string format. In case you pass an
    """
    if type(in_date) == datetime or type(in_date) == date:
        return in_date.strftime('%Y-%m-%dT%H:%M:%SZ')
    else:
        try:
            return datetime.strptime(in_date, '%Y%m%d').strftime('%Y-%m-%dT%H:%M:%SZ')
        except ValueError:
            return in_date
# Start
# Define initial date, end date, and parameter for download SAR
cwd = os.getcwd()
footprint = geojson_to_wkt(read_geojson(cwd+config.geojson))
type_sar = config.producttype # can cange to SLC
orbit = config.orbitdirection
### Set Date for DTW Classifier or RF Classifier
if config.classification_mode == 1:
    in_date = format_date(config.start_date)
    end_date = format_date(config.end_date)
    print(("Step 1: Download Sentinel SAR Product in " + config.name_of_area +
          ' area with select Dates between ' + in_date + ' and ' + end_date))
    logger.info("Download Sentinel SAR Product in " + config.name_of_area +
          ' area with select Dates between ' + in_date + ' and ' + end_date)
else:
    select_date = datetime.strptime(config.rf_date, '%Y%m%d')  # type: datetime
    step = timedelta(days=12)
    select_date2 = select_date - step
    print("Step 1: Download Sentinel SAR Product in " + config.name_of_area +
          ' area with select Dates between ' + format_date(select_date2) + ' and ' + format_date(select_date))
    logger.info("Download Sentinel SAR Product in " + config.name_of_area +
          ' area with select Dates between ' + format_date(select_date2) + ' and ' + format_date(select_date))
    end_date = format_date(select_date+ timedelta(days=1))
    in_date = format_date(select_date2)

url =  config.url
username = config.username # ask ITC for the username and password
password = config.password

#Get info product
api = SentinelAPI(username, password) # fill with SMARTSeeds user and password
products = api.query(footprint,
                     producttype =type_sar,
                     orbitdirection =orbit,
                     date='[{0} TO {1}]'.format(in_date, end_date)
                     )

# convert to Pandas DataFrame
df = api.to_dataframe(products)
df = df.drop_duplicates(subset=['orbitnumber'], keep='first')
df = df.sort_values(['beginposition'], ascending=[True])

if len(df) == 1:
    logger.error("Can not create stack from 1 product sentinel from selected dates")
    logger.error('Product should more than 1. Please select the correct date')
    raise Exception('the correct date must be: '+df['beginposition'][0].strftime('%Y%m%d')+' or '+\
                (df['beginposition'][0]+timedelta(days=12)).strftime('%Y%m%d'))
    #sys.exit(1)

### Write update date to text file
updt_dir = os.getcwd()+"/update_config/"
if not os.path.exists(updt_dir):
    os.makedirs(updt_dir)
f = open(os.getcwd()+"/update_config/update_date.txt", 'w' )
f.write( df['beginposition'][0].strftime('%Y%m%d') + '\n' )
f.write( df['beginposition'][len(df)-1].strftime('%Y%m%d') )
f.close()

dirpath = cwd + config.sentineldirpath
if not os.path.exists(dirpath):
    os.makedirs(dirpath)

# Start Download
for entry in range(0, len(df)):
    # The uuid element allows to create the path to the file
    uuid_element = df['uuid'][entry]
    id_sar = df['identifier'][entry]
    sentinel_link = df['link'][entry]

    # Destinationpath with filename where download to be stored
    destinationpath = dirpath + id_sar + '.zip'

    if os.path.exists(destinationpath):
        logger.info(id_sar + ' already downloaded')
        print(id_sar + ' already downloaded')
    else:
        # Download file and read
        try:
            api.download(df['uuid'][entry], directory_path=dirpath, checksum=True)
            logger.info("Successfully downloaded" + id_sar+ 'in to'+ destinationpath)
        except:
            logger.warning("error connection!.... Download Interrupted!")
            print "error connection!.... Download Interrupted!"
            time.sleep(1)

# delete all incomplete file
for item in os.listdir(dirpath):
    if item.endswith(".incomplete"):
        os.remove(os.path.join(dirpath, item))