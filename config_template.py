import os

# ---- Choose classification mode (1 for Vegetable classification, and 2 for Crop growth stage classification) ---- #
#####################################################################################################################
classification_mode = 1
#####################################################################################################################


# ----------------- To Acess and Download Sentinel Data the Copernicus Open Access Hub --------------- #
########################################################################################################
username = "*******"
password =  "*******"
url = "https://scihub.copernicus.eu/apihub/"
# url = "https://scihub.copernicus.eu/dhus/"
geojson = "/data_input/geojson/smart_tiles/tile_1.geojson"
project_area = "/data_input/geojson/project_area.geojson"
smart_tiles = "/data_input/geojson/smart_tiles/"
platformname = "Sentinel-1"
producttype = "GRD"
orbitdirection = "Descending"
sentineldirpath = "//sentineldata_all//" # Directory to save downloaded sentinel File
name_of_area = "tile_1"
########################################################################################################


# ---------------- Directory of Preprocessing, Subset and Stack file using GPT- *.xml ----------------- #
#########################################################################################################
xmlpath="/data_input/xml/preprocess_xml/"
xmlpathsubset = "/data_input/xml/subset_xml/"
xmlpathstack = "/data_input/xml/stack_xml/"
xmlprocesspath = "/data_output/Temp_file/XMLprocess/"
xmlprocesspathsubset = "/data_output/Temp_file/XMLprocess/"
xmlprocesspathstack = "/data_output/Temp_file/XMLprocess/"
xmlpreprocessresultsubset="/data_output/Temp_file/subset_preprocess_result/"
#########################################################################################################


# ----------------- Directory of Preprocessing, Subset and Stack file using snappy -------------------- #
#########################################################################################################
snappy_ops = "/data_input/snappy/operators.txt"
snappy_ops2 = "/data_input/snappy/operators2.txt"
snappy_params = "/data_input/snappy/parameters/"
########################################################################################################


# --------- Directory Output of Preprocesses, subset, and stack (with GPT- *.xml or Snappy) ----------- #
#########################################################################################################
# Output for Phase Preprocess
preprocess_result = "/data_output/Preprocess_result/"

# Directory output for Phase Subset-Stack (Snappy or using XML) each for DTW and RF classification mode
if classification_mode == 1: # 1 for DTW Class 2 for Random Forest Class
    subset_stack_result = "/data_output/Subset_Stack_result/time_series_DTW/"
else:
    subset_stack_result = "/data_output/Subset_Stack_result/random_forest/"
xmlpreprocessresultstack = subset_stack_result
########################################################################################################


# ------------ Target name output for Phase Subset-Stack - DTW and RF classification mode ------------ #
########################################################################################################
target_name_dtw = name_of_area + "_S1A_DTW_Vegetable_classification"
target_name_rf = name_of_area + "_S1A_RF_Crop_growth_stage_classification"
########################################################################################################


# --------------------- Parameter for Vegetable classification with DTW Time Series ------------------- #
#########################################################################################################
log_file_dtw = "/logfile/Vegetable classification/"
start_date = "20190401"
end_date="20191031"
select_date = ["01Apr2019","31Oct2019"]
days_of_intrvl = 12
time_formatDTW = "%d%b%Y" #dayMonthYear, for example: 12Dec2018
splt_Str = "_"
block_size = [4058, 60]
doFilter = 1
K_val = 1
mat_files = "/data_input/Vegetable classification/mat_files/"
mask_files = "/data_input/Vegetable classification/landcover_mask/"
dtw_mat_save_dir = "/data_output/Vegetable classification/result_mat/"
dtw_tif_save_dir = "/data_output/Vegetable classification/result_geotiff/"
dtw_save_name = "crop_DTWMap"
#######
# Filename format


########################################################################################################

# ---------------- Parameter for Crop growth stage classification with Random Forest ----------------- #
########################################################################################################
log_file_rf = "/logfile/Crop growth stage classification/"
rf_date = "20181028"
chunk_size = 5000000 # use large size if you dont use subset
# Random Forest Parameter
n_estimators = 1000
random_state = 42
max_depth = 15
train_path ="/data_input/Crop growth stage classification/mat_files/train_mat.mat"
rf_mat_save_dir = "/data_output/Crop growth stage classification/result_mat/"
rf_tif_save_dir = "/data_output/Crop growth stage classification/result_geotiff/"
rf_save_name = "growth_RFMap"
########################################################################################################


# --------------------------------------- For Send logfile via email ---------------------------------- #
email_user = "me@some_email.com"
pswd = "*******"
send_to = "me@some_email.com"
#########################################################################################################


# ---------------------------------------- Updating Data ---------------------------------------------- #
#########################################################################################################
try:
    file = open(os.getcwd()+"/update_config/update_date.txt", "r")
    get_date = file.readlines()
    new_start_date = get_date[0].strip()
    new_end_date = get_date[1].strip()
    file.close()
except:pass

# Update folder preprocess
try:
    file = open(os.getcwd()+"/update_config/update_preprocess.txt", "r")
    get_fldr = file.readlines()
    new_folder = get_fldr[0].strip()
    new_preprocessresult = preprocess_result + new_folder +"/"
    file.close()
except:
    pass

# Update folder subset and stack
try:
    file = open(os.getcwd()+"/update_config/update_subset_stack.txt", "r")
    get_fldr = file.readlines()
    new_folder = get_fldr[0].strip()
    new_sub_stack_result = new_folder
    file.close()
except:
    pass
###########################################################################################################