# -*- coding: utf-8 -*-
"""
SMARTSeeds - Automation of the image processing workflow as part of the G4AW project.

Copyright (c) 2018-2019 Faculty of Geo-information Science and Earth Observation (ITC), University of Twente,
                        and Faculty of Mathematics and Natural Science, IPB University.
All rights reserved.

This program produce under of this project will become part of the overall output of the G4AW project SMARTSeeds.

Author
---------
@Agmalaro
@Imas.S.Sitanggang
@Hendrik
@mengmeng_li
@Wietske

"""
import glob
import time
import os
import imp
import logging
import config
import del_fil_dir
import traceback
# from send_mail import send_email

home = os.getcwd()
xml_exist_file = glob.glob(os.getcwd()+config.xmlpath+"*.XML")
xml_subset_exist_file = glob.glob(os.getcwd()+config.xmlpathsubset+"*.XML")
xml_stack__exist_file = glob.glob(os.getcwd()+config.xmlpathstack+"*.XML")

if config.classification_mode == 1:
    log_dir = home + config.log_file_dtw
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_name = log_dir+time.strftime("%Y%m%d(Date)_%H%p(Time)")+'_logfile'
else:
    log_dir = home + config.log_file_rf
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_name = log_dir+time.strftime("%Y%m%d(Date)_%H%p(Time)")+'_logfile'

logging.basicConfig(filename=log_name+'.log',
                    level=logging.INFO,
                    format="%(levelname)s %(asctime)s - %(message)s",
                    filemode='w')
logger = logging.getLogger()
flag = 0
try:
    # Step 1 : Downloading the file
    try:
        logger.info('Start Phase 1 : Downloading SAR Sentinel(S1) file')
        import step1
        logger.info('Phase 1 Completed\n\n')
    except Exception:
        flag = 1
        raise

    # Step 2 : Operation of Graphic Processing Framework using GPT or Snappy
    try:
        if xml_exist_file:
            logger.info('Start Phase 2 : Operation of Graphic Processing Framework using GPT')
            import step2_xml
            import step2_xml_2
            logger.info('Phase 2 Completed\n\n')
        else:
            logger.info('Start Phase 2  : Operation of Graphic Processing Framework using snappy')
            import step2_snappy  # as a note, this will not work if Product more than 10
            logger.info('Phase 2 Completed\n\n')
    except Exception:
        flag = 2
        raise


    # Step 3 : Subset and stack operation using GPT or Snappy
    try:
        if xml_stack__exist_file:
            if xml_subset_exist_file:
                logger.info('Start Phase 3  : Subset and stack operation using GPT')
                import step3_subset_xml
                import step3_subset_xml_2
                import step3_stack_xml
                import step3_stack_xml_2
                logger.info('Phase 3 Completed\n\n')
            else:
                logger.info('Start Phase 3  : Subset and stack operation using GPT')
                import step3_stack_xml
                import step3_stack_xml_2
                logger.info('Phase 3 Completed\n\n')
        else:
            logger.info('Start Phase 3  : Subset and stack operation using snappy')
            import step3_subset_stack_snappy
            logger.info('Phase 3 Completed\n\n')
    except Exception:
        flag = 3
        raise

    # Step 4: Run Classifier Mode Time Series (DTW) or Random classifier
    try:
        if config.classification_mode == 1:
            logger.info('Start Phase 4 : Run Classifier Mode: Time Series (DTW)')
            import step4_Vegetble_class as step4
            step4.main()
            logger.info('Phase 4 Completed\n\n')
        else:
            logger.info('Start Phase 4 : Run Classifier Mode: Random Forest')
            import step4_RF_class as step4
            step4.main()
            logger.info('Phase 4 Completed\n\n')
    except Exception:
        flag = 4
        raise
    try:
        imp.reload(del_fil_dir)
    except Exception:
        pass

    if flag==0:
        print('All Phases is Done..')
        logger.info('All Phases is Done..\n\n')
        subject = 'SAR Automation result_' + time.strftime("%Y-%m-%d_%H:%M:%S")
        body = 'All Phases is Done'+'\nSARAuto has been run successfully'
        # send_email(log_name, subject, body)

except Exception:
    print("There was an error while running Phase " + str(flag))
    tb = traceback.format_exc()
    logger.error("There was an error while running Phase " + str(flag))
    logger.error(tb)
    subject = 'SAR Automation result_' + time.strftime("%Y-%m-%d_%H:%M:%S")
    body = "There was an error while running Phase " + str(flag)
    # send_email(log_name, subject, body)