# V4.0 2017-05-03
# INPUT  : no_overlap record.
# OUTPUT : flux rope records with detailed information.
# NOTE   : Modularize code. Use MyPythonPackage.fluxrope module.

import os
import sys
import pickle
import numpy as np # Scientific calculation package.
from datetime import datetime # Import datetime class from datetime package.
from datetime import timedelta
import pandas as pd
sys.path.append('/Users/jz0006/GoogleDrive/GS/GS_FluxRopeDetectionPackage')
import MyPythonPackage.fluxrope as FR

pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 200)

###############################################################################

print('\nCheck the python version on Mac OS X, /usr/local/bin/python should be used:')
os.system('which python')
#homedir = os.environ['HOME']
rootDir = '/Users/jz0006/GoogleDrive/GS/'
inputDataDir = 'GS_DataPickleFormat/preprocessed/'
inputDir = 'GS_SearchResult/no_overlap/'
outputDir = rootDir + 'GS_SearchResult/detailed_info/'
if not os.path.exists(outputDir):
    os.makedirs(outputDir)

year_start = int(sys.argv[1])
year_end   = int(sys.argv[2])

for year in xrange(year_start, year_end+1, 1):
    year_str = str(year)
    
    # Read in all data.
    print('Reading data...')
    GS_oneYearData_DataFrame = pd.read_pickle(rootDir + inputDataDir + 'GS_' + year_str + '_AllData_DataFrame_preprocessed.p')
    # Check data property.
    print('Checking DataFrame keys... {}'.format(GS_oneYearData_DataFrame.keys()))
    print('Checking DataFrame shape... {}'.format(GS_oneYearData_DataFrame.shape))
    print('Data Time start: {}'.format(GS_oneYearData_DataFrame.index[0]))
    print('Data Time end: {}'.format(GS_oneYearData_DataFrame.index[-1]))
    # Check the NaNs in each variable.
    print('Checking the number of NaNs in DataFrame...')
    len_GS_oneYearData_DataFrame = len(GS_oneYearData_DataFrame)
    for key in GS_oneYearData_DataFrame.keys():
        num_notNaN = GS_oneYearData_DataFrame[key].isnull().values.sum()
        percent_notNaN = 100.0 - num_notNaN * 100.0 / len_GS_oneYearData_DataFrame
        print('The number of NaNs in {} is {}, integrity is {}%'.format(key, num_notNaN, round(percent_notNaN, 2)))
    # Get more flux rope information.
    search_result_no_overlap_DF_fileName = rootDir + inputDir + year_str + '_no_overlap.p'
    search_result_detail_info_DF = FR.get_more_flux_rope_info(GS_oneYearData_DataFrame, search_result_no_overlap_DF_fileName, output_dir=outputDir, output_filename=year_str+'_detailed_info')
    print(len(search_result_detail_info_DF))





        
