# V4.0 2017-05-03
# INPUT  : raw searching result, combined into one pickle file.
# OUTPUT : flux rope list without overlapped records.
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
'''
import matplotlib
import matplotlib.pyplot as plt # Plot package, make matlab style plots.
from matplotlib.ticker import AutoMinorLocator
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import FuncFormatter
from matplotlib.lines import Line2D
import matplotlib.dates as mdates
import matplotlib.cbook as cbook
'''

import warnings
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")

# Terminal output format.
pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 300)

# Physics constants.
mu0 = 4.0 * np.pi * 1e-7 #(N/A^2) magnetic constant permeability of free space vacuum permeability
m_proton = 1.6726219e-27 # Proton mass. In kg.
factor_deg2rad = np.pi/180.0 # Convert degree to rad.
k_Boltzmann = 1.3806488e-23 # Boltzmann constant, in J/K.
# Parameters.
dt = 60.0 # For 1 minute resolution data, the time increment is 60 seconds.

#=============================================================================

rootDir = '/Users/jz0006/GoogleDrive/GS/'
inputDataDir = 'GS_DataPickleFormat/preprocessed/'
inputDir = 'GS_SearchResult/raw_result/raw_record_list_dict/'
outputDir = rootDir + 'GS_SearchResult/no_overlap/'
if not os.path.exists(outputDir):
    os.makedirs(outputDir)

year_start = int(sys.argv[1])
year_end   = int(sys.argv[2])

for year in xrange(year_start, year_end+1):
    year_str = str(year)
    
    print('year = {}'.format(year))
    
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
    
    # Clean up overlapped records.
    oneYear_dict_fileName = rootDir + inputDir + year_str + '_raw_record_list_dict.p'
    shockList_DF = pd.read_pickle(open('/Users/jz0006/GoogleDrive/GS/GS_FluxRopeDetectionPackage/shockList/IPShock_ACE_or_WIND_1996_2016_DF.p', 'rb'))
    
    search_result_no_overlap_DF = FR.clean_up_raw_result(GS_oneYearData_DataFrame, oneYear_dict_fileName, walenTest_k_threshold=0.3, min_residue_diff=0.12, min_residue_fit=0.14, output_dir=outputDir, output_filename=year_str+'_no_overlap', isPrintIntermediateDF=False, isVerbose=True, isRemoveShock=True, shockList_DF=shockList_DF, spacecraftID='WIND')
    #print(len(search_result_no_overlap_DF))













    
