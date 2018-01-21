import os
import sys
import glob
import pickle
import numpy as np # Scientific calculation package.
from datetime import datetime # Import datetime class from datetime package.
from datetime import timedelta
import pandas as pd
sys.path.append('/Users/jz0006/GoogleDrive/GS/GS_FluxRopeDetectionPackage')
import MyPythonPackage.fluxrope as FR


# Terminal output format.
pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 300)

#=============================================================================

#year_str = sys.argv[1]
#year = int(year_str)
rootDir = '/Users/jz0006/GoogleDrive/GS/'
inputDataDir = 'GS_DataPickleFormat/preprocessed/'
inputDir = 'GS_SearchResult/raw_result/'
outputDir = 'GS_SearchResult/raw_result/raw_record_list_dict/'


for year in xrange(1996, 2017):
    year_str = str(year)
    # Create an empty dictionary to store one year data.
    oneYear_dict = {'true':{}, 'timeRange':{'datetimeStart':datetime(year,1,1,0,0), 'datetimeEnd':datetime(year,12,31,23,59)}, 'false':{}}
    for duration_str in ('010~020','020~030','030~040','040~050','050~060','060~080','080~100','100~120','120~140','140~160','160~180'):
        oneFileName_temp = rootDir + inputDir + year_str + '/' + year_str + '_true_' + duration_str + 'min.p'
        print(oneFileName_temp)
        recordList_temp = pickle.load(open(oneFileName_temp, 'rb'))
        # Append oneDuration_dict to oneYear_dict.
        oneYear_dict['true'][duration_str] = recordList_temp # oneYear_dict['true'] is a dictionary.
    oneYear_dict_fileName = rootDir + outputDir + year_str + '_raw_record_list_dict.p'
    pickle.dump(oneYear_dict, open(oneYear_dict_fileName, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)


exit()







# Physics constants.
mu0 = 4.0 * np.pi * 1e-7 #(N/A^2) magnetic constant permeability of free space vacuum permeability
m_proton = 1.6726219e-27 # Proton mass. In kg.
factor_deg2rad = np.pi/180.0 # Convert degree to rad.
k_Boltzmann = 1.3806488e-23 # Boltzmann constant, in J/K.
# Parameters.
dt = 60.0 # For 1 minute resolution data, the time increment is 60 seconds.



# Read in all data for walen test.
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
    
