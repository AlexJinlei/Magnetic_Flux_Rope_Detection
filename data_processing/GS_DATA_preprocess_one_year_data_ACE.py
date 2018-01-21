#!/usr/local/bin/python

'''
Preprocess data.
'''
import os
import sys
import glob
import pickle
import numpy as np # Scientific calculation package.
from numpy import linalg as la
import math
import pandas as pd
from aenum import Enum # Enum data type
from ai import cdas # Import CDAweb API package.
import astropy
from spacepy import pycdf
from scipy.signal import savgol_filter # Savitzky-Golay filter
import scipy as sp
from datetime import datetime # Import datetime class from datetime package.
# class datetime.timedelta([days[, seconds[, microseconds[, milliseconds[, minutes[, hours[, weeks]]]]]]])
from datetime import timedelta
import matplotlib
import matplotlib.pyplot as plt # Plot package, make matlab style plots.
from matplotlib.ticker import AutoMinorLocator
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import FuncFormatter
from matplotlib.lines import Line2D
import matplotlib.dates as mdates
import matplotlib.cbook as cbook

################################################################################################################

pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 200)

print('\nCheck the python version on Mac OS X, /usr/local/bin/python should be used:')
os.system('which python')
homedir = os.environ['HOME']
rootDir = '/Users/jz0006/GoogleDrive/GS/'
inputDir = rootDir + 'data_cache/'
outputDir = rootDir + 'GS_DataPickleFormat/'

year_str = sys.argv[1]
year = int(year_str)
datetimeStartAll = datetime(year, 1,1,0,0,0)
datetimeEndAll   = datetime(year, 12,31,23,59,59)

B_filename = glob.glob(inputDir+'ac_h0s_mfi_'+year_str+'*.cdf')[0]
SW_filename = glob.glob(inputDir+'ac_h0s_swe_'+year_str+'*.cdf')[0]
EPM_filename = glob.glob(inputDir+'ac_h1s_epm_'+year_str+'*.cdf')[0]

# Time range for plot.
datetime_start = datetime(year,1,1,0,0,0)
datetime_end = datetime(year,12,31,23,59,59)

# Read CDF file, return an dictionary object.
# B_cdffile.keys() = ['Epoch', 'BGSE', 'cartesian', 'metavar0'].
# B_cdffile['BGSE'] is an Var object. It is a CDF variable.
# This object does not directly store the data from the CDF;
# rather, it provides access to the data in a format that much
# like a Python list or numpy :class:`~numpy.ndarray`.
# Note that, in CDAweb API, it will be converted to dictionary object automatically by API.
print('Reading cdf files:')
print('Reading '+ B_filename + '...')
B_cdffile = pycdf.CDF(B_filename) # Magnetic field.
print('Reading '+ SW_filename + '...')
SW_cdffile = pycdf.CDF(SW_filename) # Solar wind data.
print('Reading '+ EPM_filename + '...')
EPM_cdffile = pycdf.CDF(EPM_filename) # Energetic particle P1~P8 data.

# Extract data from cdf file.
print('Extracting data from cdf file...')
print('Extracting BGSE_Epoch...')
BGSE_Epoch = B_cdffile['Epoch'][...]
print('Extracting BGSEc...')
BGSE = B_cdffile['BGSEc'][...] # Magnetic file in GSE coordinate.
print('Extracting SW_Epoch...')
SW_Epoch = SW_cdffile['Epoch'][...]
print('Extracting V_GSE...')
VGSE = SW_cdffile['V_GSE'][...] # Solar wind speed in GSE coordinate.
print('Extracting Np...')
Np = SW_cdffile['Np'][...] # Proton number density.
print('Extracting Tpr...')
Tpr = SW_cdffile['Tpr'][...] # Proton thermal speed.
print('Extracting alpha_ratio...')
Alpha_ratio = SW_cdffile['alpha_ratio'][...] # Na/Np.
print('Extracting EPM_Epoch...')
EPM_Epoch = EPM_cdffile['Epoch'][...]
print('Extracting EPM P1...')
EPM_P1 = EPM_cdffile['P1'][...]
print('Extracting EPM P2...')
EPM_P2 = EPM_cdffile['P2'][...]
print('Extracting EPM P3...')
EPM_P3 = EPM_cdffile['P3'][...]
print('Extracting EPM P4...')
EPM_P4 = EPM_cdffile['P4'][...]
print('Extracting EPM P5...')
EPM_P5 = EPM_cdffile['P5'][...]
print('Extracting EPM P6...')
EPM_P6 = EPM_cdffile['P6'][...]
print('Extracting EPM P7...')
EPM_P7 = EPM_cdffile['P7'][...]
print('Extracting EPM P8...')
EPM_P8 = EPM_cdffile['P8'][...]

# Trim data. Some times cdas API will download wrong time range.
print('Trimming data to specified time range...')
# Trim BGSE data.
selected_index = [(BGSE_Epoch > datetimeStartAll) & (BGSE_Epoch < datetimeEndAll)]
BGSE_Epoch = BGSE_Epoch[selected_index]
BGSE = BGSE[selected_index]
# Trim V_GSE, Np, Tpr, and alpha_ratio data.
selected_index = [(SW_Epoch > datetimeStartAll) & (SW_Epoch < datetimeEndAll)]
SW_Epoch = SW_Epoch[selected_index]
VGSE = VGSE[selected_index]
Np = Np[selected_index]
Tpr = Tpr[selected_index]
Alpha_ratio = Alpha_ratio[selected_index]
# Trim P1 ~ P8 data.
selected_index = [(EPM_Epoch > datetimeStartAll) & (EPM_Epoch < datetimeEndAll)]
EPM_P1 = EPM_P1[selected_index]
EPM_P2 = EPM_P2[selected_index]
EPM_P3 = EPM_P3[selected_index]
EPM_P4 = EPM_P4[selected_index]
EPM_P5 = EPM_P5[selected_index]
EPM_P6 = EPM_P6[selected_index]
EPM_P7 = EPM_P7[selected_index]
EPM_P8 = EPM_P8[selected_index]

# Process missing value. missing value = -9.9999998e+30.
BGSE[abs(BGSE) > 80] = np.nan # B field.
Np[Np < -1e+10] = np.nan # Proton number density.
VGSE[abs(VGSE) > 1500] = np.nan # Solar wind speed.
Tpr[Tpr < -1e+10] = np.nan # Proton temperature, radial component of T tensor.
print(Alpha_ratio)
Alpha_ratio[Alpha_ratio < -1e+10] = np.nan # Na/Np.
print(Alpha_ratio)
EPM_P1[EPM_P1 < -1e+10] = np.nan
EPM_P2[EPM_P2 < -1e+10] = np.nan
EPM_P3[EPM_P3 < -1e+10] = np.nan
EPM_P4[EPM_P4 < -1e+10] = np.nan
EPM_P5[EPM_P5 < -1e+10] = np.nan
EPM_P6[EPM_P6 < -1e+10] = np.nan
EPM_P7[EPM_P7 < -1e+10] = np.nan
EPM_P8[EPM_P8 < -1e+10] = np.nan


print(min(EPM_P1), max(EPM_P1))
print(min(EPM_P2), max(EPM_P2))
print(min(EPM_P3), max(EPM_P3))
print(min(EPM_P4), max(EPM_P4))
print(min(EPM_P5), max(EPM_P5))
print(min(EPM_P6), max(EPM_P6))
print(min(EPM_P7), max(EPM_P7))
print(min(EPM_P8), max(EPM_P8))


# Put data into DataFrame.
print('Putting BGSE into DataFrame...')
BGSE_DataFrame = pd.DataFrame(BGSE, index = BGSE_Epoch, columns = ['Bx', 'By', 'Bz'])
print('Putting VGSE into DataFrame...')
VGSE_DataFrame = pd.DataFrame(VGSE, index = SW_Epoch, columns = ['Vx', 'Vy', 'Vz'])
print('Putting Np into DataFrame...')
Np_DataFrame = pd.DataFrame(Np, index = SW_Epoch, columns = ['Np'])
print('Putting Tp into DataFrame...')
Tp_DataFrame = pd.DataFrame(Tpr, index = SW_Epoch, columns = ['Tp'])
print('Putting Alpha_ratio into DataFrma...')
Alpha_ratio_DataFrame = pd.DataFrame(Alpha_ratio, index = SW_Epoch, columns = ['Alpha_ratio'])
print('Putting P1 ~ P8 into DataFrame...')
P1_8_DataFrame = pd.DataFrame({'P1': EPM_P1, 'P2': EPM_P2, 'P3': EPM_P3, 'P4': EPM_P4, 'P5': EPM_P5, 'P6': EPM_P6, 'P7': EPM_P7, 'P8': EPM_P8}, index = EPM_Epoch)

# Drop duplicated records. This is the flaw of the source data.
print('Dropping duplicated records...')
BGSE_DataFrame.drop_duplicates(keep='first', inplace=True) # Drop duplicated records, keep first one.
VGSE_DataFrame.drop_duplicates(keep='first', inplace=True) # Drop duplicated records, keep first one.
Np_DataFrame.drop_duplicates(keep='first', inplace=True) # Drop duplicated records, keep first one.
Tp_DataFrame.drop_duplicates(keep='first', inplace=True) # Drop duplicated records, keep first one.
Alpha_ratio_DataFrame.drop_duplicates(keep='first', inplace=True) # Drop duplicated records, keep first one.
P1_8_DataFrame.drop_duplicates(keep='first', inplace=True) # Drop duplicated records, keep first one.

# Sort data by time index. Time series data may be not in order, This is the flaw of the source data.
print('Sorting data...')
BGSE_DataFrame.sort_index(axis=0, ascending=True, inplace=True, kind='quicksort')
VGSE_DataFrame.sort_index(axis=0, ascending=True, inplace=True, kind='quicksort')
Np_DataFrame.sort_index(axis=0, ascending=True, inplace=True, kind='quicksort')
Tp_DataFrame.sort_index(axis=0, ascending=True, inplace=True, kind='quicksort')
Alpha_ratio_DataFrame.sort_index(axis=0, ascending=True, inplace=True, kind='quicksort')
P1_8_DataFrame.sort_index(axis=0, ascending=True, inplace=True, kind='quicksort')
P1_8_log_DataFrame = np.log10(P1_8_DataFrame)


# 1)
#=================================== Plot VGSE, BGSE, Np, Tp, and Alpha_ratio to check data quality ===============================
print('VGSE_DataFrame.shape = {}'.format(VGSE_DataFrame.shape))
print('BGSE_DataFrame.shape = {}'.format(BGSE_DataFrame.shape))

fig_line_width = 0.1
fig_ylabel_fontsize = 9
fig_xtick_fontsize = 8
fig_ytick_fontsize = 8
fig_legend_size = 5
fig,ax = plt.subplots(9,1, sharex=True,figsize=(18, 12))
Vx_plot = ax[0]
Vy_plot = ax[1]
Vz_plot = ax[2]
Bx_plot = ax[3]
By_plot = ax[4]
Bz_plot = ax[5]
Np_plot = ax[6]
Tp_plot = ax[7]
Alpha_ratio_plot = ax[8]

# Plotting Vx data.
Vx_plot.plot(VGSE_DataFrame[datetime_start:datetime_end].index, VGSE_DataFrame['Vx'][datetime_start:datetime_end],color = 'blue', linewidth=fig_line_width, label='Vx')
Vx_plot.set_ylabel('Vx', fontsize=fig_ylabel_fontsize)
Vx_plot.legend(loc='upper left',prop={'size':fig_legend_size})
Vx_plot.tick_params(axis='y', which='major', labelsize=fig_ytick_fontsize)
# Plotting Vy data.
Vy_plot.plot(VGSE_DataFrame[datetime_start:datetime_end].index, VGSE_DataFrame['Vy'][datetime_start:datetime_end],color = 'blue', linewidth=fig_line_width, label='Vy')
Vy_plot.set_ylabel('Vy', fontsize=fig_ylabel_fontsize)
Vy_plot.legend(loc='upper left',prop={'size':fig_legend_size})
Vy_plot.tick_params(axis='y', which='major', labelsize=fig_ytick_fontsize)
# Plotting Vz data.
Vz_plot.plot(VGSE_DataFrame[datetime_start:datetime_end].index, VGSE_DataFrame['Vz'][datetime_start:datetime_end],color = 'blue', linewidth=fig_line_width, label='Vz')
Vz_plot.set_ylabel('Vz', fontsize=fig_ylabel_fontsize)
Vz_plot.legend(loc='upper left',prop={'size':fig_legend_size})
Vz_plot.tick_params(axis='y', which='major', labelsize=fig_ytick_fontsize)
# Plotting Bx data.
Bx_plot.plot(BGSE_DataFrame[datetime_start:datetime_end].index, BGSE_DataFrame['Bx'][datetime_start:datetime_end],color = 'blue', linewidth=fig_line_width, label='Bx')
Bx_plot.set_ylabel('Bx', fontsize=fig_ylabel_fontsize)
Bx_plot.legend(loc='upper left',prop={'size':fig_legend_size})
Bx_plot.tick_params(axis='y', which='major', labelsize=fig_ytick_fontsize)
# Plotting By data.
By_plot.plot(BGSE_DataFrame[datetime_start:datetime_end].index, BGSE_DataFrame['By'][datetime_start:datetime_end],color = 'blue', linewidth=fig_line_width, label='By')
By_plot.set_ylabel('By', fontsize=fig_ylabel_fontsize)
By_plot.tick_params(axis='y', which='major', labelsize=fig_ytick_fontsize)
By_plot.legend(loc='upper left',prop={'size':fig_legend_size})
# Plotting Bz data.
Bz_plot.plot(BGSE_DataFrame[datetime_start:datetime_end].index, BGSE_DataFrame['Bz'][datetime_start:datetime_end],color = 'blue', linewidth=fig_line_width, label='Bz')
Bz_plot.set_ylabel('Bz', fontsize=fig_ylabel_fontsize)
Bz_plot.legend(loc='upper left',prop={'size':fig_legend_size})
Bz_plot.tick_params(axis='y', which='major', labelsize=fig_ytick_fontsize)
# Plotting Np data.
Np_plot.plot(Np_DataFrame[datetime_start:datetime_end].index, Np_DataFrame['Np'][datetime_start:datetime_end],color = 'blue', linewidth=fig_line_width, label='Np')
Np_plot.set_ylabel('Np', fontsize=fig_ylabel_fontsize)
Np_plot.legend(loc='upper left',prop={'size':fig_legend_size})
Np_plot.tick_params(axis='y', which='major', labelsize=fig_ytick_fontsize)
# Plotting Tp data.
Tp_plot.plot(Tp_DataFrame[datetime_start:datetime_end].index, Tp_DataFrame['Tp'][datetime_start:datetime_end],color = 'blue', linewidth=fig_line_width, label='Tp') # Filtered data.
Tp_plot.set_ylabel('Tp', fontsize=fig_ylabel_fontsize)
Tp_plot.legend(loc='upper left',prop={'size':fig_legend_size})
Tp_plot.tick_params(axis='y', which='major', labelsize=fig_ytick_fontsize)
# Plotting Alpha_ratio data.
Alpha_ratio_plot.plot(Alpha_ratio_DataFrame[datetime_start:datetime_end].index, Alpha_ratio_DataFrame['Alpha_ratio'][datetime_start:datetime_end],color = 'blue', linewidth=fig_line_width, label='Alpha_ratio') # Filtered data.
Alpha_ratio_plot.set_ylabel('Alpha_ratio', fontsize=fig_ylabel_fontsize)
Alpha_ratio_plot.legend(loc='upper left',prop={'size':fig_legend_size})
Alpha_ratio_plot.tick_params(axis='y', which='major', labelsize=fig_ytick_fontsize)

# Save plot.
fig.savefig('GS_VGSE_BGSE_Np_Tp_AlphaRatio('+str(datetime_start)+'~'+str(datetime_end)+').png',\
            format='png', dpi=500, bbox_inches='tight')

print('Done.')


# 2)
#========================================= Plot P1 ~ P8 to check data quality =========================================
print('P1_8_DataFrame.shape = {}'.format(P1_8_DataFrame.shape))

fig_line_width = 0.1
fig_ylabel_fontsize = 9
fig_xtick_fontsize = 8
fig_ytick_fontsize = 8
fig_legend_size = 5
fig,ax = plt.subplots(1,1, sharex=True,figsize=(18, 4))
P1_plot = ax
'''
P2_plot = ax[1]
P3_plot = ax[2]
P4_plot = ax[3]
P5_plot = ax[4]
P6_plot = ax[5]
P7_plot = ax[6]
P8_plot = ax[7]
'''

colors = ['red', 'orange', 'yellow', 'green', 'cyan', 'blue', 'magenta', 'black']
# Plotting P1 data.
P1_plot.plot(P1_8_log_DataFrame[datetime_start:datetime_end].index, P1_8_log_DataFrame['P1'][datetime_start:datetime_end],color = colors[0], linewidth=fig_line_width, label='P1')
P1_plot.set_ylim([-3,8])
P1_plot.set_ylabel('P1(log)', fontsize=fig_ylabel_fontsize)
P1_plot.legend(loc='upper left',prop={'size':fig_legend_size})
P1_plot.tick_params(axis='y', which='major', labelsize=fig_ytick_fontsize)
# Plotting P2 data.
P1_plot.plot(P1_8_log_DataFrame[datetime_start:datetime_end].index, P1_8_log_DataFrame['P2'][datetime_start:datetime_end],color = colors[1], linewidth=fig_line_width, label='P2')
P1_plot.set_ylim([-3,8])
P1_plot.set_ylabel('P2(log)', fontsize=fig_ylabel_fontsize)
P1_plot.legend(loc='upper left',prop={'size':fig_legend_size})
P1_plot.tick_params(axis='y', which='major', labelsize=fig_ytick_fontsize)
# Plotting P3 data.
P1_plot.plot(P1_8_log_DataFrame[datetime_start:datetime_end].index, P1_8_log_DataFrame['P3'][datetime_start:datetime_end],color = colors[2], linewidth=fig_line_width, label='P3')
P1_plot.set_ylim([-3,8])
P1_plot.set_ylabel('P3(log)', fontsize=fig_ylabel_fontsize)
P1_plot.legend(loc='upper left',prop={'size':fig_legend_size})
P1_plot.tick_params(axis='y', which='major', labelsize=fig_ytick_fontsize)
# Plotting P4 data.
P1_plot.plot(P1_8_log_DataFrame[datetime_start:datetime_end].index, P1_8_log_DataFrame['P4'][datetime_start:datetime_end],color = colors[3], linewidth=fig_line_width, label='P4')
P1_plot.set_ylim([-3,8])
P1_plot.set_ylabel('P4(log)', fontsize=fig_ylabel_fontsize)
P1_plot.legend(loc='upper left',prop={'size':fig_legend_size})
P1_plot.tick_params(axis='y', which='major', labelsize=fig_ytick_fontsize)
# Plotting P5 data.
P1_plot.plot(P1_8_log_DataFrame[datetime_start:datetime_end].index, P1_8_log_DataFrame['P5'][datetime_start:datetime_end],color = colors[4], linewidth=fig_line_width, label='P5')
P1_plot.set_ylim([-3,8])
P1_plot.set_ylabel('P5(log)', fontsize=fig_ylabel_fontsize)
P1_plot.legend(loc='upper left',prop={'size':fig_legend_size})
P1_plot.tick_params(axis='y', which='major', labelsize=fig_ytick_fontsize)
# Plotting P6 data.
P1_plot.plot(P1_8_log_DataFrame[datetime_start:datetime_end].index, P1_8_log_DataFrame['P6'][datetime_start:datetime_end],color = colors[5], linewidth=fig_line_width, label='P6')
P1_plot.set_ylim([-3,8])
P1_plot.set_ylabel('P6(log)', fontsize=fig_ylabel_fontsize)
P1_plot.legend(loc='upper left',prop={'size':fig_legend_size})
P1_plot.tick_params(axis='y', which='major', labelsize=fig_ytick_fontsize)
# Plotting P7 data.
P1_plot.plot(P1_8_log_DataFrame[datetime_start:datetime_end].index, P1_8_log_DataFrame['P7'][datetime_start:datetime_end],color = colors[6], linewidth=fig_line_width, label='P7')
P1_plot.set_ylim([-3,8])
P1_plot.set_ylabel('P7(log)', fontsize=fig_ylabel_fontsize)
P1_plot.legend(loc='upper left',prop={'size':fig_legend_size})
P1_plot.tick_params(axis='y', which='major', labelsize=fig_ytick_fontsize)
# Plotting P8 data.
P1_plot.plot(P1_8_log_DataFrame[datetime_start:datetime_end].index, P1_8_log_DataFrame['P8'][datetime_start:datetime_end],color = colors[7], linewidth=fig_line_width, label='P8')
P1_plot.set_ylim([-3,8])
P1_plot.set_ylabel('P8(log)', fontsize=fig_ylabel_fontsize)
P1_plot.legend(loc='upper left',prop={'size':fig_legend_size})
P1_plot.tick_params(axis='y', which='major', labelsize=fig_ytick_fontsize)

# Save plot.
fig.savefig('GS_P1~P8_log('+str(datetime_start)+'~'+str(datetime_end)+').png',\
            format='png', dpi=500, bbox_inches='tight')

# ============================== Resample data to 1min resolution ==============================.

n_interp_limit = 10

# Resample BGSE data into one minute resolution.
# Linear fit first, to make sure there is no NaN. NaN will propagate by resample.
# Interpolate according to timestamps. Cannot handle boundary. Do not interpolate NaN longer than 10.
BGSE_DataFrame.interpolate(method='time', inplace=True, limit=n_interp_limit)
# Process boundary.
#BGSE_DataFrame.fillna(method='ffill', inplace=True) # Fill missing data, forward copy.
#BGSE_DataFrame.fillna(method='bfill', inplace=True) # Fill missing data, backward copy.
print('Resampling BGSE data into 1 minute resolution...')
BGSE_DataFrame.drop_duplicates(keep='first', inplace=True) # Drop duplicated records, keep first one.
# Resample to 1 minute resolution. New added records will be filled with NaN.
BGSE_DataFrame = BGSE_DataFrame.resample('1T').mean()
# Interpolate according to timestamps. Cannot handle boundary.
BGSE_DataFrame.interpolate(method='time', inplace=True, limit=n_interp_limit)
# Process boundary.
#BGSE_DataFrame.fillna(method='ffill', inplace=True) # Fill missing data, forward copy.
#BGSE_DataFrame.fillna(method='bfill', inplace=True) # Fill missing data, backward copy.
print('BGSE_DataFrame shape : {}'.format(BGSE_DataFrame.shape))
print('The total records in BGSE : {}'.format(len(BGSE_DataFrame.index)))
print('Max(BGSE) = {}, Min(BGSE) = {}'.format(BGSE_DataFrame.max(), BGSE_DataFrame.min()))
print('Done.')

# Resample VGSE data into one minute resolution.
# Interpolate according to timestamps. Cannot handle boundary.
VGSE_DataFrame.interpolate(method='time', inplace=True, limit=n_interp_limit)
# Process boundary.
#VGSE_DataFrame.fillna(method='ffill', inplace=True) # Fill missing data, forward copy.
#VGSE_DataFrame.fillna(method='bfill', inplace=True) # Fill missing data, backward copy.
print('Resampling VGSE data into 1 minute resolution...')
VGSE_DataFrame.drop_duplicates(keep='first', inplace=True) # Drop duplicated records, keep first one.
# Resample to 1 minute resolution. New added records will be filled with NaN.
VGSE_DataFrame = VGSE_DataFrame.resample('1T').mean()
# Interpolate according to timestamps. Cannot handle boundary.
VGSE_DataFrame.interpolate(method='time', inplace=True, limit=n_interp_limit)
# Process boundary.
#VGSE_DataFrame.fillna(method='ffill', inplace=True) # Fill missing data, forward copy.
#VGSE_DataFrame.fillna(method='bfill', inplace=True) # Fill missing data, backward copy.
print('VGSE_DataFrame shape : {}'.format(VGSE_DataFrame.shape))
print('The total records in VGSE : {}'.format(len(VGSE_DataFrame.index)))
print('Max(VGSE) = {}, Min(VGSE) = {}'.format(VGSE_DataFrame.max(), VGSE_DataFrame.min()))
print('Done.')

# Resample Np data into one minute resolution.
# Linear fit first, to make sure there is no NaN. NaN will propagate by resample.
# Interpolate according to timestamps. Cannot handle boundary.
Np_DataFrame.interpolate(method='time', inplace=True, limit=n_interp_limit)
# Process boundary.
#Np_DataFrame.fillna(method='ffill', inplace=True) # Fill missing data, forward copy.
#Np_DataFrame.fillna(method='bfill', inplace=True) # Fill missing data, backward copy.
print('Resampling Np data into 1 minute resolution...')
Np_DataFrame.drop_duplicates(keep='first', inplace=True) # Drop duplicated records, keep first one.
# Resample to 1 minute resolution. New added records will be filled with NaN.
# Resample to 30 second first, or data points will shift too much, as large as 1 min.
Np_DataFrame = Np_DataFrame.resample('1T').mean()
# Interpolate according to timestamps. Cannot handle boundary.
Np_DataFrame.interpolate(method='time', inplace=True, limit=n_interp_limit)
# Process boundary.
#Np_DataFrame.fillna(method='ffill', inplace=True) # Fill missing data, forward copy.
#Np_DataFrame.fillna(method='bfill', inplace=True) # Fill missing data, backward copy.
# Now, downsample to 1 minute.
#Np_DataFrame = Np_DataFrame.resample('1T').mean()
print('Np_DataFrame shape : {}'.format(Np_DataFrame.shape))
print('The total records in Np : {}'.format(len(Np_DataFrame.index)))
print('Max(Np) = {}, Min(Np) = {}'.format(Np_DataFrame.max(), Np_DataFrame.min()))
print('Done.')

# Resample Tp data into one minute resolution.
# Linear fit first, to make sure there is no NaN. NaN will propagate by resample.
# Interpolate according to timestamps. Cannot handle boundary.
Tp_DataFrame.interpolate(method='time', inplace=True)
# Process boundary.
#Tp_DataFrame.fillna(method='ffill', inplace=True) # Fill missing data, forward copy.
#Tp_DataFrame.fillna(method='bfill', inplace=True) # Fill missing data, backward copy.
print('Resampling Tp data into 1 minute resolution...')
Tp_DataFrame.drop_duplicates(keep='first', inplace=True) # Drop duplicated records, keep first one.
# Resample to 1 minute resolution. New added records will be filled with NaN.
# Resample to 30 second first, or data points will shift too much, as large as 1 min.
Tp_DataFrame = Tp_DataFrame.resample('1T').mean()
# Interpolate according to timestamps. Cannot handle boundary.
Tp_DataFrame.interpolate(method='time', inplace=True, limit=n_interp_limit)
# Process boundary.
#Tp_DataFrame.fillna(method='ffill', inplace=True) # Fill missing data, forward copy.
#Tp_DataFrame.fillna(method='bfill', inplace=True) # Fill missing data, backward copy.
# Now, downsample to 1 minute.
# Tp_DataFrame = Tp_DataFrame.resample(rule='60S', base=30).mean()
print('Tp_DataFrame shape : {}'.format(Tp_DataFrame.shape))
print('The total records in Tp : {}'.format(len(Tp_DataFrame.index)))
print('Max(Tp) = {}, Min(Tp) = {}'.format(Tp_DataFrame.max(), Tp_DataFrame.min()))
print('Done.')

# Resample Alpha_ratio data into one minute resolution.
# Interpolate according to timestamps. Cannot handle boundary.
Alpha_ratio_DataFrame.interpolate(method='time', inplace=True, limit=n_interp_limit)
# Process boundary.
#Alpha_ratio_DataFrame.fillna(method='ffill', inplace=True) # Fill missing data, forward copy.
#Alpha_ratio_DataFrame.fillna(method='bfill', inplace=True) # Fill missing data, backward copy.
print('Resampling Alpha_ratio data into 1 minute resolution...')
Alpha_ratio_DataFrame.drop_duplicates(keep='first', inplace=True) # Drop duplicated records, keep first one.
# Resample to 1 minute resolution. New added records will be filled with NaN.
Alpha_ratio_DataFrame = Alpha_ratio_DataFrame.resample('1T').mean()
# Interpolate according to timestamps. Cannot handle boundary.
Alpha_ratio_DataFrame.interpolate(method='time', inplace=True, limit=n_interp_limit)
# Process boundary.
#Alpha_ratio_DataFrame.fillna(method='ffill', inplace=True) # Fill missing data, forward copy.
#Alpha_ratio_DataFrame.fillna(method='bfill', inplace=True) # Fill missing data, backward copy.
print('Alpha_ratio_DataFrame shape : {}'.format(Alpha_ratio_DataFrame.shape))
print('The total records in Alpha_ratio : {}'.format(len(Alpha_ratio_DataFrame.index)))
print('Max(Alpha_ratio) = {}, Min(Alpha_ratio) = {}'.format(Alpha_ratio_DataFrame.max(), Alpha_ratio_DataFrame.min()))
print('Done.')

# Resample P1_8 data into one minute resolution.
# Linear fit first, to make sure there is no NaN. NaN will propagate by resample.
# Interpolate according to timestamps. Cannot handle boundary.
P1_8_DataFrame.interpolate(method='time', inplace=True)
# Process boundary.
#Tp_DataFrame.fillna(method='ffill', inplace=True) # Fill missing data, forward copy.
#Tp_DataFrame.fillna(method='bfill', inplace=True) # Fill missing data, backward copy.
print('Resampling P1_8 data into 1 minute resolution...')
P1_8_DataFrame.drop_duplicates(keep='first', inplace=True) # Drop duplicated records, keep first one.
# Resample to 1 minute resolution. New added records will be filled with NaN.
# Resample to 30 second first, or data points will shift too much, as large as 1 min.
P1_8_DataFrame = P1_8_DataFrame.resample('1T').mean()
# Interpolate according to timestamps. Cannot handle boundary.
P1_8_DataFrame.interpolate(method='time', inplace=True, limit=n_interp_limit)
# Process boundary.
#P1_8_DataFrame.fillna(method='ffill', inplace=True) # Fill missing data, forward copy.
#P1_8_DataFrame.fillna(method='bfill', inplace=True) # Fill missing data, backward copy.
# Now, downsample to 1 minute.
# Tp_DataFrame = Tp_DataFrame.resample(rule='60S', base=30).mean()
print('P1_8_DataFrame shape : {}'.format(P1_8_DataFrame.shape))
print('The total records in Tp : {}'.format(len(P1_8_DataFrame.index)))
print('Max(P1_8) = {}, Min(P1_8) = {}'.format(P1_8_DataFrame.max(), P1_8_DataFrame.min()))
print('Done.')


# Merge all DataFrames into one according to time index.
# Calculate time range in minutes.
timeRangeInMinutes = int((datetimeEndAll - datetimeStartAll).total_seconds())//60
# Generate timestamp index.
index_datetime = np.asarray([datetimeStartAll + timedelta(minutes=x) for x in range(0, timeRangeInMinutes)])
# Generate empty DataFrame according using index_datetime as index.
GS_AllData_DataFrame = pd.DataFrame(index=index_datetime)
# Merge all DataFrames.
GS_AllData_DataFrame = pd.concat([GS_AllData_DataFrame, BGSE_DataFrame, VGSE_DataFrame, Np_DataFrame, Tp_DataFrame, Alpha_ratio_DataFrame, P1_8_DataFrame], axis=1)
# Save merged DataFrame into pickle file.
GS_AllData_DataFrame.to_pickle(outputDir + 'GS_'+str(year)+'_AllData_DataFrame_preprocessed_ACE.p')
print('Checking the number of NaNs in GS_AllData_DataFrame...')
len_GS_AllData_DataFrame = len(GS_AllData_DataFrame)
for key in GS_AllData_DataFrame.keys():
    num_notNaN = GS_AllData_DataFrame[key].isnull().values.sum()
    percent_notNaN = 100.0 - num_notNaN * 100.0 / len_GS_AllData_DataFrame
    print('The number of NaNs in {} is {}, integrity is {}%'.format(key, num_notNaN, round(percent_notNaN, 2)))
#print(GS_AllData_DataFrame)






