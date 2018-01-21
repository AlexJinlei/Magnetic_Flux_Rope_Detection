#!/usr/local/bin/python

'''
Preprocess data.

Note that, Tp is in therma speed, not converted to Kelvin.
'''
import os
import pickle
import numpy as np # Scientific calculation package.
from numpy import linalg as la
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
####################### User defined module #######################

def MaxMinVariance(B_input):
    # Vector space transformation.
    covM = np.cov(B_input.T) # Calculate the covariance matrix of Magnetic Field.
    # Calculate the eigenvalues and eigenvectors of covariance matrix.
    w, v = la.eig(covM) # w are eigenvalues, and v are eigenvectors.
    # Sort the eigenvalues and arrange eigenvectors by sorted eigenvalues.
    w_i = np.argsort(w) # w_i is sorted index of w
    lambda3 = w[w_i[0]] # lambda3, minimum variance
    lambda2 = w[w_i[1]] # lambda2, intermediate variance.
    lambda1 = w[w_i[2]] # lambda1, maximum variance.
    #print('lambda1 = ',lambda1)
    #print('lambda2 = ',lambda2)
    #print('lambda3 = ',lambda3)
    return lambda1, lambda2, lambda3

################################################################################################################
print('\nCheck the python version on Mac OS X, /usr/local/bin/python should be used:')
os.system('which python')
homedir = os.environ['HOME']
rootDir = '/Users/jz0006/GoogleDrive/GS/GS_DataPickleFormat/'

year = 2016
datetimeStartAll = datetime(year, 1,1,0,0,0)
datetimeEndAll   = datetime(year, 12,31,23,59,59)

B_filename = 'wi_h0s_mfi_20160101000030_20161231235930.cdf'
#Electron_filename = 'wi_h5s_swe_20150101235941_20151231000010.cdf'
SW_filename = 'wi_k0s_swe_20160101000045_20161231235835.cdf'

ElectronDataExist = False

# Time range for plot.
datetime_start = datetime(year,2,1,0,0,0)
datetime_end = datetime(year,2,1,3,59,59)

# Read CDF file, return an dictionary object.
# B_cdffile.keys() = ['Epoch', 'BGSE', 'cartesian', 'metavar0'].
# B_cdffile['BGSE'] is an Var object. It is a CDF variable.
# This object does not directly store the data from the CDF;
# rather, it provides access to the data in a format that much
# like a Python list or numpy :class:`~numpy.ndarray`.
# Note that, in CDAweb API, it will be converted to dictionary object automatically by API.
print('Reading cdf files:')
print('Reading '+ B_filename + '...')
B_cdffile = pycdf.CDF('~/GoogleDrive/GS/data_cache/'+str(year)+'/' + B_filename) # Magnetic field.
if ElectronDataExist:
    print('Reading '+ Electron_filename + '...')
    Electron_cdffile = pycdf.CDF('~/GoogleDrive/GS/data_cache/'+str(year)+'/' + Electron_filename) # Electron data.
print('Reading '+ SW_filename + '...')
SW_cdffile = pycdf.CDF('~/GoogleDrive/GS/data_cache/'+str(year)+'/' + SW_filename) # Solar wind data.

# Extract data from cdf file.
print('Extracting data from cdf file...')
print('Extracting BGSE_Epoch...')
BGSE_Epoch = B_cdffile['Epoch'][...]
print('Extracting BGSE...')
BGSE = B_cdffile['BGSE'][...] # Magnetic file in GSE coordinate.
if ElectronDataExist:
    print('Extracting Te_Epoch...')
    Te_Epoch = Electron_cdffile['Epoch'][...]
    print('Extracting Te...') # The key is different in different time range.
    if (datetimeStartAll>=datetime(1994,12,29,0,0,2) and datetimeStartAll<=datetime(2001,5,31,23,59,57)):
        Te = Electron_cdffile['Te'][...] # Electron temperature.
    elif (datetimeStartAll>=datetime(2002,8,16,0,0,5) and datetimeStartAll<=datetime(2015,5,1,0,0,12)):
        Te = Electron_cdffile['T_elec'][...] # Electron temperature.
print('Extracting SW_Epoch...')
SW_Epoch = SW_cdffile['Epoch'][...]
print('Extracting VGSE...')
VGSE = SW_cdffile['V_GSE'][...] # Solar wind speed in GSE coordinate.
print('Extracting Np...')
Np = SW_cdffile['Np'][...] # Proton number density.
print('Extracting Tp...')
Tp = SW_cdffile['THERMAL_SPD'][...] # Proton thermal speed.

# Trim data. Some times cdas API will download wrong time range.
print('Trimming data to specified time range...')
# Trim BGSE data.
selected_index = [(BGSE_Epoch > datetimeStartAll) & (BGSE_Epoch < datetimeEndAll)]
BGSE_Epoch = BGSE_Epoch[selected_index]
BGSE = BGSE[selected_index]
# Trim VGSE, Np, and Tp data. Share same SW_Epoch.
selected_index = [(SW_Epoch > datetimeStartAll) & (SW_Epoch < datetimeEndAll)]
SW_Epoch = SW_Epoch[selected_index]
VGSE = VGSE[selected_index]
Np = Np[selected_index]
Tp = Tp[selected_index]
# Trim Te data, if exists.
if 'Te' in locals(): # If Te is defined.
    print('Te exists.')
    selected_index = [(Te_Epoch > datetimeStartAll) & (Te_Epoch < datetimeEndAll)]
    Te_Epoch = Te_Epoch[selected_index]
    Te = Te[selected_index]
#print(Te.shape)
#print(Te_Epoch.shape)

# Process missing value. missing value = -9.9999998e+30.
BGSE[abs(BGSE) > 80] = np.nan # B field.
Np[Np < -1e+10] = np.nan # Proton number density.
VGSE[abs(VGSE) > 1500] = np.nan # Solar wind speed.
Tp[Tp < -1e+10] = np.nan # Proton temperature.
if 'Te' in locals(): # If Elec_Te_all is defined.
    Te[Te < -1e+10] = np.nan # Electron temperature.

# Put data into DataFrame.
print('Putting BGSE into DataFrame...')
BGSE_DataFrame = pd.DataFrame(BGSE, index = BGSE_Epoch, columns = ['Bx', 'By', 'Bz'])
print('Putting VGSE into DataFrame...')
VGSE_DataFrame = pd.DataFrame(VGSE, index = SW_Epoch, columns = ['Vx', 'Vy', 'Vz'])
print('Putting Np into DataFrame...')
Np_DataFrame = pd.DataFrame(Np, index = SW_Epoch, columns = ['Np'])
print('Putting Tp into DataFrame...')
Tp_DataFrame = pd.DataFrame(Tp, index = SW_Epoch, columns = ['Tp'])
print('Putting Te into DataFrame...')
if 'Te' in locals(): # If Elec_Te_all is defined.
    print('Te exists.')
    Te_DataFrame = pd.DataFrame(Te, index = Te_Epoch, columns = ['Te'])

# Drop duplicated records. This is the flaw of the source data.
print('Dropping duplicated records...')
VGSE_DataFrame.drop_duplicates(keep='first', inplace=True) # Drop duplicated records, keep first one.
BGSE_DataFrame.drop_duplicates(keep='first', inplace=True) # Drop duplicated records, keep first one.
Np_DataFrame.drop_duplicates(keep='first', inplace=True) # Drop duplicated records, keep first one.
Tp_DataFrame.drop_duplicates(keep='first', inplace=True) # Drop duplicated records, keep first one.
if 'Te' in locals(): # If Elec_Te_all is defined.
    print('Sorting Te...')
    Te_DataFrame.drop_duplicates(keep='first', inplace=True) # Drop duplicated records, keep first one.

# Sort data by time index. Time series data may be not in order, This is the flaw of the source data.
print('Sorting data...')
VGSE_DataFrame.sort_index(axis=0, ascending=True, inplace=True, kind='quicksort')
BGSE_DataFrame.sort_index(axis=0, ascending=True, inplace=True, kind='quicksort')
Np_DataFrame.sort_index(axis=0, ascending=True, inplace=True, kind='quicksort')
Tp_DataFrame.sort_index(axis=0, ascending=True, inplace=True, kind='quicksort')
if 'Te' in locals(): # If Elec_Te_all is defined.
    Te_DataFrame.sort_index(axis=0, ascending=True, inplace=True, kind='quicksort')

#========================================= Process VGSE missing value =========================================
print('\nProcessing VGSE...')
# Keep original data.
VGSE_DataFrame0 = VGSE_DataFrame.copy(deep=True)
print('VGSE_DataFrame.shape = {}'.format(VGSE_DataFrame.shape))

# Remove all data which fall outside three standard deviations.
# 1*std = 68.27%
# 2*std = 95.45%
# 3*std = 99.73%
n_removed_Vx_total = 0
n_removed_Vy_total = 0
n_removed_Vz_total = 0
print('\nRemove all VGSE data which fall outside three standard deviations...')
n_std = 3 # Three std.
Vx_std, Vy_std, Vz_std = VGSE_DataFrame.std(skipna=True, numeric_only=True)
Vx_mean, Vy_mean, Vz_mean = VGSE_DataFrame.mean(skipna=True, numeric_only=True)
Vx_remove = (VGSE_DataFrame['Vx']<(Vx_mean-n_std*Vx_std))|(VGSE_DataFrame['Vx']>(Vx_mean+n_std*Vx_std))
Vy_remove = (VGSE_DataFrame['Vy']<(Vy_mean-n_std*Vy_std))|(VGSE_DataFrame['Vy']>(Vy_mean+n_std*Vy_std))
Vz_remove = (VGSE_DataFrame['Vz']<(Vz_mean-n_std*Vz_std))|(VGSE_DataFrame['Vz']>(Vz_mean+n_std*Vz_std))
VGSE_DataFrame['Vx'][Vx_remove] = np.nan
VGSE_DataFrame['Vy'][Vy_remove] = np.nan
VGSE_DataFrame['Vz'][Vz_remove] = np.nan
print('V_std:', Vx_std, Vy_std, Vz_std)
print('V_mean:', Vx_mean, Vy_mean, Vz_mean)
Vx_lower_boundary = Vx_mean-n_std*Vx_std
Vx_upper_boundary = Vx_mean+n_std*Vx_std
Vy_lower_boundary = Vy_mean-n_std*Vy_std
Vy_upper_boundary = Vy_mean+n_std*Vy_std
Vz_lower_boundary = Vz_mean-n_std*Vz_std
Vz_upper_boundary = Vz_mean+n_std*Vz_std
print('The VGSE Vx value range within 3 std is [{}, {}]'.format(Vx_lower_boundary, Vx_upper_boundary))
print('The VGSE Vy value range within 3 std is [{}, {}]'.format(Vy_lower_boundary, Vy_upper_boundary))
print('The VGSE Vz value range within 3 std is [{}, {}]'.format(Vz_lower_boundary, Vz_upper_boundary))
n_removed_Vx = sum(Vx_remove)
n_removed_Vy = sum(Vy_remove)
n_removed_Vz = sum(Vz_remove)
n_removed_Vx_total += n_removed_Vx
n_removed_Vy_total += n_removed_Vy
n_removed_Vz_total += n_removed_Vz
print('In Vx, {} data has been removed!'.format(n_removed_Vx))
print('In Vy, {} data has been removed!'.format(n_removed_Vy))
print('In Vz, {} data has been removed!'.format(n_removed_Vz))
print('Till now, in Vx, {} records have been removed!'.format(n_removed_Vx_total))
print('Till now, in Vy, {} records have been removed!'.format(n_removed_Vy_total))
print('Till now, in Vz, {} records have been removed!'.format(n_removed_Vz_total))
print('\n')

# Apply Butterworth filter two times.
# 1st time, set cutoff frequency to 0.005 to remove large outliers.
# 2nd time, set cutoff frequency to 0.05 to remove spikes.
for Wn in [0.005, 0.05]:
    if Wn==0.005:
        print('Applying Butterworth filter 1st time, remove large outliers...')
    if Wn==0.05:
        print('Applying Butterworth filter 2nd time, remove spikes...')
    # Note that, sp.signal.butter cannot handle np.nan. Fill the nan before using it.
    VGSE_DataFrame.fillna(method='ffill', inplace=True) # Fill missing data, forward copy.
    VGSE_DataFrame.fillna(method='bfill', inplace=True) # Fill missing data, backward copy.
    # Create an empty DataFrame to store the filtered data.
    VGSE_LowPass = pd.DataFrame(index = VGSE_DataFrame.index, columns = ['Vx', 'Vy', 'Vz'])
    # Design the Buterworth filter.
    N  = 2    # Filter order
    B, A = sp.signal.butter(N, Wn, output='ba')
    # Apply the filter.
    VGSE_LowPass['Vx'] = sp.signal.filtfilt(B, A, VGSE_DataFrame['Vx'])
    VGSE_LowPass['Vy'] = sp.signal.filtfilt(B, A, VGSE_DataFrame['Vy'])
    VGSE_LowPass['Vz'] = sp.signal.filtfilt(B, A, VGSE_DataFrame['Vz'])
    # Calculate the difference between VGSE_LowPass and VGSE_DataFrame.
    VGSE_dif = pd.DataFrame(index = VGSE_DataFrame.index, columns = ['Vx', 'Vy', 'Vz']) # Generate empty DataFrame.
    VGSE_dif['Vx'] = VGSE_DataFrame['Vx'] - VGSE_LowPass['Vx']
    VGSE_dif['Vy'] = VGSE_DataFrame['Vy'] - VGSE_LowPass['Vy']
    VGSE_dif['Vz'] = VGSE_DataFrame['Vz'] - VGSE_LowPass['Vz']
    # Calculate the mean and standard deviation of VGSE_dif.
    Vx_dif_std, Vy_dif_std, Vz_dif_std = VGSE_dif.std(skipna=True, numeric_only=True)
    Vx_dif_mean, Vy_dif_mean, Vz_dif_mean = VGSE_dif.mean(skipna=True, numeric_only=True)
    # Set the values fall outside n*std to np.nan.
    n_dif_std = 3
    Vx_remove = (VGSE_dif['Vx']<(Vx_dif_mean-n_dif_std*Vx_dif_std))|(VGSE_dif['Vx']>(Vx_dif_mean+n_dif_std*Vx_dif_std))
    Vy_remove = (VGSE_dif['Vy']<(Vy_dif_mean-n_dif_std*Vy_dif_std))|(VGSE_dif['Vy']>(Vy_dif_mean+n_dif_std*Vy_dif_std))
    Vz_remove = (VGSE_dif['Vz']<(Vz_dif_mean-n_dif_std*Vz_dif_std))|(VGSE_dif['Vz']>(Vz_dif_mean+n_dif_std*Vz_dif_std))
    VGSE_DataFrame['Vx'][Vx_remove] = np.nan
    VGSE_DataFrame['Vy'][Vy_remove] = np.nan
    VGSE_DataFrame['Vz'][Vz_remove] = np.nan
    print('V_dif_std:', Vx_dif_std, Vy_dif_std, Vz_dif_std)
    print('V_dif_mean:', Vx_dif_mean, Vy_dif_mean, Vz_dif_mean)
    Vx_dif_lower_boundary = Vx_dif_mean-n_std*Vx_dif_std
    Vx_dif_upper_boundary = Vx_dif_mean+n_std*Vx_dif_std
    Vy_dif_lower_boundary = Vy_dif_mean-n_std*Vy_dif_std
    Vy_dif_upper_boundary = Vy_dif_mean+n_std*Vy_dif_std
    Vz_dif_lower_boundary = Vz_dif_mean-n_std*Vz_dif_std
    Vz_dif_upper_boundary = Vz_dif_mean+n_std*Vz_dif_std
    print('The VGSE Vx_dif value range within three std is [{}, {}]'.format(Vx_dif_lower_boundary, Vx_dif_upper_boundary))
    print('The VGSE Vy_dif value range within three std is [{}, {}]'.format(Vy_dif_lower_boundary, Vy_dif_upper_boundary))
    print('The VGSE Vz_dif value range within three std is [{}, {}]'.format(Vz_dif_lower_boundary, Vz_dif_upper_boundary))
    n_removed_Vx = sum(Vx_remove)
    n_removed_Vy = sum(Vy_remove)
    n_removed_Vz = sum(Vz_remove)
    n_removed_Vx_total += n_removed_Vx
    n_removed_Vy_total += n_removed_Vy
    n_removed_Vz_total += n_removed_Vz
    print('In Vx, this operation removed {} records!'.format(n_removed_Vx))
    print('In Vy, this operation removed {} records!'.format(n_removed_Vy))
    print('In Vz, this operation removed {} records!!'.format(n_removed_Vz))
    print('Till now, in Vx, {} records have been removed!'.format(n_removed_Vx_total))
    print('Till now, in Vy, {} records have been removed!'.format(n_removed_Vy_total))
    print('Till now, in Vz, {} records have been removed!'.format(n_removed_Vz_total))
    print('\n')

# Plot VGSE filter process.
print('Plotting VGSE filtering process...')
fig_line_width = 0.1
fig_ylabel_fontsize = 9
fig_xtick_fontsize = 8
fig_ytick_fontsize = 8
fig_legend_size = 5
fig,ax = plt.subplots(6,1, sharex=True,figsize=(18, 10))
Vx_plot = ax[0]
Vx_dif = ax[1]
Vy_plot = ax[2]
Vy_dif = ax[3]
Vz_plot = ax[4]
Vz_dif = ax[5]
# Plotting Vx filter process.
Vx_plot.plot(VGSE_DataFrame0[datetime_start:datetime_end].index, VGSE_DataFrame0['Vx'][datetime_start:datetime_end],\
             color = 'red', linewidth=fig_line_width, label='Vx_original') # Original data.
Vx_plot.plot(VGSE_DataFrame[datetime_start:datetime_end].index, VGSE_DataFrame['Vx'][datetime_start:datetime_end],\
             color = 'blue', linewidth=fig_line_width, label='Vx_processed') # Filtered data.
Vx_plot.plot(VGSE_LowPass[datetime_start:datetime_end].index, VGSE_LowPass['Vx'][datetime_start:datetime_end],\
             color = 'black', linewidth=fig_line_width, label='Vx_LowPass') # Low pass curve.
Vx_plot.set_ylabel('Vx', fontsize=fig_ylabel_fontsize)
Vx_plot.legend(loc='upper left',prop={'size':fig_legend_size})
Vx_plot.tick_params(axis='y', which='major', labelsize=fig_ytick_fontsize)
Vx_dif.plot(VGSE_dif[datetime_start:datetime_end].index, VGSE_dif['Vx'][datetime_start:datetime_end],\
            color = 'green', linewidth=fig_line_width) # Difference data.
Vx_dif.tick_params(axis='y', which='major', labelsize=fig_ytick_fontsize)
Vx_dif.set_ylabel('Vx_dif', fontsize=fig_ylabel_fontsize)
# Plotting Vy filter process.
Vy_plot.plot(VGSE_DataFrame0[datetime_start:datetime_end].index, VGSE_DataFrame0['Vy'][datetime_start:datetime_end],\
             color = 'red', linewidth=fig_line_width, label='Vy_original') # Original data.
Vy_plot.plot(VGSE_DataFrame[datetime_start:datetime_end].index, VGSE_DataFrame['Vy'][datetime_start:datetime_end],\
             color = 'blue', linewidth=fig_line_width, label='Vy_processed') # Filtered data.
Vy_plot.plot(VGSE_LowPass[datetime_start:datetime_end].index, VGSE_LowPass['Vy'][datetime_start:datetime_end],\
             color = 'black', linewidth=fig_line_width, label='Vy_LowPass') # Low pass curve.
Vy_plot.set_ylabel('Vy', fontsize=fig_ylabel_fontsize)
Vy_plot.legend(loc='upper left',prop={'size':fig_legend_size})
Vy_plot.tick_params(axis='y', which='major', labelsize=fig_ytick_fontsize)
Vy_dif.plot(VGSE_dif[datetime_start:datetime_end].index, VGSE_dif['Vy'][datetime_start:datetime_end],\
            color = 'green', linewidth=fig_line_width) # Difference data.
Vy_dif.tick_params(axis='y', which='major', labelsize=fig_ytick_fontsize)
Vy_dif.set_ylabel('Vy_dif', fontsize=fig_ylabel_fontsize)
# Plotting Vz filter process.
Vz_plot.plot(VGSE_DataFrame0[datetime_start:datetime_end].index, VGSE_DataFrame0['Vz'][datetime_start:datetime_end],\
             color = 'red', linewidth=fig_line_width, label='Vz_original') # Original data.
Vz_plot.plot(VGSE_DataFrame[datetime_start:datetime_end].index, VGSE_DataFrame['Vz'][datetime_start:datetime_end],\
             color = 'blue', linewidth=fig_line_width, label='Vz_processed') # Filtered data.
Vz_plot.plot(VGSE_LowPass[datetime_start:datetime_end].index, VGSE_LowPass['Vz'][datetime_start:datetime_end],\
             color = 'black', linewidth=fig_line_width, label='Vz_LowPass') # Low pass curve.
Vz_plot.set_ylabel('Vz', fontsize=fig_ylabel_fontsize)
Vz_plot.legend(loc='upper left',prop={'size':fig_legend_size})
Vz_plot.tick_params(axis='y', which='major', labelsize=fig_ytick_fontsize)
Vz_dif.plot(VGSE_dif[datetime_start:datetime_end].index, VGSE_dif['Vz'][datetime_start:datetime_end],\
            color = 'green', linewidth=fig_line_width) # Difference data.
Vz_dif.tick_params(axis='y', which='major', labelsize=fig_ytick_fontsize)
Vz_dif.set_ylabel('Vz_dif', fontsize=fig_ylabel_fontsize)
# This is a shared axis for all subplot
Vz_dif.tick_params(axis='x', which='major', labelsize=fig_xtick_fontsize)
# Save plot.
fig.savefig('GS_filter_process_VGSE('+str(datetime_start)+'~'+str(datetime_end)+').png',\
            format='png', dpi=500, bbox_inches='tight')

#========================================= Process BGSE missing value =========================================
print('\nProcessing BGSE...')
# Keep original data.
BGSE_DataFrame0 = BGSE_DataFrame.copy(deep=True)
print('BGSE_DataFrame.shape = {}'.format(BGSE_DataFrame.shape))


# Remove all data which fall outside three standard deviations.
# 1*std = 68.27%
# 2*std = 95.45%
# 3*std = 99.73%
n_removed_Bx_total = 0
n_removed_By_total = 0
n_removed_Bz_total = 0

'''
print('\nRemove all BGSE data which fall outside 4 standard deviations...')
n_std = 4 # 4 std.
Bx_std, By_std, Bz_std = BGSE_DataFrame.std(skipna=True, numeric_only=True)
Bx_mean, By_mean, Bz_mean = BGSE_DataFrame.mean(skipna=True, numeric_only=True)
Bx_remove = (BGSE_DataFrame['Bx']<(Bx_mean-n_std*Bx_std))|(BGSE_DataFrame['Bx']>(Bx_mean+n_std*Bx_std))
By_remove = (BGSE_DataFrame['By']<(By_mean-n_std*By_std))|(BGSE_DataFrame['By']>(By_mean+n_std*By_std))
Bz_remove = (BGSE_DataFrame['Bz']<(Bz_mean-n_std*Bz_std))|(BGSE_DataFrame['Bz']>(Bz_mean+n_std*Bz_std))
BGSE_DataFrame['Bx'][Bx_remove] = np.nan
BGSE_DataFrame['By'][By_remove] = np.nan
BGSE_DataFrame['Bz'][Bz_remove] = np.nan
print('B_std:', Bx_std, By_std, Bz_std)
print('B_mean:', Bx_mean, By_mean, Bz_mean)
Bx_lower_boundary = Bx_mean-n_std*Bx_std
Bx_upper_boundary = Bx_mean+n_std*Bx_std
By_lower_boundary = By_mean-n_std*By_std
By_upper_boundary = By_mean+n_std*By_std
Bz_lower_boundary = Bz_mean-n_std*Bz_std
Bz_upper_boundary = Bz_mean+n_std*Bz_std
print('The BGSE Bx value range within 4 std is [{}, {}]'.format(Bx_lower_boundary, Bx_upper_boundary))
print('The BGSE By value range within 4 std is [{}, {}]'.format(By_lower_boundary, By_upper_boundary))
print('The BGSE Bz value range within 4 std is [{}, {}]'.format(Bz_lower_boundary, Bz_upper_boundary))
n_removed_Bx = sum(Bx_remove)
n_removed_By = sum(By_remove)
n_removed_Bz = sum(Bz_remove)
n_removed_Bx_total += n_removed_Bx
n_removed_By_total += n_removed_By
n_removed_Bz_total += n_removed_Bz
print('In Bx, {} data has been removed!'.format(n_removed_Bx))
print('In By, {} data has been removed!'.format(n_removed_By))
print('In Bz, {} data has been removed!'.format(n_removed_Bz))
print('Till now, in Bx, {} records have been removed!'.format(n_removed_Bx_total))
print('Till now, in By, {} records have been removed!'.format(n_removed_By_total))
print('Till now, in Bz, {} records have been removed!'.format(n_removed_Bz_total))
print('\n')
'''

# Apply Butterworth filter.
# Set cutoff frequency to 0.45 to remove spikes.
for Wn in [0.45]:
    print('Applying Butterworth filter with cut frequency = {}, remove spikes...'.format(Wn))
    # Note that, sp.signal.butter cannot handle np.nan. Fill the nan before using it.
    BGSE_DataFrame.fillna(method='ffill', inplace=True) # Fill missing data, forward copy.
    BGSE_DataFrame.fillna(method='bfill', inplace=True) # Fill missing data, backward copy.
    # Create an empty DataFrame to store the filtered data.
    BGSE_LowPass = pd.DataFrame(index = BGSE_DataFrame.index, columns = ['Bx', 'By', 'Bz'])
    # Design the Buterworth filter.
    N  = 2    # Filter order
    B, A = sp.signal.butter(N, Wn, output='ba')
    # Apply the filter.
    BGSE_LowPass['Bx'] = sp.signal.filtfilt(B, A, BGSE_DataFrame['Bx'])
    BGSE_LowPass['By'] = sp.signal.filtfilt(B, A, BGSE_DataFrame['By'])
    BGSE_LowPass['Bz'] = sp.signal.filtfilt(B, A, BGSE_DataFrame['Bz'])
    # Calculate the difference between BGSE_LowPass and BGSE_DataFrame.
    BGSE_dif = pd.DataFrame(index = BGSE_DataFrame.index, columns = ['Bx', 'By', 'Bz']) # Generate empty DataFrame.
    BGSE_dif['Bx'] = BGSE_DataFrame['Bx'] - BGSE_LowPass['Bx']
    BGSE_dif['By'] = BGSE_DataFrame['By'] - BGSE_LowPass['By']
    BGSE_dif['Bz'] = BGSE_DataFrame['Bz'] - BGSE_LowPass['Bz']
    # Calculate the mean and standard deviation of BGSE_dif.
    Bx_dif_std, By_dif_std, Bz_dif_std = BGSE_dif.std(skipna=True, numeric_only=True)
    Bx_dif_mean, By_dif_mean, Bz_dif_mean = BGSE_dif.mean(skipna=True, numeric_only=True)
    # Set the values fall outside n*std to np.nan.
    n_dif_std = 3
    Bx_remove = (BGSE_dif['Bx']<(Bx_dif_mean-n_dif_std*Bx_dif_std))|(BGSE_dif['Bx']>(Bx_dif_mean+n_dif_std*Bx_dif_std))
    By_remove = (BGSE_dif['By']<(By_dif_mean-n_dif_std*By_dif_std))|(BGSE_dif['By']>(By_dif_mean+n_dif_std*By_dif_std))
    Bz_remove = (BGSE_dif['Bz']<(Bz_dif_mean-n_dif_std*Bz_dif_std))|(BGSE_dif['Bz']>(Bz_dif_mean+n_dif_std*Bz_dif_std))
    BGSE_DataFrame['Bx'][Bx_remove] = np.nan
    BGSE_DataFrame['By'][By_remove] = np.nan
    BGSE_DataFrame['Bz'][Bz_remove] = np.nan
    print('B_dif_std:', Bx_dif_std, By_dif_std, Bz_dif_std)
    print('B_dif_mean:', Bx_dif_mean, By_dif_mean, Bz_dif_mean)
    Bx_dif_lower_boundary = Bx_dif_mean-n_std*Bx_dif_std
    Bx_dif_upper_boundary = Bx_dif_mean+n_std*Bx_dif_std
    By_dif_lower_boundary = By_dif_mean-n_std*By_dif_std
    By_dif_upper_boundary = By_dif_mean+n_std*By_dif_std
    Bz_dif_lower_boundary = Bz_dif_mean-n_std*Bz_dif_std
    Bz_dif_upper_boundary = Bz_dif_mean+n_std*Bz_dif_std
    print('The BGSE Bx_dif value range within three std is [{}, {}]'.format(Bx_dif_lower_boundary, Bx_dif_upper_boundary))
    print('The BGSE By_dif value range within three std is [{}, {}]'.format(By_dif_lower_boundary, By_dif_upper_boundary))
    print('The BGSE Bz_dif value range within three std is [{}, {}]'.format(Bz_dif_lower_boundary, Bz_dif_upper_boundary))
    n_removed_Bx = sum(Bx_remove)
    n_removed_By = sum(By_remove)
    n_removed_Bz = sum(Bz_remove)
    n_removed_Bx_total += n_removed_Bx
    n_removed_By_total += n_removed_By
    n_removed_Bz_total += n_removed_Bz
    print('In Bx, this operation removed {} records!'.format(n_removed_Bx))
    print('In By, this operation removed {} records!'.format(n_removed_By))
    print('In Bz, this operation removed {} records!!'.format(n_removed_Bz))
    print('Till now, in Bx, {} records have been removed!'.format(n_removed_Bx_total))
    print('Till now, in By, {} records have been removed!'.format(n_removed_By_total))
    print('Till now, in Bz, {} records have been removed!'.format(n_removed_Bz_total))
    print('\n')

# Plot BGSE filter process.
print('Plotting BGSE filtering process...')
fig_line_width = 0.1
fig_ylabel_fontsize = 9
fig_xtick_fontsize = 8
fig_ytick_fontsize = 8
fig_legend_size = 5
fig,ax = plt.subplots(6,1, sharex=True,figsize=(18, 10))
Bx_plot = ax[0]
Bx_dif = ax[1]
By_plot = ax[2]
By_dif = ax[3]
Bz_plot = ax[4]
Bz_dif = ax[5]
# Plotting Bx filter process.
Bx_plot.plot(BGSE_DataFrame0[datetime_start:datetime_end].index, BGSE_DataFrame0['Bx'][datetime_start:datetime_end],\
             color = 'red', linewidth=fig_line_width, label='Bx_original') # Original data.
Bx_plot.plot(BGSE_DataFrame[datetime_start:datetime_end].index, BGSE_DataFrame['Bx'][datetime_start:datetime_end],\
             color = 'blue', linewidth=fig_line_width, label='Bx_processed') # Filtered data.
Bx_plot.plot(BGSE_LowPass[datetime_start:datetime_end].index, BGSE_LowPass['Bx'][datetime_start:datetime_end],\
             color = 'black', linewidth=fig_line_width, label='Bx_LowPass') # Low pass curve.
Bx_plot.set_ylabel('Bx', fontsize=fig_ylabel_fontsize)
Bx_plot.legend(loc='upper left',prop={'size':fig_legend_size})
Bx_plot.tick_params(axis='y', which='major', labelsize=fig_ytick_fontsize)
Bx_dif.plot(BGSE_dif[datetime_start:datetime_end].index, BGSE_dif['Bx'][datetime_start:datetime_end],\
            color = 'green', linewidth=fig_line_width) # Difference data.
Bx_dif.tick_params(axis='y', which='major', labelsize=fig_ytick_fontsize)
Bx_dif.set_ylabel('Bx_dif', fontsize=fig_ylabel_fontsize)
# Plotting By filter process.
By_plot.plot(BGSE_DataFrame0[datetime_start:datetime_end].index, BGSE_DataFrame0['By'][datetime_start:datetime_end],\
             color = 'red', linewidth=fig_line_width, label='By_original') # Original data.
By_plot.plot(BGSE_DataFrame[datetime_start:datetime_end].index, BGSE_DataFrame['By'][datetime_start:datetime_end],\
             color = 'blue', linewidth=fig_line_width, label='By_processed') # Filtered data.
By_plot.plot(BGSE_LowPass[datetime_start:datetime_end].index, BGSE_LowPass['By'][datetime_start:datetime_end],\
             color = 'black', linewidth=fig_line_width, label='By_LowPass') # Low pass curve.
By_plot.set_ylabel('By', fontsize=fig_ylabel_fontsize)
By_plot.tick_params(axis='y', which='major', labelsize=fig_ytick_fontsize)
By_plot.legend(loc='upper left',prop={'size':fig_legend_size})
By_dif.plot(BGSE_dif[datetime_start:datetime_end].index, BGSE_dif['By'][datetime_start:datetime_end],\
            color = 'green', linewidth=fig_line_width) # Difference data.
By_dif.tick_params(axis='y', which='major', labelsize=fig_ytick_fontsize)
By_dif.set_ylabel('By_dif', fontsize=fig_ylabel_fontsize)
# Plotting Bz filter process.
Bz_plot.plot(BGSE_DataFrame0[datetime_start:datetime_end].index, BGSE_DataFrame0['Bz'][datetime_start:datetime_end],\
             color = 'red', linewidth=fig_line_width, label='Bz_original') # Original data.
Bz_plot.plot(BGSE_DataFrame[datetime_start:datetime_end].index, BGSE_DataFrame['Bz'][datetime_start:datetime_end],\
             color = 'blue', linewidth=fig_line_width, label='Bz_processed') # Filtered data.
Bz_plot.plot(BGSE_LowPass[datetime_start:datetime_end].index, BGSE_LowPass['Bz'][datetime_start:datetime_end],\
             color = 'black', linewidth=fig_line_width, label='Bz_LowPass') # Low pass curve.
Bz_plot.set_ylabel('Bz', fontsize=fig_ylabel_fontsize)
Bz_plot.tick_params(axis='y', which='major', labelsize=fig_ytick_fontsize)
Bz_plot.legend(loc='upper left',prop={'size':fig_legend_size})
Bz_dif.plot(BGSE_dif[datetime_start:datetime_end].index, BGSE_dif['Bz'][datetime_start:datetime_end],\
            color = 'green', linewidth=fig_line_width) # Difference data.
Bz_dif.tick_params(axis='y', which='major', labelsize=fig_ytick_fontsize)
Bz_dif.set_ylabel('Bz_dif', fontsize=fig_ylabel_fontsize)
# This is a shared axis for all subplot
Bz_dif.tick_params(axis='x', which='major', labelsize=fig_xtick_fontsize)
# Save plot.
fig.savefig('GS_filter_process_BGSE('+str(datetime_start)+'~'+str(datetime_end)+').png',\
            format='png', dpi=500, bbox_inches='tight')

print('Done.')


# ========================================= Process Np missing value =========================================
print('\nProcessing Np...')
# Keep original data.
Np_DataFrame0 = Np_DataFrame.copy(deep=True)
print('Np_DataFrame.shape = {}'.format(Np_DataFrame.shape))

# Remove all data which fall outside 4 standard deviations.
# 1*std = 68.27%
# 2*std = 95.45%
# 3*std = 99.73%
# 3.5*std = 99.95%
# 4*std = 99.99%
n_removed_Np_total = 0
print('\nRemove all Np data which fall outside 3.5 standard deviations...')
n_std = 3.5 # 3.5 std.
Np_std = Np_DataFrame.std(skipna=True, numeric_only=True)[0]
Np_mean = Np_DataFrame.mean(skipna=True, numeric_only=True)[0]
Np_remove = (Np_DataFrame['Np']<(Np_mean-n_std*Np_std))|(Np_DataFrame['Np']>(Np_mean+n_std*Np_std))
Np_DataFrame['Np'][Np_remove] = np.nan
print('Np_std:', Np_std)
print('Np_mean:', Np_mean)
Np_lower_boundary = Np_mean-n_std*Np_std
Np_upper_boundary = Np_mean+n_std*Np_std
print('The Np value range within 3.5 std is [{}, {}]'.format(Np_lower_boundary, Np_upper_boundary))
n_removed_Np = sum(Np_remove)
n_removed_Np_total += n_removed_Np
print('In Np, {} data has been removed!'.format(n_removed_Np))
print('Till now, in Np, {} records have been removed!'.format(n_removed_Np_total))
print('\n')

# Apply Butterworth filter to Np.
# Set cutoff frequency to 0.5 to remove spikes.
for Wn in [0.05, 0.7]: # Np
    print('Applying Butterworth filter with cut frequency = {}, remove spikes...'.format(Wn))
    # Note that, sp.signal.butter cannot handle np.nan. Fill the nan before using it.
    Np_DataFrame.fillna(method='ffill', inplace=True) # Fill missing data, forward copy.
    Np_DataFrame.fillna(method='bfill', inplace=True) # Fill missing data, backward copy.
    # Create an empty DataFrame to store the filtered data.
    Np_LowPass = pd.DataFrame(index = Np_DataFrame.index, columns = ['Np'])
    # Design the Buterworth filter.
    N  = 2    # Filter order
    B, A = sp.signal.butter(N, Wn, output='ba')
    # Apply the filter.
    Np_LowPass['Np'] = sp.signal.filtfilt(B, A, Np_DataFrame['Np'])
    # Calculate the difference between Np_LowPass and Np_DataFrame.
    Np_dif = pd.DataFrame(index = Np_DataFrame.index, columns = ['Np']) # Generate empty DataFrame.
    Np_dif['Np'] = Np_DataFrame['Np'] - Np_LowPass['Np']
    # Calculate the mean and standard deviation of Np_dif. Np_dif_std is a Series object, so [0] is added.
    Np_dif_std = Np_dif.std(skipna=True, numeric_only=True)[0]
    Np_dif_mean = Np_dif.mean(skipna=True, numeric_only=True)[0]
    # Set the values fall outside n*std to np.nan.
    n_dif_std = 3
    Np_remove = (Np_dif['Np']<(Np_dif_mean-n_dif_std*Np_dif_std))|(Np_dif['Np']>(Np_dif_mean+n_dif_std*Np_dif_std))
    Np_DataFrame[Np_remove] = np.nan
    print('Np_dif_std:', Np_dif_std)
    print('Np_dif_mean:', Np_dif_mean)
    Np_dif_lower_boundary = Np_dif_mean-n_dif_std*Np_dif_std
    Np_dif_upper_boundary = Np_dif_mean+n_dif_std*Np_dif_std
    print('The Np_dif value range within 3 std is [{}, {}]'.format(Np_dif_lower_boundary, Np_dif_upper_boundary))
    n_removed_Np = sum(Np_remove)
    n_removed_Np_total += n_removed_Np
    print('In Np, this operation removed {} records!'.format(n_removed_Np))
    print('Till now, in Np, {} records have been removed!'.format(n_removed_Np_total))

# ========================================= Process Tp missing value =========================================
print('\nProcessing Tp...')
# Keep original data.
Tp_DataFrame0 = Tp_DataFrame.copy(deep=True)
print('Tp_DataFrame.shape = {}'.format(Tp_DataFrame.shape))

# Remove all data which fall outside 4 standard deviations.
# 1*std = 68.27%
# 2*std = 95.45%
# 3*std = 99.73%
# 3.5*std = 99.95%
# 4*std = 99.99%
n_removed_Tp_total = 0
print('\nRemove all Tp data which fall outside 3.5 standard deviations...')
n_std = 3.5 # 3.5 std.
Tp_std = Tp_DataFrame.std(skipna=True, numeric_only=True)[0]
Tp_mean = Tp_DataFrame.mean(skipna=True, numeric_only=True)[0]
Tp_remove = (Tp_DataFrame['Tp']<(Tp_mean-n_std*Tp_std))|(Tp_DataFrame['Tp']>(Tp_mean+n_std*Tp_std))
Tp_DataFrame['Tp'][Tp_remove] = np.nan
print('Tp_std:', Tp_std)
print('Tp_mean:', Tp_mean)
Tp_lower_boundary = Tp_mean-n_std*Tp_std
Tp_upper_boundary = Tp_mean+n_std*Tp_std
print('The Tp value range within 3.5 std is [{}, {}]'.format(Tp_lower_boundary, Tp_upper_boundary))
n_removed_Tp = sum(Tp_remove)
n_removed_Tp_total += n_removed_Tp
print('In Tp, {} data has been removed!'.format(n_removed_Tp))
print('Till now, in Tp, {} records have been removed!'.format(n_removed_Tp_total))
print('\n')

# Apply Butterworth filter.
# Set cutoff frequency to 0.7 to remove spikes.
for Wn in [0.05, 0.7]: # Tp
    print('Applying Butterworth filter with cut frequency = {}, remove spikes...'.format(Wn))
    # Note that, sp.signal.butter cannot handle np.nan. Fill the nan before using it.
    Tp_DataFrame.fillna(method='ffill', inplace=True) # Fill missing data, forward copy.
    Tp_DataFrame.fillna(method='bfill', inplace=True) # Fill missing data, backward copy.
    # Create an empty DataFrame to store the filtered data.
    Tp_LowPass = pd.DataFrame(index = Tp_DataFrame.index, columns = ['Tp'])
    # Design the Buterworth filter.
    N  = 2    # Filter order
    B, A = sp.signal.butter(N, Wn, output='ba')
    # Apply the filter.
    Tp_LowPass['Tp'] = sp.signal.filtfilt(B, A, Tp_DataFrame['Tp'])
    # Calculate the difference between Tp_LowPass and Tp_DataFrame.
    Tp_dif = pd.DataFrame(index = Tp_DataFrame.index, columns = ['Tp']) # Generate empty DataFrame.
    Tp_dif['Tp'] = Tp_DataFrame['Tp'] - Tp_LowPass['Tp']
    # Calculate the mean and standard deviation of Tp_dif. Tp_dif_std is a Series object, so [0] is added.
    Tp_dif_std = Tp_dif.std(skipna=True, numeric_only=True)[0]
    Tp_dif_mean = Tp_dif.mean(skipna=True, numeric_only=True)[0]
    # Set the values fall outside n*std to np.nan.
    n_dif_std = 3
    Tp_remove = (Tp_dif['Tp']<(Tp_dif_mean-n_dif_std*Tp_dif_std))|(Tp_dif['Tp']>(Tp_dif_mean+n_dif_std*Tp_dif_std))
    Tp_DataFrame[Tp_remove] = np.nan
    print('Tp_dif_std:', Tp_dif_std)
    print('Tp_dif_mean:', Tp_dif_mean)
    Tp_dif_lower_boundary = Tp_dif_mean-n_dif_std*Tp_dif_std
    Tp_dif_upper_boundary = Tp_dif_mean+n_dif_std*Tp_dif_std
    print('The Tp_dif value range within 3 std is [{}, {}]'.format(Tp_dif_lower_boundary, Tp_dif_upper_boundary))
    n_removed_Tp = sum(Tp_remove)
    n_removed_Tp_total += n_removed_Tp
    print('In Tp, this operation removed {} records!'.format(n_removed_Tp))
    print('Till now, in Tp, {} records have been removed!'.format(n_removed_Tp_total))
    print('\n')

# ========================================= Process Te missing value =========================================
if 'Te' in locals():
    print('\nProcessing Te...')
    # Keep original data.
    Te_DataFrame0 = Te_DataFrame.copy(deep=True)
    print('Te_DataFrame.shape = {}'.format(Te_DataFrame.shape))

    # Remove all data which fall outside 3.5 standard deviations.
    # 1*std = 68.27%
    # 2*std = 95.45%
    # 3*std = 99.73%
    # 3.5*std = 99.95%
    # 4*std = 99.99%
    n_removed_Te_total = 0
    print('\nRemove all Te data which fall outside 3.5 standard deviations...')
    n_std = 3.5 # 3.5 std.
    Te_std = Te_DataFrame.std(skipna=True, numeric_only=True)[0]
    Te_mean = Te_DataFrame.mean(skipna=True, numeric_only=True)[0]
    Te_remove = (Te_DataFrame['Te']<(Te_mean-n_std*Te_std))|(Te_DataFrame['Te']>(Te_mean+n_std*Te_std))
    Te_DataFrame['Te'][Te_remove] = np.nan
    print('Te_std:', Te_std)
    print('Te_mean:', Te_mean)
    Te_lower_boundary = Te_mean-n_std*Te_std
    Te_upper_boundary = Te_mean+n_std*Te_std
    print('The Te value range within 3.5 std is [{}, {}]'.format(Te_lower_boundary, Te_upper_boundary))
    n_removed_Te = sum(Te_remove)
    n_removed_Te_total += n_removed_Te
    print('In Te, {} data has been removed!'.format(n_removed_Te))
    print('Till now, in Te, {} records have been removed!'.format(n_removed_Te_total))
    print('\n')

    # Apply Butterworth filter to Te.
    # Set cutoff frequency to 0.45 to remove spikes.
    for Wn in [0.05, 0.45]:
        print('Applying Butterworth filter with cut frequency = {}, remove spikes...'.format(Wn))
        # Note that, sp.signal.butter cannot handle np.nan. Fill the nan before using it.
        Te_DataFrame.fillna(method='ffill', inplace=True) # Fill missing data, forward copy.
        Te_DataFrame.fillna(method='bfill', inplace=True) # Fill missing data, backward copy.
        # Create an empty DataFrame to store the filtered data.
        Te_LowPass = pd.DataFrame(index = Te_DataFrame.index, columns = ['Te'])
        # Design the Buterworth filter.
        N  = 2    # Filter order
        B, A = sp.signal.butter(N, Wn, output='ba')
        # Apply the filter.
        Te_LowPass['Te'] = sp.signal.filtfilt(B, A, Te_DataFrame['Te'])
        # Calculate the difference between Tp_LowPass and Tp_DataFrame.
        Te_dif = pd.DataFrame(index = Te_DataFrame.index, columns = ['Te']) # Generate empty DataFrame.
        Te_dif['Te'] = Te_DataFrame['Te'] - Te_LowPass['Te']
        # Calculate the mean and standard deviation of Te_dif. Te_dif_std is a Series object, so [0] is added.
        Te_dif_std = Te_dif.std(skipna=True, numeric_only=True)[0]
        Te_dif_mean = Te_dif.mean(skipna=True, numeric_only=True)[0]
        # Set the values fall outside n*std to np.nan.
        n_dif_std = 3
        Te_remove = (Te_dif['Te']<(Te_dif_mean-n_dif_std*Te_dif_std))|(Te_dif['Te']>(Te_dif_mean+n_dif_std*Te_dif_std))
        Te_DataFrame[Te_remove] = np.nan
        print('Te_dif_std:', Te_dif_std)
        print('Te_dif_mean:', Te_dif_mean)
        Te_dif_lower_boundary = Te_dif_mean-n_dif_std*Te_dif_std
        Te_dif_upper_boundary = Te_dif_mean+n_dif_std*Te_dif_std
        print('The Te_dif value range within 3 std is [{}, {}]'.format(Te_dif_lower_boundary, Te_dif_upper_boundary))
        n_removed_Te = sum(Te_remove)
        n_removed_Te_total += n_removed_Te
        print('In Te, this operation removed {} records!'.format(n_removed_Te))
        print('Till now, in Te, {} records have been removed!'.format(n_removed_Te_total))
        print('\n')

# ===================================== Plot Np, Tp and Te filter process =====================================

fig_line_width = 0.1
fig_ylabel_fontsize = 9
fig_xtick_fontsize = 8
fig_ytick_fontsize = 8
fig_legend_size = 5
fig,ax = plt.subplots(6,1, sharex=True,figsize=(18, 10))
Np_plot = ax[0]
Np_dif_plot = ax[1]
Tp_plot = ax[2]
Tp_dif_plot = ax[3]
Te_plot = ax[4]
Te_dif_plot = ax[5]

# Plotting Np filter process.
print('Plotting Np filtering process...')
Np_plot.plot(Np_DataFrame0[datetime_start:datetime_end].index, Np_DataFrame0['Np'][datetime_start:datetime_end],\
             color = 'red', linewidth=fig_line_width, label='Np_original') # Original data.
Np_plot.plot(Np_DataFrame[datetime_start:datetime_end].index, Np_DataFrame['Np'][datetime_start:datetime_end],\
             color = 'blue', linewidth=fig_line_width, label='Np_processed') # Filtered data.
Np_plot.plot(Np_LowPass[datetime_start:datetime_end].index, Np_LowPass['Np'][datetime_start:datetime_end],\
             color = 'black', linewidth=fig_line_width, label='Np_LowPass') # Low pass curve.
Np_plot.set_ylabel('Np', fontsize=fig_ylabel_fontsize)
Np_plot.legend(loc='upper left',prop={'size':fig_legend_size})
Np_plot.tick_params(axis='y', which='major', labelsize=fig_ytick_fontsize)
Np_dif_plot.plot(Np_dif[datetime_start:datetime_end].index, Np_dif['Np'][datetime_start:datetime_end], color = 'green', linewidth=fig_line_width) # Difference data.
Np_dif_plot.tick_params(axis='y', which='major', labelsize=fig_ytick_fontsize)
Np_dif_plot.set_ylabel('Np_dif', fontsize=fig_ylabel_fontsize)
# Plotting Tp filter process.
print('Plotting Tp filtering process...')
Tp_plot.plot(Tp_DataFrame0[datetime_start:datetime_end].index, Tp_DataFrame0['Tp'][datetime_start:datetime_end],\
             color = 'red', linewidth=fig_line_width, label='Tp_original') # Original data.
Tp_plot.plot(Tp_DataFrame[datetime_start:datetime_end].index, Tp_DataFrame['Tp'][datetime_start:datetime_end],\
             color = 'blue', linewidth=fig_line_width, label='Tp_processed') # Filtered data.
Tp_plot.plot(Tp_LowPass[datetime_start:datetime_end].index, Tp_LowPass['Tp'][datetime_start:datetime_end],\
             color = 'black', linewidth=fig_line_width, label='Tp_LowPass') # Low pass curve.
Tp_plot.set_ylabel('Tp', fontsize=fig_ylabel_fontsize)
Tp_plot.legend(loc='upper left',prop={'size':fig_legend_size})
Tp_plot.tick_params(axis='y', which='major', labelsize=fig_ytick_fontsize)
Tp_dif_plot.plot(Tp_dif[datetime_start:datetime_end].index, Tp_dif['Tp'][datetime_start:datetime_end],\
                 color = 'green', linewidth=fig_line_width) # Difference data.
Tp_dif_plot.tick_params(axis='y', which='major', labelsize=fig_ytick_fontsize)
Tp_dif_plot.set_ylabel('Tp_dif', fontsize=fig_ylabel_fontsize)
# Plotting Te filter process.
if 'Te' in locals():
    print('Plotting Te filtering process...')
    Te_plot.plot(Te_DataFrame0[datetime_start:datetime_end].index, Te_DataFrame0['Te'][datetime_start:datetime_end],\
    color = 'red', linewidth=fig_line_width, label='Te_original') # Original data.
    Te_plot.plot(Te_DataFrame[datetime_start:datetime_end].index, Te_DataFrame['Te'][datetime_start:datetime_end],\
    color = 'blue', linewidth=fig_line_width, label='Te_processed') # Filtered data.
    Te_plot.plot(Te_LowPass[datetime_start:datetime_end].index, Te_LowPass['Te'][datetime_start:datetime_end],\
    color = 'black', linewidth=fig_line_width, label='Te_LowPass') # Low pass curve.
    Te_plot.set_ylabel('Te', fontsize=fig_ylabel_fontsize)
    Te_plot.legend(loc='upper left',prop={'size':fig_legend_size})
    Te_plot.tick_params(axis='y', which='major', labelsize=fig_ytick_fontsize)
    Te_dif_plot.plot(Te_dif[datetime_start:datetime_end].index, Te_dif['Te'][datetime_start:datetime_end],\
    color = 'green', linewidth=fig_line_width) # Difference data.
    Te_dif_plot.set_ylabel('Te_dif', fontsize=fig_ylabel_fontsize)
    Te_dif_plot.tick_params(axis='y', which='major', labelsize=fig_ytick_fontsize)
    # This is a shared axis for all subplot
    Te_dif_plot.tick_params(axis='x', which='major', labelsize=fig_xtick_fontsize)
else:
    # This is a shared axis for all subplot
    Tp_dif_plot.tick_params(axis='x', which='major', labelsize=fig_xtick_fontsize)
# Save plot.
fig.savefig('GS_filter_process_Np_Tp_Te('+str(datetime_start)+'~'+str(datetime_end)+').png',\
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

if 'Te' in locals():
    # Resample Te data into one minute resolution.
    # Linear fit first, to make sure there is no NaN. NaN will propagate by resample.
    # Interpolate according to timestamps. Cannot handle boundary.
    Te_DataFrame.interpolate(method='time', inplace=True, limit=n_interp_limit)
    # Process boundary.
    #Te_DataFrame.fillna(method='ffill', inplace=True) # Fill missing data, forward copy.
    #Te_DataFrame.fillna(method='bfill', inplace=True) # Fill missing data, backward copy.
    print('Resampling Te data into 1 minute resolution...')
    Te_DataFrame.drop_duplicates(keep='first', inplace=True) # Drop duplicated records, keep first one.
    # Resample to 1 minute resolution. New added records will be filled with NaN.
    # Resample to 30 second first, or data points will shift too much, as large as 1 min.
    Te_DataFrame = Te_DataFrame.resample('1T').mean()
    # Interpolate according to timestamps. Cannot handle boundary.
    Te_DataFrame.interpolate(method='time', inplace=True, limit=n_interp_limit)
    # Process boundary.
    #Te_DataFrame.fillna(method='ffill', inplace=True) # Fill missing data, forward copy.
    #Te_DataFrame.fillna(method='bfill', inplace=True) # Fill missing data, backward copy.
    # Now, downsample to 1 minute.
    # Te_DataFrame = Te_DataFrame.resample(rule='60S', base=30).mean()
    print('Te_DataFrame shape : {}'.format(Te_DataFrame.shape))
    print('The total records in Te : {}'.format(len(Te_DataFrame.index)))
    print('Max(Te) = {}, Min(Te) = {}'.format(Te_DataFrame.max(), Te_DataFrame.min()))
    print('Done.')


fig_line_width = 0.1
fig_ylabel_fontsize = 9
fig_xtick_fontsize = 8
fig_ytick_fontsize = 8
fig_legend_size = 5


'''

# Plot resampled BGSE.
print('Plotting resampled BGSE...')
fig,ax = plt.subplots(3,1, sharex=True,figsize=(18, 10))
ax[0].plot(BGSE_DataFrame0[datetime_start:datetime_end].index, BGSE_DataFrame0['Bx'][datetime_start:datetime_end],\
           color = 'red', linewidth=fig_line_width, label='Bx_original')
ax[0].plot(BGSE_DataFrame[datetime_start:datetime_end].index, BGSE_DataFrame['Bx'][datetime_start:datetime_end],\
           color = 'black', linewidth=fig_line_width, label='Bx_resampled')
ax[0].legend(loc='upper left',prop={'size':fig_legend_size})
ax[0].set_ylabel('Bx', fontsize=fig_ylabel_fontsize)
ax[1].plot(BGSE_DataFrame0[datetime_start:datetime_end].index, BGSE_DataFrame0['By'][datetime_start:datetime_end],\
           color = 'red', linewidth=fig_line_width, label='By_original')
ax[1].plot(BGSE_DataFrame[datetime_start:datetime_end].index, BGSE_DataFrame['By'][datetime_start:datetime_end],\
           color = 'black', linewidth=fig_line_width, label='By_resampled')
ax[1].legend(loc='upper left',prop={'size':fig_legend_size})
ax[1].set_ylabel('By', fontsize=fig_ylabel_fontsize)
ax[2].plot(BGSE_DataFrame0[datetime_start:datetime_end].index, BGSE_DataFrame0['Bz'][datetime_start:datetime_end],\
           color = 'red', linewidth=fig_line_width, label='Bz_original')
ax[2].plot(BGSE_DataFrame[datetime_start:datetime_end].index, BGSE_DataFrame['Bz'][datetime_start:datetime_end],\
           color = 'black', linewidth=fig_line_width, label='Bz_resampled')
ax[2].legend(loc='upper left',prop={'size':fig_legend_size})
ax[2].set_ylabel('Bz', fontsize=fig_ylabel_fontsize)
# This is a shared x axis.
ax[2].tick_params(axis='x', which='major', labelsize=fig_xtick_fontsize)
fig.savefig('GS_resampled_BGSE('+str(datetime_start)+'~'+str(datetime_end)+').png',\
            format='png', dpi=500, bbox_inches='tight')
print('Done.')

# Plot resampled VGSE.
print('Plotting resampled VGSE...')
fig,ax = plt.subplots(3,1, sharex=True,figsize=(18, 10))
ax[0].plot(VGSE_DataFrame0[datetime_start:datetime_end].index, VGSE_DataFrame0['Vx'][datetime_start:datetime_end],\
           color = 'red', linewidth=fig_line_width, label='Vx_original')
ax[0].plot(VGSE_DataFrame[datetime_start:datetime_end].index, VGSE_DataFrame['Vx'][datetime_start:datetime_end],\
           color = 'black', linewidth=fig_line_width, label='Vx_resampled')
ax[0].legend(loc='upper left',prop={'size':fig_legend_size})
ax[0].set_ylabel('Vx', fontsize=fig_ylabel_fontsize)
ax[1].plot(VGSE_DataFrame0[datetime_start:datetime_end].index, VGSE_DataFrame0['Vy'][datetime_start:datetime_end],\
           color = 'red', linewidth=fig_line_width, label='Vy_original')
ax[1].plot(VGSE_DataFrame[datetime_start:datetime_end].index, VGSE_DataFrame['Vy'][datetime_start:datetime_end],\
           color = 'black', linewidth=fig_line_width, label='Vy_resampled')
ax[1].legend(loc='upper left',prop={'size':fig_legend_size})
ax[1].set_ylabel('Vy', fontsize=fig_ylabel_fontsize)
ax[2].plot(VGSE_DataFrame0[datetime_start:datetime_end].index, VGSE_DataFrame0['Vz'][datetime_start:datetime_end],\
           color = 'red', linewidth=fig_line_width, label='Vz_original')
ax[2].plot(VGSE_DataFrame[datetime_start:datetime_end].index, VGSE_DataFrame['Vz'][datetime_start:datetime_end],\
           color = 'black', linewidth=fig_line_width, label='Vz_resampled')
ax[2].legend(loc='upper left',prop={'size':fig_legend_size})
ax[2].set_ylabel('Vz', fontsize=fig_ylabel_fontsize)
# This is a shared x axis.
ax[2].tick_params(axis='x', which='major', labelsize=fig_xtick_fontsize)
fig.savefig('GS_resampled_VGSE('+str(datetime_start)+'~'+str(datetime_end)+').png',\
            format='png', dpi=500, bbox_inches='tight')
print('Done.')

# Plot resampled Np, Tp, and Te.
print('Plotting resampled Np, Tp, and Te...')
line_width = 0.4
fig,ax = plt.subplots(3,1, sharex=True,figsize=(18, 10))
ax[0].plot(Np_DataFrame0[datetime_start:datetime_end].index, Np_DataFrame0['Np'][datetime_start:datetime_end], linestyle='solid', linewidth=fig_line_width, color='red', label='Np_original') #,marker='+', markersize = 5
ax[0].plot(Np_DataFrame[datetime_start:datetime_end].index, Np_DataFrame['Np'][datetime_start:datetime_end], linestyle='solid', linewidth=fig_line_width, color='black', label='Np_resampled') #,marker='x',markersize = 5
ax[0].set_ylabel('Np', fontsize=fig_ylabel_fontsize)
ax[0].legend(loc='upper left',prop={'size':fig_legend_size})
ax[1].plot(Tp_DataFrame0[datetime_start:datetime_end].index, Tp_DataFrame0['Tp'][datetime_start:datetime_end], linestyle='solid', linewidth=fig_line_width, color='red', label='Tp_original') #,marker='x',markersize = 5
ax[1].plot(Tp_DataFrame[datetime_start:datetime_end].index, Tp_DataFrame['Tp'][datetime_start:datetime_end], linestyle='solid', linewidth=fig_line_width, color='black', label='Tp_resampled') #,marker='x',markersize = 5
ax[1].set_ylabel('Tp', fontsize=fig_ylabel_fontsize)
ax[1].legend(loc='upper left',prop={'size':fig_legend_size})
if 'Te' in locals():
    ax[2].plot(Te_DataFrame0[datetime_start:datetime_end].index, Te_DataFrame0['Te'][datetime_start:datetime_end], linestyle='solid', linewidth=fig_line_width, color='red', label='Te_original') #,marker='x',markersize = 5
    ax[2].plot(Te_DataFrame[datetime_start:datetime_end].index, Te_DataFrame['Te'][datetime_start:datetime_end], linestyle='solid', linewidth=fig_line_width, color='black', label='Te_resampled') #,marker='x',markersize = 5
    ax[2].set_ylabel('Te', fontsize=fig_ylabel_fontsize)
    # This is a shared axis for all subplot
    ax[2].tick_params(axis='x', which='major', labelsize=fig_xtick_fontsize)
    ax[2].legend(loc='upper left',prop={'size':fig_legend_size})
else:
    # This is a shared axis for all subplot
    ax[1].tick_params(axis='x', which='major', labelsize=fig_xtick_fontsize)
fig.savefig('GS_resampled_Np_Tp_Te('+str(datetime_start)+'~'+str(datetime_end)+').png',\
            format='png', dpi=500, bbox_inches='tight')
print('Done.')

'''

# Merge all DataFrames into one according to time index.
# Calculate time range in minutes.
timeRangeInMinutes = int((datetimeEndAll - datetimeStartAll).total_seconds())//60
# Generate timestamp index.
index_datetime = np.asarray([datetimeStartAll + timedelta(minutes=x) for x in range(0, timeRangeInMinutes)])
# Generate empty DataFrame according using index_datetime as index.
GS_AllData_DataFrame = pd.DataFrame(index=index_datetime)
# Merge all DataFrames.
if 'Te' in locals():
    GS_AllData_DataFrame = pd.concat([GS_AllData_DataFrame, \
    BGSE_DataFrame, VGSE_DataFrame, Np_DataFrame, Tp_DataFrame, Te_DataFrame], axis=1)
else:
    GS_AllData_DataFrame = pd.concat([GS_AllData_DataFrame, \
    BGSE_DataFrame, VGSE_DataFrame, Np_DataFrame, Tp_DataFrame], axis=1)
# Save merged DataFrame into pickle file.
GS_AllData_DataFrame.to_pickle(rootDir + 'GS_'+str(year)+'_AllData_DataFrame_preprocessed.p')
print('Checking the number of NaNs in GS_AllData_DataFrame...')
len_GS_AllData_DataFrame = len(GS_AllData_DataFrame)
for key in GS_AllData_DataFrame.keys():
    num_notNaN = GS_AllData_DataFrame[key].isnull().values.sum()
    percent_notNaN = 100.0 - num_notNaN * 100.0 / len_GS_AllData_DataFrame
    print('The number of NaNs in {} is {}, integrity is {}%'.format(key, num_notNaN, round(percent_notNaN, 2)))
#print(GS_AllData_DataFrame)

# Save original data into dataframe.
# Interpolate original data, remove NaN.
BGSE_DataFrame0.interpolate(method='time', inplace=True, limit=5)
VGSE_DataFrame0.interpolate(method='time', inplace=True, limit=5)
Np_DataFrame0.interpolate(method='time', inplace=True, limit=5)
Tp_DataFrame0.interpolate(method='time', inplace=True, limit=5)
if 'Te' in locals():
    Te_DataFrame0.interpolate(method='time', inplace=True, limit=5)
# Resample original data to 1T.
BGSE_DataFrame0 = BGSE_DataFrame0.resample('1T').mean()
VGSE_DataFrame0 = VGSE_DataFrame0.resample('1T').mean()
Np_DataFrame0 = Np_DataFrame0.resample('1T').mean()
Tp_DataFrame0 = Tp_DataFrame0.resample('1T').mean()
if 'Te' in locals():
    Te_DataFrame0 = Te_DataFrame0.resample('1T').mean()
# Reindex, or will raise ValueError: Shape of passed values is (), indices imply ().
BGSE_DataFrame0.reindex(index_datetime)
VGSE_DataFrame0.reindex(index_datetime)
Np_DataFrame0.reindex(index_datetime)
Tp_DataFrame0.reindex(index_datetime)
if 'Te' in locals():
    Te_DataFrame0.reindex(index_datetime)
# Merge all original DataFrames.
if 'Te' in locals():
    GS_AllData_DataFrame_original = pd.concat([BGSE_DataFrame0, VGSE_DataFrame0, Np_DataFrame0, Tp_DataFrame0, Te_DataFrame0], axis=1)
else:
    GS_AllData_DataFrame_original = pd.concat([BGSE_DataFrame0, VGSE_DataFrame0, Np_DataFrame0, Tp_DataFrame0], axis=1)
# Interpolate 1T original data, remove NaN.
GS_AllData_DataFrame_original.interpolate(method='time', inplace=True, limit=5)
print('Checking the number of NaNs in GS_AllData_DataFrame_original...')
len_GS_AllData_DataFrame_original = len(GS_AllData_DataFrame_original)
for key in GS_AllData_DataFrame_original.keys():
    num_notNaN = GS_AllData_DataFrame_original[key].isnull().values.sum()
    percent_notNaN = 100.0 - num_notNaN * 100.0 / len_GS_AllData_DataFrame_original
    print('The number of NaNs in {} is {}, integrity is {}%'.format(key, num_notNaN, round(percent_notNaN, 2)))
# Save merged DataFrame into pickle file.
GS_AllData_DataFrame_original.to_pickle(rootDir + 'GS_'+str(year)+'_AllData_DataFrame_original.p')
#print(GS_AllData_DataFrame_original)

'''
# Save seperated original DataFrames.
VGSE_DataFrame0.to_pickle('./GS_'+str(year)+'_VGSE_DataFrame.p')
BGSE_DataFrame0.to_pickle('./GS_'+str(year)+'_BGSE_DataFrame.p')
Np_DataFrame0.to_pickle('./GS_'+str(year)+'_Np_DataFrame.p')
Tp_DataFrame0.to_pickle('./GS_'+str(year)+'_Tp_DataFrame.p')
if 'Te' in locals():
    Te_DataFrame0.to_pickle('./GS_'+str(year)+'_Te_DataFrame.p')
'''






