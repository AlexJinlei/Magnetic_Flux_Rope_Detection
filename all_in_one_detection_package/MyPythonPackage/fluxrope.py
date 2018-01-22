from __future__ import division # Treat integer as float.
import os
import sys
import numpy as np # Scientific calculation package.
from numpy import linalg as la
import math
#from aenum import Enum # Enum data type
from ai import cdas # Import CDAweb API package.
import astropy
from spacepy import pycdf
from scipy import integrate
from scipy import stats
from scipy.signal import savgol_filter # Savitzky-Golay filter
import scipy as sp
import time
from datetime import datetime # Import datetime class from datetime package.
from datetime import timedelta
import pandas as pd
import pickle
import multiprocessing
from collections import OrderedDict

import matplotlib
import matplotlib.pyplot as plt # Plot package, make matlab style plots.
from matplotlib.ticker import AutoMinorLocator
from matplotlib import dates
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import FuncFormatter
from matplotlib.lines import Line2D
import matplotlib.dates as mdates
import matplotlib.cbook as cbook
from PIL import Image

############################################## User defined functions ##############################################

# Global variables.

# Physics constants.
mu0 = 4.0 * np.pi * 1e-7 #(N/A^2) magnetic constant permeability of free space vacuum permeability
m_proton = 1.6726219e-27 # Proton mass. In kg.
factor_deg2rad = np.pi/180.0 # Convert degree to rad.


# Download data from ACE or WIND.
def download_data(source, data_cache_dir, datetimeStart, datetimeEnd):
    # Need packages:
    # from ai import cdas # Import CDAweb API package.
    # from datetime import datetime # Import datetime class from datetime package.

    # If turn cache on, do not download from one dataset more than one time. There is a bug to casue error.
    # Make sure download every variables you need from one dataset at once.
    cdas.set_cache(True, data_cache_dir)
    
    if source == 'ACE':
        print('Downloading data from ACE spacecraft...')
        print('The common data coverage of AC_H0_MFI, AC_H0_SWE, and AC_H3_EPM is:')
        print('1998/02/04 00:00:31 - 2016/12/24 23:59:56')
        
        if(datetimeStart<datetime(1998,2,4,0,0,31) or datetimeEnd>datetime(2016,12,24,23,59,56)):
            print('Specified time range is out of available range!')
            return None

        # Download magnetic field data from ACE.
        # The dimension of [WI_H0_MFI['BGSE'] is N X 3(N row, 3 column)
        # [Available Time Range: 1997/09/02 00:00:12 - 2016/12/24 23:59:56]
        print('Downloading AC_H0_MFI...')
        # AC_H0_MFI [Available Time Range: 1997/09/02 00:00:12 - 2016/12/24 23:59:56]
        try:
            AC_H0_MFI = cdas.get_data('istp_public', 'AC_H0_MFI', datetimeStart, datetimeEnd, ['BGSEc'], cdf=True)
        except cdas.NoDataError:
            print('No AC_H0_MFI data!')
            AC_H0_MFI = None
        print('Done.')

        # Download solar wind data Np, V_GSE, and Thermal speed.
        # [Available Time Range: 1998/02/04 00:00:31 - 2017/01/20 23:59:42]
        print('Downloading AC_H0_SWE...')
        # Np: Solar Wind Proton Number Density, scalar.
        # V_GSE: Solar Wind Velocity in GSE coord., 3 components.
        # Tpr: radial component of the proton temperature. The radial component of the proton temperature is the (1,1) component of the temperature tensor, along the radial direction. It is obtained by integration of the ion (proton) distribution function.
        # Alpha to proton density ratio.
        try:
            AC_H0_SWE = cdas.get_data('istp_public', 'AC_H0_SWE', datetimeStart, datetimeEnd, ['Np','V_GSE','Tpr','alpha_ratio'], cdf=True)
        except cdas.NoDataError:
            print('No AC_H0_SWE data!')
            AC_H0_SWE = None
        print('Done.')

        # Download EPAM data.
        # [Available Time Range: 1997/08/30 17:01:00 - 2016/12/30 23:53:00]
        print('Downloading AC_H3_EPM...')
        try:
            AC_H3_EPM = cdas.get_data('istp_public', 'AC_H3_EPM', datetimeStart, datetimeEnd, ['P1p','P2p','P3p','P4p','P5p','P6p','P7p','P8p','P1','P2','P3','P4','P5','P6','P7','P8'], cdf=True)
        except cdas.NoDataError:
            print('No AC_H3_EPM data!')
            AC_H3_EPM = None
        print('Done.')
        
        # Construct return value.
        data_dict = {'ID':'ACE', 'timeRange':{'datetimeStart':datetimeStart, 'datetimeEnd':datetimeEnd}, 'AC_H0_MFI':AC_H0_MFI, 'AC_H0_SWE':AC_H0_SWE, 'AC_H3_EPM':AC_H3_EPM}
    
    elif source == 'WIND':
        print('Downloading data from WIND spacecraft...')
        print('The common data coverage of WI_H0_MFI, WI_K0_SWE, and WI_H0(5)_SWE is:')
        print('1994/12/29 00:00:02 - 2017/03/15 23:59:32')
        
        if(datetimeStart<datetime(1994,12,29,0,0,2) or datetimeEnd>datetime(2017,3,15,23,59,32)):
            print('Specified time range is out of available range!')
            return None
        
        # Download magnetic field data.
        # The dimension of [WI_H0_MFI['BGSE'] is N X 3(N row, 3 column)
        # [Available Time Range: 1994/11/12 00:00:30 - 2017/04/12 23:59:30]
        print('Downloading WI_H0_MFI...')
        try:
            WI_H0_MFI = cdas.get_data('istp_public', 'WI_H0_MFI', datetimeStart, datetimeEnd, ['BGSE'], cdf=True)
        except cdas.NoDataError:
            print('No WI_H0_MFI data!')
            WI_H0_MFI = None
        print('Done.')

        # Download solar wind data.
        # [Available Time Range: 1994/11/01 12:00:00 - 2017/04/18 23:58:58]
        print('Downloading WI_K0_SWE...')
        try:
            WI_K0_SWE = cdas.get_data('istp_public', 'WI_K0_SWE', datetimeStart, datetimeEnd, ['Np','V_GSE','THERMAL_SPD'], cdf=True)
        except cdas.NoDataError:
            print('No WI_K0_SWE data!')
            WI_K0_SWE = None
        print('Done.')

        # Download electron temprature data. Unit is Kelvin.
        # WI_H0_SWE time span = [1994/12/29 00:00:02, 2001/05/31 23:59:57]
        # WI_H5_SWE time span = [2002/08/16 00:00:05, 2017/03/15 23:59:32]
        if (datetimeStart>=datetime(1994,12,29,0,0,2) and datetimeEnd<=datetime(2001,5,31,23,59,57)):
            print('Downloading WI_H0_SWE...')
            try:
                WI_H0_SWE = cdas.get_data('istp_public', 'WI_H0_SWE', datetimeStart, datetimeEnd, ['Te'], cdf=True)
            except cdas.NoDataError:
                print('No electron data available!')
        elif (datetimeStart>=datetime(2002,8,16,0,0,5) and datetimeEnd<=datetime(2017,3,15,23,59,32)):
            print('Downloading WI_H5_SWE...')
            try:
                WI_H5_SWE = cdas.get_data('istp_public', 'WI_H5_SWE', datetimeStart, datetimeEnd, ['T_elec'], cdf=True)
            except cdas.NoDataError:
                print('No electron data available!')

        else:
            print('No electron data available!')

        # Download Alpha number density(Na (n/cc) from non-linear analysis).
        # [Available Time Range: 1994/11/17 19:50:45 - 2017/04/13 23:58:36]
        print('Downloading WI_H1_SWE...')
        try:
            WI_H1_SWE = cdas.get_data('istp_public', 'WI_H1_SWE', datetimeStart, datetimeEnd, ['Alpha_Na_nonlin'], cdf=True)
        except cdas.NoDataError:
            print('No WI_H1_SWE data!')
            WI_H1_SWE = None
        print('Done.')

        # Construct return value.
        if 'WI_H0_SWE' in locals():
            data_dict = {'ID':'WIND', 'timeRange':{'datetimeStart':datetimeStart, 'datetimeEnd':datetimeEnd}, 'WI_H0_MFI':WI_H0_MFI, 'WI_K0_SWE':WI_K0_SWE, 'WI_H1_SWE':WI_H1_SWE, 'WI_H0_SWE':WI_H0_SWE}
        elif 'WI_H5_SWE' in locals():
            data_dict = {'ID':'WIND', 'timeRange':{'datetimeStart':datetimeStart, 'datetimeEnd':datetimeEnd}, 'WI_H0_MFI':WI_H0_MFI, 'WI_K0_SWE':WI_K0_SWE, 'WI_H1_SWE':WI_H1_SWE, 'WI_H5_SWE':WI_H5_SWE}
        else:
            data_dict = {'ID':'WIND', 'timeRange':{'datetimeStart':datetimeStart, 'datetimeEnd':datetimeEnd}, 'WI_H0_MFI':WI_H0_MFI, 'WI_K0_SWE':WI_K0_SWE, 'WI_H1_SWE':WI_H1_SWE}

    else:
        print('Please specify the correct spacecraft ID, \'WIND\' or \'ACE\'!')
        data_dict = None

    return data_dict


############################################## Preprocess Data ###########################################

def preprocess_data(data_dict, data_pickle_dir, **kwargs):
    # All temperature are converted to Kelvin.

    # Extract data from dict variable.
    isPlotFilterProcess = False
    isVerbose = False
    isCheckDataIntegrity = False
    isSaveOriginalData = False
    
    # Physics constants.
    mu0 = 4.0 * np.pi * 1e-7 #(N/A^2) magnetic constant permeability of free space vacuum permeability
    m_proton = 1.6726219e-27 # Proton mass. In kg.
    factor_deg2rad = np.pi/180.0 # Convert degree to rad.
    k_Boltzmann = 1.3806488e-23 # Boltzmann constant, in J/K.
    
    if 'isPlotFilterProcess' in kwargs:
        if ((kwargs['isPlotFilterProcess']==True)or(kwargs['isPlotFilterProcess']==False)):
            isPlotFilterProcess = kwargs['isPlotFilterProcess']
    
    timeStart = data_dict['timeRange']['datetimeStart']
    timeEnd = data_dict['timeRange']['datetimeEnd']
    
    # Truncate datetime, remove seconds and miliseconds. Or will be misaligned when resample.
    timeStart = timeStart.replace(second=0, microsecond=0)
    timeEnd = timeEnd.replace(second=0, microsecond=0)
    
    timeStart_str = timeStart.strftime('%Y%m%d%H%M')
    timeEnd_str = timeEnd.strftime('%Y%m%d%H%M')
    
    if data_dict['ID']=='WIND':
        print('\nSpacecraft ID: WIND')
        
        print('Extracting BGSE_Epoch...')
        if data_dict['WI_H0_MFI'] is not None:
            BGSE_Epoch = data_dict['WI_H0_MFI']['Epoch']
        else:
            BGSE_Epoch = None
        
        print('Extracting BGSE...')
        if data_dict['WI_H0_MFI'] is not None:
            BGSE = data_dict['WI_H0_MFI']['BGSE']
        else:
            BGSE = None
        
        # WI_H0_SWE time span = [1994/12/29 00:00:02, 2001/05/31 23:59:57]
        if 'WI_H0_SWE' in data_dict:
            print('Extracting Te_Epoch...')
            Te_Epoch = data_dict['WI_H0_SWE']['Epoch']
            print('Extracting Te...')
            Te = data_dict['WI_H0_SWE']['Te']
        # WI_H5_SWE time span = [2002/08/16 00:00:05, 2017/03/15 23:59:32]
        elif 'WI_H5_SWE' in data_dict:
            print('Extracting Te_Epoch...')
            Te_Epoch = data_dict['WI_H5_SWE']['Epoch']
            print('Extracting Te...')
            Te = data_dict['WI_H5_SWE']['T_elec']
        else:
            print('No electron data.')

        print('Extracting SW_Epoch...')
        if data_dict['WI_K0_SWE'] is not None:
            SW_Epoch = data_dict['WI_K0_SWE']['Epoch'] # Solar wind time stamps.
        else:
            SW_Epoch = None
            
        print('Extracting VGSE...')
        if data_dict['WI_K0_SWE'] is not None:
            VGSE = data_dict['WI_K0_SWE']['V_GSE'] # Solar wind speed in GSE coordinate.
        else:
            VGSE = None
            
        print('Extracting Np...')
        if data_dict['WI_K0_SWE'] is not None:
            Np = data_dict['WI_K0_SWE']['Np'] # Proton number density.
        else:
            Np = None
        
        print('Extracting Tp...')
        if data_dict['WI_K0_SWE'] is not None:
            Tp = data_dict['WI_K0_SWE']['THERMAL_SPD'] # Proton thermal speed.
        else:
            Tp = None

        print('Extracting N_alpha_Epoch...')
        if data_dict['WI_H1_SWE'] is not None:
            N_alpha_Epoch = data_dict['WI_H1_SWE']['Epoch'] # N_alpha time stamps.
        else:
            N_alpha_Epoch = None
        
        print('Extracting N_alpha...') # From WI_H1_SWE.
        if data_dict['WI_H1_SWE'] is not None:
            N_alpha = data_dict['WI_H1_SWE']['Alpha_Na_nonlin'] # Alpha number density Na (n/cc) from non-linear analysis.
        else:
            N_alpha = None

        # Process missing value. missing value = -9.9999998e+30.
        if BGSE is not None:
            BGSE[abs(BGSE) > 80] = np.nan # B field.
        if Np is not None:
            Np[Np < -1e+10] = np.nan # Proton number density.
        if VGSE is not None:
            VGSE[abs(VGSE) > 1500] = np.nan # Solar wind speed.
        if Tp is not None:
            Tp[Tp < -1e+10] = np.nan # Proton temperature.
        if N_alpha is not None:
            N_alpha[N_alpha < -1e+10] = np.nan # Alpha number density.
            N_alpha[N_alpha > 1e+3] = np.nan # Alpha number density.
        if 'Te' in locals(): # If Elec_Te_all is defined.
            Te[Te < -1e+10] = np.nan # Electron temperature.

        # Put data into DataFrame.
        print('Putting Data into DataFrame...')
        if BGSE_Epoch is None: 
            return None
        else:
            BGSE_DataFrame = pd.DataFrame(BGSE, index = BGSE_Epoch, columns = ['Bx', 'By', 'Bz'])
            
        if SW_Epoch is not None:
            VGSE_DataFrame = pd.DataFrame(VGSE, index = SW_Epoch, columns = ['Vx', 'Vy', 'Vz'])
            Np_DataFrame = pd.DataFrame(Np, index = SW_Epoch, columns = ['Np'])
            Tp_DataFrame = pd.DataFrame(Tp, index = SW_Epoch, columns = ['Tp'])
        else:
            VGSE_DataFrame = pd.DataFrame(None, columns = ['Vx', 'Vy', 'Vz'])
            Np_DataFrame = pd.DataFrame(None, columns = ['Np'])
            Tp_DataFrame = pd.DataFrame(None, columns = ['Tp'])
        
        if N_alpha_Epoch is None:
            N_alpha_DataFrame = pd.DataFrame(N_alpha, index = N_alpha_Epoch, columns = ['N_alpha'])
        else:
            N_alpha_DataFrame = pd.DataFrame(None, columns = ['N_alpha'])
        
        if 'Te' in locals(): # If Elec_Te_all is defined.
            Te_DataFrame = pd.DataFrame(Te, index = Te_Epoch, columns = ['Te'])
            
        # Trim data. Some times cdas API will download wrong time range.
        if BGSE_Epoch is not None:
            BGSE_DataFrame = BGSE_DataFrame[(BGSE_DataFrame.index>=timeStart)&(BGSE_DataFrame.index<=timeEnd)]
        if SW_Epoch is not None:
            VGSE_DataFrame = VGSE_DataFrame[(VGSE_DataFrame.index>=timeStart)&(VGSE_DataFrame.index<=timeEnd)]
            Np_DataFrame = Np_DataFrame[(Np_DataFrame.index>=timeStart)&(Np_DataFrame.index<=timeEnd)]
            Tp_DataFrame = Tp_DataFrame[(Tp_DataFrame.index>=timeStart)&(Tp_DataFrame.index<=timeEnd)]
        if N_alpha_Epoch is not None:
            N_alpha_DataFrame = N_alpha_DataFrame[(N_alpha_DataFrame.index>=timeStart)&(N_alpha_DataFrame.index<=timeEnd)]
        if 'Te' in locals():
            Te_DataFrame = Te_DataFrame[(Te_DataFrame.index>=timeStart)&(Te_DataFrame.index<=timeEnd)]

        # Drop duplicated records. This is the flaw of the source data.
        print('Dropping duplicated records...')
        VGSE_DataFrame.drop_duplicates(keep='first', inplace=True) # Drop duplicated records, keep first one.
        BGSE_DataFrame.drop_duplicates(keep='first', inplace=True) # Drop duplicated records, keep first one.
        Np_DataFrame.drop_duplicates(keep='first', inplace=True) # Drop duplicated records, keep first one.
        Tp_DataFrame.drop_duplicates(keep='first', inplace=True) # Drop duplicated records, keep first one.
        N_alpha_DataFrame.drop_duplicates(keep='first', inplace=True) # Drop duplicated records, keep first one.
        if 'Te' in locals(): # If Elec_Te_all is defined.
            Te_DataFrame.drop_duplicates(keep='first', inplace=True) # Drop duplicated records, keep first one.

        # Sort data by time index. Time series data may be not in order, This is the flaw of the source data.
        print('Sorting data...')
        VGSE_DataFrame.sort_index(axis=0, ascending=True, inplace=True, kind='quicksort')
        BGSE_DataFrame.sort_index(axis=0, ascending=True, inplace=True, kind='quicksort')
        Np_DataFrame.sort_index(axis=0, ascending=True, inplace=True, kind='quicksort')
        Tp_DataFrame.sort_index(axis=0, ascending=True, inplace=True, kind='quicksort')
        N_alpha_DataFrame.sort_index(axis=0, ascending=True, inplace=True, kind='quicksort')
        if 'Te' in locals(): # If Elec_Te_all is defined.
            Te_DataFrame.sort_index(axis=0, ascending=True, inplace=True, kind='quicksort')
        
        #========================================= Process BGSE missing value =========================================
        print('\nProcessing BGSE...')
        # Keep original data.
        BGSE_DataFrame0 = BGSE_DataFrame.copy(deep=True)
        #print('BGSE_DataFrame.shape = {}'.format(BGSE_DataFrame.shape))

        n_removed_Bx_total = 0
        n_removed_By_total = 0
        n_removed_Bz_total = 0

        # Apply Butterworth filter.
        for Wn in [0.45]:
            print('Applying Butterworth filter with cutoff frequency = {}, remove spikes...'.format(Wn))
            
            # Note that, sp.signal.butter cannot handle np.nan. Fill the nan before using it.
            BGSE_DataFrame.fillna(method='ffill', inplace=True) # Fill missing data, forward copy.
            BGSE_DataFrame.fillna(method='bfill', inplace=True) # Fill missing data, backward copy.
            # Create an empty DataFrame to store the filtered data.
            BGSE_LowPass = pd.DataFrame(index = BGSE_DataFrame.index, columns = ['Bx', 'By', 'Bz'])
            # Design the Buterworth filter.
            N  = 2    # Filter order
            B, A = sp.signal.butter(N, Wn, output='ba')
            # Apply the filter.
            try:
                BGSE_LowPass['Bx'] = sp.signal.filtfilt(B, A, BGSE_DataFrame['Bx'])
            except:
                print('Encounter exception, skip sp.signal.filtfilt operation!')
                BGSE_LowPass['Bx'] = BGSE_DataFrame['Bx'].copy()
            
            try:
                BGSE_LowPass['By'] = sp.signal.filtfilt(B, A, BGSE_DataFrame['By'])
            except:
                print('Encounter exception, skip sp.signal.filtfilt operation!')
                BGSE_LowPass['By'] = BGSE_DataFrame['By'].copy()
            
            try:
                BGSE_LowPass['Bz'] = sp.signal.filtfilt(B, A, BGSE_DataFrame['Bz'])
            except:
                print('Encounter exception, skip sp.signal.filtfilt operation!')
                BGSE_LowPass['Bz'] = BGSE_DataFrame['Bz'].copy()
                
            # Calculate the difference between BGSE_LowPass and BGSE_DataFrame.
            BGSE_dif = pd.DataFrame(index = BGSE_DataFrame.index, columns = ['Bx', 'By', 'Bz']) # Generate empty DataFrame.
            BGSE_dif['Bx'] = BGSE_DataFrame['Bx'] - BGSE_LowPass['Bx']
            BGSE_dif['By'] = BGSE_DataFrame['By'] - BGSE_LowPass['By']
            BGSE_dif['Bz'] = BGSE_DataFrame['Bz'] - BGSE_LowPass['Bz']
            # Calculate the mean and standard deviation of BGSE_dif.
            Bx_dif_std, By_dif_std, Bz_dif_std = BGSE_dif.std(skipna=True, numeric_only=True)
            Bx_dif_mean, By_dif_mean, Bz_dif_mean = BGSE_dif.mean(skipna=True, numeric_only=True)
            # Set the values fall outside n*std to np.nan.
            n_dif_std = 3.89 # 99.99%
            Bx_remove = (BGSE_dif['Bx']<(Bx_dif_mean-n_dif_std*Bx_dif_std))|(BGSE_dif['Bx']>(Bx_dif_mean+n_dif_std*Bx_dif_std))
            By_remove = (BGSE_dif['By']<(By_dif_mean-n_dif_std*By_dif_std))|(BGSE_dif['By']>(By_dif_mean+n_dif_std*By_dif_std))
            Bz_remove = (BGSE_dif['Bz']<(Bz_dif_mean-n_dif_std*Bz_dif_std))|(BGSE_dif['Bz']>(Bz_dif_mean+n_dif_std*Bz_dif_std))
            BGSE_DataFrame['Bx'][Bx_remove] = np.nan
            BGSE_DataFrame['By'][By_remove] = np.nan
            BGSE_DataFrame['Bz'][Bz_remove] = np.nan
            
            Bx_dif_lower_boundary = Bx_dif_mean-n_dif_std*Bx_dif_std
            Bx_dif_upper_boundary = Bx_dif_mean+n_dif_std*Bx_dif_std
            By_dif_lower_boundary = By_dif_mean-n_dif_std*By_dif_std
            By_dif_upper_boundary = By_dif_mean+n_dif_std*By_dif_std
            Bz_dif_lower_boundary = Bz_dif_mean-n_dif_std*Bz_dif_std
            Bz_dif_upper_boundary = Bz_dif_mean+n_dif_std*Bz_dif_std
            
            n_removed_Bx = sum(Bx_remove)
            n_removed_By = sum(By_remove)
            n_removed_Bz = sum(Bz_remove)
            n_removed_Bx_total += n_removed_Bx
            n_removed_By_total += n_removed_By
            n_removed_Bz_total += n_removed_Bz
            
            if isVerbose:
                print('B_dif_std:', Bx_dif_std, By_dif_std, Bz_dif_std)
                print('B_dif_mean:', Bx_dif_mean, By_dif_mean, Bz_dif_mean)
                print('The BGSE Bx_dif value range within {} std is [{}, {}]'.format(n_dif_std, Bx_dif_lower_boundary, Bx_dif_upper_boundary))
                print('The BGSE By_dif value range within {} std is [{}, {}]'.format(n_dif_std, By_dif_lower_boundary, By_dif_upper_boundary))
                print('The BGSE Bz_dif value range within {} std is [{}, {}]'.format(n_dif_std, Bz_dif_lower_boundary, Bz_dif_upper_boundary))
                print('In Bx, this operation removed {} records!'.format(n_removed_Bx))
                print('In By, this operation removed {} records!'.format(n_removed_By))
                print('In Bz, this operation removed {} records!!'.format(n_removed_Bz))
                print('Till now, in Bx, {} records have been removed!'.format(n_removed_Bx_total))
                print('Till now, in By, {} records have been removed!'.format(n_removed_By_total))
                print('Till now, in Bz, {} records have been removed!'.format(n_removed_Bz_total))
                print('\n')
        
        # If plot filter process or not.
        if isPlotFilterProcess:
            # Plot BGSE filter process.
            print('Plotting BGSE filtering process...')
            fig_line_width = 0.5
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
            Bx_plot.plot(BGSE_DataFrame0.index, BGSE_DataFrame0['Bx'].fillna(0),\
                         color = 'red', linewidth=fig_line_width, label='Bx_original') # Original data.
            Bx_plot.plot(BGSE_DataFrame.index, BGSE_DataFrame['Bx'].fillna(0),\
                         color = 'blue', linewidth=fig_line_width, label='Bx_processed') # Filtered data.
            Bx_plot.plot(BGSE_LowPass.index, BGSE_LowPass['Bx'].fillna(0),\
                         color = 'black', linewidth=fig_line_width, label='Bx_LowPass') # Low pass curve.
            Bx_plot.set_ylabel('Bx', fontsize=fig_ylabel_fontsize)
            Bx_plot.legend(loc='upper left',prop={'size':fig_legend_size})
            Bx_plot.tick_params(axis='y', which='major', labelsize=fig_ytick_fontsize)
            Bx_dif.plot(BGSE_dif.index, BGSE_dif['Bx'].fillna(0),\
                        color = 'green', linewidth=fig_line_width) # Difference data.
            Bx_dif.tick_params(axis='y', which='major', labelsize=fig_ytick_fontsize)
            Bx_dif.set_ylabel('Bx_dif', fontsize=fig_ylabel_fontsize)
            # Plotting By filter process.
            By_plot.plot(BGSE_DataFrame0.index, BGSE_DataFrame0['By'].fillna(0),\
                         color = 'red', linewidth=fig_line_width, label='By_original') # Original data.
            By_plot.plot(BGSE_DataFrame.index, BGSE_DataFrame['By'].fillna(0),\
                         color = 'blue', linewidth=fig_line_width, label='By_processed') # Filtered data.
            By_plot.plot(BGSE_LowPass.index, BGSE_LowPass['By'].fillna(0),\
                         color = 'black', linewidth=fig_line_width, label='By_LowPass') # Low pass curve.
            By_plot.set_ylabel('By', fontsize=fig_ylabel_fontsize)
            By_plot.tick_params(axis='y', which='major', labelsize=fig_ytick_fontsize)
            By_plot.legend(loc='upper left',prop={'size':fig_legend_size})
            By_dif.plot(BGSE_dif.index, BGSE_dif['By'].fillna(0),\
                        color = 'green', linewidth=fig_line_width) # Difference data.
            By_dif.tick_params(axis='y', which='major', labelsize=fig_ytick_fontsize)
            By_dif.set_ylabel('By_dif', fontsize=fig_ylabel_fontsize)
            # Plotting Bz filter process.
            Bz_plot.plot(BGSE_DataFrame0.index, BGSE_DataFrame0['Bz'].fillna(0),\
                         color = 'red', linewidth=fig_line_width, label='Bz_original') # Original data.
            Bz_plot.plot(BGSE_DataFrame.index, BGSE_DataFrame['Bz'].fillna(0),\
                         color = 'blue', linewidth=fig_line_width, label='Bz_processed') # Filtered data.
            Bz_plot.plot(BGSE_LowPass.index, BGSE_LowPass['Bz'].fillna(0),\
                         color = 'black', linewidth=fig_line_width, label='Bz_LowPass') # Low pass curve.
            Bz_plot.set_ylabel('Bz', fontsize=fig_ylabel_fontsize)
            Bz_plot.tick_params(axis='y', which='major', labelsize=fig_ytick_fontsize)
            Bz_plot.legend(loc='upper left',prop={'size':fig_legend_size})
            Bz_dif.plot(BGSE_dif.index, BGSE_dif['Bz'].fillna(0),\
                        color = 'green', linewidth=fig_line_width) # Difference data.
            Bz_dif.tick_params(axis='y', which='major', labelsize=fig_ytick_fontsize)
            Bz_dif.set_ylabel('Bz_dif', fontsize=fig_ylabel_fontsize)
            # This is a shared axis for all subplot
            Bz_dif.tick_params(axis='x', which='major', labelsize=fig_xtick_fontsize)
            # Save plot.
            fig.savefig(data_pickle_dir + '/ACE_filter_process_BGSE_' + timeStart_str + '_' + timeEnd_str + '.eps',format='eps', dpi=500, bbox_inches='tight')

        #========================================= Process VGSE =========================================
        if not VGSE_DataFrame.empty:
            print('\nProcessing VGSE...')
            # Keep original data.
            VGSE_DataFrame0 = VGSE_DataFrame.copy(deep=True)

            # Remove all data which fall outside three standard deviations.
            n_removed_Vx_total = 0
            n_removed_Vy_total = 0
            n_removed_Vz_total = 0

            n_std = 3.89 # 99.99%.
            Vx_std, Vy_std, Vz_std = VGSE_DataFrame.std(skipna=True, numeric_only=True)
            Vx_mean, Vy_mean, Vz_mean = VGSE_DataFrame.mean(skipna=True, numeric_only=True)
            Vx_remove = (VGSE_DataFrame['Vx']<(Vx_mean-n_std*Vx_std))|(VGSE_DataFrame['Vx']>(Vx_mean+n_std*Vx_std))
            Vy_remove = (VGSE_DataFrame['Vy']<(Vy_mean-n_std*Vy_std))|(VGSE_DataFrame['Vy']>(Vy_mean+n_std*Vy_std))
            Vz_remove = (VGSE_DataFrame['Vz']<(Vz_mean-n_std*Vz_std))|(VGSE_DataFrame['Vz']>(Vz_mean+n_std*Vz_std))
            VGSE_DataFrame['Vx'][Vx_remove] = np.nan
            VGSE_DataFrame['Vy'][Vy_remove] = np.nan
            VGSE_DataFrame['Vz'][Vz_remove] = np.nan

            Vx_lower_boundary = Vx_mean-n_std*Vx_std
            Vx_upper_boundary = Vx_mean+n_std*Vx_std
            Vy_lower_boundary = Vy_mean-n_std*Vy_std
            Vy_upper_boundary = Vy_mean+n_std*Vy_std
            Vz_lower_boundary = Vz_mean-n_std*Vz_std
            Vz_upper_boundary = Vz_mean+n_std*Vz_std

            n_removed_Vx = sum(Vx_remove)
            n_removed_Vy = sum(Vy_remove)
            n_removed_Vz = sum(Vz_remove)
            n_removed_Vx_total += n_removed_Vx
            n_removed_Vy_total += n_removed_Vy
            n_removed_Vz_total += n_removed_Vz

            if isVerbose:
                print('\nRemove all VGSE data which fall outside three standard deviations...')
                print('V_std:', Vx_std, Vy_std, Vz_std)
                print('V_mean:', Vx_mean, Vy_mean, Vz_mean)
                print('The VGSE Vx value range within {} std is [{}, {}]'.format(n_std, Vx_lower_boundary, Vx_upper_boundary))
                print('The VGSE Vy value range within {} std is [{}, {}]'.format(n_std, Vy_lower_boundary, Vy_upper_boundary))
                print('The VGSE Vz value range within {} std is [{}, {}]'.format(n_std, Vz_lower_boundary, Vz_upper_boundary))
                print('In Vx, {} data has been removed!'.format(n_removed_Vx))
                print('In Vy, {} data has been removed!'.format(n_removed_Vy))
                print('In Vz, {} data has been removed!'.format(n_removed_Vz))
                print('Until now, in Vx, {} records have been removed!'.format(n_removed_Vx_total))
                print('Until now, in Vy, {} records have been removed!'.format(n_removed_Vy_total))
                print('Until now, in Vz, {} records have been removed!'.format(n_removed_Vz_total))
                print('\n')

            # Apply Butterworth filter two times.
            for Wn in [0.005, 0.05]:
                if Wn==0.005:
                    print('Applying Butterworth filter with cutoff frequency = 0.005, remove large outliers...')
                if Wn==0.05:
                    print('Applying Butterworth filter with cutoff frequency = 0.05, remove spikes...')
                # Note that, sp.signal.butter cannot handle np.nan. Fill the nan before using it.
                VGSE_DataFrame.fillna(method='ffill', inplace=True) # Fill missing data, forward copy.
                VGSE_DataFrame.fillna(method='bfill', inplace=True) # Fill missing data, backward copy.
                # Create an empty DataFrame to store the filtered data.
                VGSE_LowPass = pd.DataFrame(index = VGSE_DataFrame.index, columns = ['Vx', 'Vy', 'Vz'])
                # Design the Buterworth filter.
                N  = 2    # Filter order
                B, A = sp.signal.butter(N, Wn, output='ba')
                # Apply the filter.
                try:
                    VGSE_LowPass['Vx'] = sp.signal.filtfilt(B, A, VGSE_DataFrame['Vx'])
                except:
                    print('Encounter exception, skip sp.signal.filtfilt operation!')
                    VGSE_LowPass['Vx'] = VGSE_DataFrame['Vx'].copy()
                    
                try:
                    VGSE_LowPass['Vy'] = sp.signal.filtfilt(B, A, VGSE_DataFrame['Vy'])
                except:
                    print('Encounter exception, skip sp.signal.filtfilt operation!')
                    VGSE_LowPass['Vy'] = VGSE_DataFrame['Vy'].copy()
                    
                try:
                    VGSE_LowPass['Vz'] = sp.signal.filtfilt(B, A, VGSE_DataFrame['Vz'])
                except:
                    print('Encounter exception, skip sp.signal.filtfilt operation!')
                    VGSE_LowPass['Vz'] = VGSE_DataFrame['Vz'].copy()
                
                # Calculate the difference between VGSE_LowPass and VGSE_DataFrame.
                VGSE_dif = pd.DataFrame(index = VGSE_DataFrame.index, columns = ['Vx', 'Vy', 'Vz']) # Generate empty DataFrame.
                VGSE_dif['Vx'] = VGSE_DataFrame['Vx'] - VGSE_LowPass['Vx']
                VGSE_dif['Vy'] = VGSE_DataFrame['Vy'] - VGSE_LowPass['Vy']
                VGSE_dif['Vz'] = VGSE_DataFrame['Vz'] - VGSE_LowPass['Vz']
                # Calculate the mean and standard deviation of VGSE_dif.
                Vx_dif_std, Vy_dif_std, Vz_dif_std = VGSE_dif.std(skipna=True, numeric_only=True)
                Vx_dif_mean, Vy_dif_mean, Vz_dif_mean = VGSE_dif.mean(skipna=True, numeric_only=True)
                # Set the values fall outside n*std to np.nan.
                n_dif_std = 3.89 # 99.99%
                Vx_remove = (VGSE_dif['Vx']<(Vx_dif_mean-n_dif_std*Vx_dif_std))|(VGSE_dif['Vx']>(Vx_dif_mean+n_dif_std*Vx_dif_std))
                Vy_remove = (VGSE_dif['Vy']<(Vy_dif_mean-n_dif_std*Vy_dif_std))|(VGSE_dif['Vy']>(Vy_dif_mean+n_dif_std*Vy_dif_std))
                Vz_remove = (VGSE_dif['Vz']<(Vz_dif_mean-n_dif_std*Vz_dif_std))|(VGSE_dif['Vz']>(Vz_dif_mean+n_dif_std*Vz_dif_std))
                VGSE_DataFrame['Vx'][Vx_remove] = np.nan
                VGSE_DataFrame['Vy'][Vy_remove] = np.nan
                VGSE_DataFrame['Vz'][Vz_remove] = np.nan
                
                Vx_dif_lower_boundary = Vx_dif_mean-n_std*Vx_dif_std
                Vx_dif_upper_boundary = Vx_dif_mean+n_std*Vx_dif_std
                Vy_dif_lower_boundary = Vy_dif_mean-n_std*Vy_dif_std
                Vy_dif_upper_boundary = Vy_dif_mean+n_std*Vy_dif_std
                Vz_dif_lower_boundary = Vz_dif_mean-n_std*Vz_dif_std
                Vz_dif_upper_boundary = Vz_dif_mean+n_std*Vz_dif_std
                
                n_removed_Vx = sum(Vx_remove)
                n_removed_Vy = sum(Vy_remove)
                n_removed_Vz = sum(Vz_remove)
                n_removed_Vx_total += n_removed_Vx
                n_removed_Vy_total += n_removed_Vy
                n_removed_Vz_total += n_removed_Vz
                
                if isVerbose:
                    print('V_dif_std:', Vx_dif_std, Vy_dif_std, Vz_dif_std)
                    print('V_dif_mean:', Vx_dif_mean, Vy_dif_mean, Vz_dif_mean)
                    print('The VGSE Vx_dif value range within {} std is [{}, {}]'.format(n_dif_std, Vx_dif_lower_boundary, Vx_dif_upper_boundary))
                    print('The VGSE Vy_dif value range within {} std is [{}, {}]'.format(n_dif_std, Vy_dif_lower_boundary, Vy_dif_upper_boundary))
                    print('The VGSE Vz_dif value range within {} std is [{}, {}]'.format(n_dif_std, Vz_dif_lower_boundary, Vz_dif_upper_boundary))
                    print('In Vx, this operation removed {} records!'.format(n_removed_Vx))
                    print('In Vy, this operation removed {} records!'.format(n_removed_Vy))
                    print('In Vz, this operation removed {} records!!'.format(n_removed_Vz))
                    print('Until now, in Vx, {} records have been removed!'.format(n_removed_Vx_total))
                    print('Until now, in Vy, {} records have been removed!'.format(n_removed_Vy_total))
                    print('Until now, in Vz, {} records have been removed!'.format(n_removed_Vz_total))
                    print('\n')
            
            # If plot filter process or not.
            if isPlotFilterProcess:
                # Plot VGSE filter process.
                print('Plotting VGSE filtering process...')
                fig_line_width = 0.5
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
                Vx_plot.plot(VGSE_DataFrame0.index, VGSE_DataFrame0['Vx'].fillna(0),\
                             color = 'red', linewidth=fig_line_width, label='Vx_original') # Original data.
                Vx_plot.plot(VGSE_DataFrame.index, VGSE_DataFrame['Vx'].fillna(0),\
                             color = 'blue', linewidth=fig_line_width, label='Vx_processed') # Filtered data.
                Vx_plot.plot(VGSE_LowPass.index, VGSE_LowPass['Vx'].fillna(0),\
                             color = 'black', linewidth=fig_line_width, label='Vx_LowPass') # Low pass curve.
                Vx_plot.set_ylabel('Vx', fontsize=fig_ylabel_fontsize)
                Vx_plot.legend(loc='upper left',prop={'size':fig_legend_size})
                Vx_plot.tick_params(axis='y', which='major', labelsize=fig_ytick_fontsize)
                Vx_dif.plot(VGSE_dif.index, VGSE_dif['Vx'].fillna(0),\
                            color = 'green', linewidth=fig_line_width) # Difference data.
                Vx_dif.tick_params(axis='y', which='major', labelsize=fig_ytick_fontsize)
                Vx_dif.set_ylabel('Vx_dif', fontsize=fig_ylabel_fontsize)
                # Plotting Vy filter process.
                Vy_plot.plot(VGSE_DataFrame0.index, VGSE_DataFrame0['Vy'].fillna(0),\
                             color = 'red', linewidth=fig_line_width, label='Vy_original') # Original data.
                Vy_plot.plot(VGSE_DataFrame.index, VGSE_DataFrame['Vy'].fillna(0),\
                             color = 'blue', linewidth=fig_line_width, label='Vy_processed') # Filtered data.
                Vy_plot.plot(VGSE_LowPass.index, VGSE_LowPass['Vy'].fillna(0),\
                             color = 'black', linewidth=fig_line_width, label='Vy_LowPass') # Low pass curve.
                Vy_plot.set_ylabel('Vy', fontsize=fig_ylabel_fontsize)
                Vy_plot.legend(loc='upper left',prop={'size':fig_legend_size})
                Vy_plot.tick_params(axis='y', which='major', labelsize=fig_ytick_fontsize)
                Vy_dif.plot(VGSE_dif.index, VGSE_dif['Vy'].fillna(0),\
                            color = 'green', linewidth=fig_line_width) # Difference data.
                Vy_dif.tick_params(axis='y', which='major', labelsize=fig_ytick_fontsize)
                Vy_dif.set_ylabel('Vy_dif', fontsize=fig_ylabel_fontsize)
                # Plotting Vz filter process.
                Vz_plot.plot(VGSE_DataFrame0.index, VGSE_DataFrame0['Vz'].fillna(0),\
                             color = 'red', linewidth=fig_line_width, label='Vz_original') # Original data.
                Vz_plot.plot(VGSE_DataFrame.index, VGSE_DataFrame['Vz'].fillna(0),\
                             color = 'blue', linewidth=fig_line_width, label='Vz_processed') # Filtered data.
                Vz_plot.plot(VGSE_LowPass.index, VGSE_LowPass['Vz'].fillna(0),\
                             color = 'black', linewidth=fig_line_width, label='Vz_LowPass') # Low pass curve.
                Vz_plot.set_ylabel('Vz', fontsize=fig_ylabel_fontsize)
                Vz_plot.legend(loc='upper left',prop={'size':fig_legend_size})
                Vz_plot.tick_params(axis='y', which='major', labelsize=fig_ytick_fontsize)
                Vz_dif.plot(VGSE_dif.index, VGSE_dif['Vz'].fillna(0),\
                            color = 'green', linewidth=fig_line_width) # Difference data.
                Vz_dif.tick_params(axis='y', which='major', labelsize=fig_ytick_fontsize)
                Vz_dif.set_ylabel('Vz_dif', fontsize=fig_ylabel_fontsize)
                # This is a shared axis for all subplot
                Vz_dif.tick_params(axis='x', which='major', labelsize=fig_xtick_fontsize)
                # Save plot.
                fig.savefig(data_pickle_dir + '/WIND_filter_process_VGSE_' + timeStart_str + '_' + timeEnd_str + '.eps',format='eps', dpi=500, bbox_inches='tight')
        else:
            # Keep original data.
            VGSE_DataFrame0 = VGSE_DataFrame.copy(deep=True)
        # ========================================= Process Np missing value =========================================
        if not Np_DataFrame.empty:
            print('\nProcessing Np...')
            # Keep original data.
            Np_DataFrame0 = Np_DataFrame.copy(deep=True)
            #print('Np_DataFrame.shape = {}'.format(Np_DataFrame.shape))

            # Remove all data which fall outside 4 standard deviations.
            n_removed_Np_total = 0
            print('Remove all Np data which fall outside 3.89 standard deviations...')
            n_std = 3.89 # 99.99%.
            Np_std = Np_DataFrame.std(skipna=True, numeric_only=True)[0]
            Np_mean = Np_DataFrame.mean(skipna=True, numeric_only=True)[0]
            Np_remove = (Np_DataFrame['Np']<(Np_mean-n_std*Np_std))|(Np_DataFrame['Np']>(Np_mean+n_std*Np_std))
            Np_DataFrame['Np'][Np_remove] = np.nan

            Np_lower_boundary = Np_mean-n_std*Np_std
            Np_upper_boundary = Np_mean+n_std*Np_std

            n_removed_Np = sum(Np_remove)
            n_removed_Np_total += n_removed_Np

            if isVerbose:
                print('Np_std:', Np_std)
                print('Np_mean:', Np_mean)
                print('The Np value range within 3.89 std is [{}, {}]'.format(Np_lower_boundary, Np_upper_boundary))
                print('In Np, {} data has been removed!'.format(n_removed_Np))
                print('Till now, in Np, {} records have been removed!'.format(n_removed_Np_total))
                print('\n')

            # Apply Butterworth filter to Np.
            for Wn in [0.05, 0.7]: # Np
                if Wn==0.05:
                    print('Applying Butterworth filter with cutoff frequency = 0.05, remove large outliers...')
                if Wn==0.7:
                    print('Applying Butterworth filter with cutoff frequency = 0.7, remove spikes...')
                # Note that, sp.signal.butter cannot handle np.nan. Fill the nan before using it.
                Np_DataFrame.fillna(method='ffill', inplace=True) # Fill missing data, forward copy.
                Np_DataFrame.fillna(method='bfill', inplace=True) # Fill missing data, backward copy.
                # Create an empty DataFrame to store the filtered data.
                Np_LowPass = pd.DataFrame(index = Np_DataFrame.index, columns = ['Np'])
                # Design the Buterworth filter.
                N  = 2    # Filter order
                B, A = sp.signal.butter(N, Wn, output='ba')
                # Apply the filter.
                try:
                    Np_LowPass['Np'] = sp.signal.filtfilt(B, A, Np_DataFrame['Np'])
                except:
                    print('Encounter exception, skip sp.signal.filtfilt operation!')
                    Np_LowPass['Np'] = Np_DataFrame['Np'].copy()
                # Calculate the difference between Np_LowPass and Np_DataFrame.
                Np_dif = pd.DataFrame(index = Np_DataFrame.index, columns = ['Np']) # Generate empty DataFrame.
                Np_dif['Np'] = Np_DataFrame['Np'] - Np_LowPass['Np']
                # Calculate the mean and standard deviation of Np_dif. Np_dif_std is a Series object, so [0] is added.
                Np_dif_std = Np_dif.std(skipna=True, numeric_only=True)[0]
                Np_dif_mean = Np_dif.mean(skipna=True, numeric_only=True)[0]
                # Set the values fall outside n*std to np.nan.
                n_dif_std = 3.89 # 99.99%.
                Np_remove = (Np_dif['Np']<(Np_dif_mean-n_dif_std*Np_dif_std))|(Np_dif['Np']>(Np_dif_mean+n_dif_std*Np_dif_std))
                Np_DataFrame[Np_remove] = np.nan
                
                Np_dif_lower_boundary = Np_dif_mean-n_dif_std*Np_dif_std
                Np_dif_upper_boundary = Np_dif_mean+n_dif_std*Np_dif_std
                
                n_removed_Np = sum(Np_remove)
                n_removed_Np_total += n_removed_Np
                
                if isVerbose:
                    print('Np_dif_std:', Np_dif_std)
                    print('Np_dif_mean:', Np_dif_mean)
                    print('The Np_dif value range within 3 std is [{}, {}]'.format(Np_dif_lower_boundary, Np_dif_upper_boundary))
                    print('In Np, this operation removed {} records!'.format(n_removed_Np))
                    print('Till now, in Np, {} records have been removed!'.format(n_removed_Np_total))
        else:
            # Keep original data.
            Np_DataFrame0 = Np_DataFrame.copy(deep=True)
        # ========================================= Process Tp missing value =========================================
        if not Tp_DataFrame.empty:
            print('\nProcessing Tp...')
            # Keep original data.
            Tp_DataFrame0 = Tp_DataFrame.copy(deep=True)

            # Remove all data which fall outside 4 standard deviations.
            n_removed_Tp_total = 0
            print('Remove all Tp data which fall outside 3.89 standard deviations...')
            n_std = 3.89 # 99.99%.
            Tp_std = Tp_DataFrame.std(skipna=True, numeric_only=True)[0]
            Tp_mean = Tp_DataFrame.mean(skipna=True, numeric_only=True)[0]
            Tp_remove = (Tp_DataFrame['Tp']<(Tp_mean-n_std*Tp_std))|(Tp_DataFrame['Tp']>(Tp_mean+n_std*Tp_std))
            Tp_DataFrame['Tp'][Tp_remove] = np.nan

            Tp_lower_boundary = Tp_mean-n_std*Tp_std
            Tp_upper_boundary = Tp_mean+n_std*Tp_std

            n_removed_Tp = sum(Tp_remove)
            n_removed_Tp_total += n_removed_Tp

            if isVerbose:
                print('Tp_std:', Tp_std)
                print('Tp_mean:', Tp_mean)
                print('The Tp value range within 3.89 std is [{}, {}]'.format(Tp_lower_boundary, Tp_upper_boundary))
                print('In Tp, {} data has been removed!'.format(n_removed_Tp))
                print('Till now, in Tp, {} records have been removed!'.format(n_removed_Tp_total))
                print('\n')

            # Apply Butterworth filter.
            for Wn in [0.05, 0.7]: # Tp
                if Wn==0.05:
                    print('Applying Butterworth filter with cutoff frequency = 0.05, remove large outliers...')
                if Wn==0.7:
                    print('Applying Butterworth filter with cutoff frequency = 0.7, remove spikes...')
                # Note that, sp.signal.butter cannot handle np.nan. Fill the nan before using it.
                Tp_DataFrame.fillna(method='ffill', inplace=True) # Fill missing data, forward copy.
                Tp_DataFrame.fillna(method='bfill', inplace=True) # Fill missing data, backward copy.
                # Create an empty DataFrame to store the filtered data.
                Tp_LowPass = pd.DataFrame(index = Tp_DataFrame.index, columns = ['Tp'])
                # Design the Buterworth filter.
                N  = 2    # Filter order
                B, A = sp.signal.butter(N, Wn, output='ba')
                # Apply the filter.
                try:
                    Tp_LowPass['Tp'] = sp.signal.filtfilt(B, A, Tp_DataFrame['Tp'])
                except:
                    print('Encounter exception, skip sp.signal.filtfilt operation!')
                    Tp_LowPass['Tp'] = Tp_DataFrame['Tp'].copy()
                # Calculate the difference between Tp_LowPass and Tp_DataFrame.
                Tp_dif = pd.DataFrame(index = Tp_DataFrame.index, columns = ['Tp']) # Generate empty DataFrame.
                Tp_dif['Tp'] = Tp_DataFrame['Tp'] - Tp_LowPass['Tp']
                # Calculate the mean and standard deviation of Tp_dif. Tp_dif_std is a Series object, so [0] is added.
                Tp_dif_std = Tp_dif.std(skipna=True, numeric_only=True)[0]
                Tp_dif_mean = Tp_dif.mean(skipna=True, numeric_only=True)[0]
                # Set the values fall outside n*std to np.nan.
                n_dif_std = 3.89 # 99.99%.
                Tp_remove = (Tp_dif['Tp']<(Tp_dif_mean-n_dif_std*Tp_dif_std))|(Tp_dif['Tp']>(Tp_dif_mean+n_dif_std*Tp_dif_std))
                Tp_DataFrame[Tp_remove] = np.nan
                
                Tp_dif_lower_boundary = Tp_dif_mean-n_dif_std*Tp_dif_std
                Tp_dif_upper_boundary = Tp_dif_mean+n_dif_std*Tp_dif_std
                
                n_removed_Tp = sum(Tp_remove)
                n_removed_Tp_total += n_removed_Tp
            
                if isVerbose:
                    print('Tp_dif_std:', Tp_dif_std)
                    print('Tp_dif_mean:', Tp_dif_mean)
                    print('The Tp_dif value range within 3 std is [{}, {}]'.format(Tp_dif_lower_boundary, Tp_dif_upper_boundary))
                    print('In Tp, this operation removed {} records!'.format(n_removed_Tp))
                    print('Till now, in Tp, {} records have been removed!'.format(n_removed_Tp_total))
                    print('\n')

            # Convert Tp from thermal speed to Kelvin. Thermal speed is in km/s. Vth = sqrt(2KT/M)
            Tp_inKelvin_array = m_proton * np.square(np.array(Tp_DataFrame['Tp'])*1e3) / (2.0*k_Boltzmann)
            #print(Tp_DataFrame['Tp'])
            Tp_DataFrame.loc[:, 'Tp'] = Tp_inKelvin_array
            #print(Tp_DataFrame['Tp'])
            
        
        else:
            # Keep original data.
            Tp_DataFrame0 = Tp_DataFrame.copy(deep=True)
            
        # ========================================= Process N_alpha missing value =========================================
        if not N_alpha_DataFrame.empty:
            print('\nProcessing N_alpha...')
            # Keep original data.
            N_alpha_DataFrame0 = N_alpha_DataFrame.copy(deep=True)
            #print('N_alpha_DataFrame.shape = {}'.format(N_alpha_DataFrame.shape))

            # Remove all data which fall outside 4 standard deviations.
            n_removed_N_alpha_total = 0
            print('Remove all N_alpha data which fall outside 3.89 standard deviations...')
            n_std = 3.89 # 99.99%.            
            N_alpha_std = N_alpha_DataFrame.std(skipna=True, numeric_only=True)[0]
            N_alpha_mean = N_alpha_DataFrame.mean(skipna=True, numeric_only=True)[0]
            N_alpha_remove = (N_alpha_DataFrame['N_alpha']<(N_alpha_mean-n_std*N_alpha_std))|(N_alpha_DataFrame['N_alpha']>(N_alpha_mean+n_std*N_alpha_std))
            N_alpha_DataFrame['N_alpha'][N_alpha_remove] = np.nan

            N_alpha_lower_boundary = N_alpha_mean-n_std*N_alpha_std
            N_alpha_upper_boundary = N_alpha_mean+n_std*N_alpha_std

            n_removed_N_alpha = sum(N_alpha_remove)
            n_removed_N_alpha_total += n_removed_N_alpha

            if isVerbose:
                print('N_alpha_std:', N_alpha_std)
                print('N_alpha_mean:', N_alpha_mean)
                print('The N_alpha value range within 3.89 std is [{}, {}]'.format(N_alpha_lower_boundary, N_alpha_upper_boundary))
                print('In N_alpha, {} data has been removed!'.format(n_removed_N_alpha))
                print('Till now, in N_alpha, {} records have been removed!'.format(n_removed_N_alpha_total))
                print('\n')

            # Apply Butterworth filter to N_alpha.
            for Wn in [0.05, 0.7]: # N_alpha
                if Wn==0.05:
                    print('Applying Butterworth filter with cutoff frequency = 0.05, remove large outliers...')
                if Wn==0.7:
                    print('Applying Butterworth filter with cutoff frequency = 0.7, remove spikes...')
                # Note that, sp.signal.butter cannot handle np.nan. Fill the nan before using it.
                N_alpha_DataFrame.fillna(method='ffill', inplace=True) # Fill missing data, forward copy.
                N_alpha_DataFrame.fillna(method='bfill', inplace=True) # Fill missing data, backward copy.
                # Create an empty DataFrame to store the filtered data.
                N_alpha_LowPass = pd.DataFrame(index = N_alpha_DataFrame.index, columns = ['N_alpha'])
                # Design the Buterworth filter.
                N  = 2    # Filter order
                B, A = sp.signal.butter(N, Wn, output='ba')
                # Apply the filter.
                try:
                    N_alpha_LowPass['N_alpha'] = sp.signal.filtfilt(B, A, N_alpha_DataFrame['N_alpha'])
                except: 
                    print('Encounter exception, skip sp.signal.filtfilt operation!')
                    N_alpha_LowPass['N_alpha'] = N_alpha_DataFrame['N_alpha'].copy()
                # Calculate the difference between N_alpha_LowPass and N_alpha_DataFrame.
                N_alpha_dif = pd.DataFrame(index = N_alpha_DataFrame.index, columns = ['N_alpha']) # Generate empty DataFrame.
                N_alpha_dif['N_alpha'] = N_alpha_DataFrame['N_alpha'] - N_alpha_LowPass['N_alpha']
                # Calculate the mean and standard deviation of N_alpha_dif. N_alpha_dif_std is a Series object, so [0] is added.
                N_alpha_dif_std = N_alpha_dif.std(skipna=True, numeric_only=True)[0]
                N_alpha_dif_mean = N_alpha_dif.mean(skipna=True, numeric_only=True)[0]
                # Set the values fall outside n*std to np.nan.
                n_dif_std = 3.89 # 99.99%.
                N_alpha_remove = (N_alpha_dif['N_alpha']<(N_alpha_dif_mean-n_dif_std*N_alpha_dif_std))|(N_alpha_dif['N_alpha']>(N_alpha_dif_mean+n_dif_std*N_alpha_dif_std))
                N_alpha_DataFrame[N_alpha_remove] = np.nan
                
                N_alpha_dif_lower_boundary = N_alpha_dif_mean-n_dif_std*N_alpha_dif_std
                N_alpha_dif_upper_boundary = N_alpha_dif_mean+n_dif_std*N_alpha_dif_std
                
                n_removed_N_alpha = sum(N_alpha_remove)
                n_removed_N_alpha_total += n_removed_N_alpha
                
                if isVerbose:
                    print('N_alpha_dif_std:', N_alpha_dif_std)
                    print('N_alpha_dif_mean:', N_alpha_dif_mean)
                    print('The N_alpha_dif value range within 3 std is [{}, {}]'.format(N_alpha_dif_lower_boundary, N_alpha_dif_upper_boundary))
                    print('In N_alpha, this operation removed {} records!'.format(n_removed_N_alpha))
                    print('Till now, in N_alpha, {} records have been removed!'.format(n_removed_N_alpha_total))
        else:
            N_alpha_DataFrame0 = N_alpha_DataFrame.copy(deep=True)

        
        # ========================================= Process Te  =========================================
        if 'Te' in locals():
            print('\nProcessing Te...')
            # Keep original data.
            Te_DataFrame0 = Te_DataFrame.copy(deep=True)
            #print('Te_DataFrame.shape = {}'.format(Te_DataFrame.shape))

            n_removed_Te_total = 0
            print('Remove all Te data which fall outside 3.89 standard deviations...')
            n_std = 3.89 # 99.99%.
            Te_std = Te_DataFrame.std(skipna=True, numeric_only=True)[0]
            Te_mean = Te_DataFrame.mean(skipna=True, numeric_only=True)[0]
            Te_remove = (Te_DataFrame['Te']<(Te_mean-n_std*Te_std))|(Te_DataFrame['Te']>(Te_mean+n_std*Te_std))
            Te_DataFrame['Te'][Te_remove] = np.nan
            
            Te_lower_boundary = Te_mean-n_std*Te_std
            Te_upper_boundary = Te_mean+n_std*Te_std
            
            n_removed_Te = sum(Te_remove)
            n_removed_Te_total += n_removed_Te
            
            if isVerbose:
                print('Te_std:', Te_std)
                print('Te_mean:', Te_mean)
                print('The Te value range within 3.5 std is [{}, {}]'.format(Te_lower_boundary, Te_upper_boundary))
                print('In Te, {} data has been removed!'.format(n_removed_Te))
                print('Till now, in Te, {} records have been removed!'.format(n_removed_Te_total))
                print('\n')

            # Apply Butterworth filter to Te.
            for Wn in [0.05, 0.45]:
                if Wn==0.05:
                    print('Applying Butterworth filter with cutoff frequency = 0.05, remove large outliers...')
                if Wn==0.45:
                    print('Applying Butterworth filter with cutoff frequency = 0.45, remove spikes...')
                # Note that, sp.signal.butter cannot handle np.nan. Fill the nan before using it.
                Te_DataFrame.fillna(method='ffill', inplace=True) # Fill missing data, forward copy.
                Te_DataFrame.fillna(method='bfill', inplace=True) # Fill missing data, backward copy.
                # Create an empty DataFrame to store the filtered data.
                Te_LowPass = pd.DataFrame(index = Te_DataFrame.index, columns = ['Te'])
                # Design the Buterworth filter.
                N  = 2    # Filter order
                B, A = sp.signal.butter(N, Wn, output='ba')
                # Apply the filter.
                try:
                    Te_LowPass['Te'] = sp.signal.filtfilt(B, A, Te_DataFrame['Te'])
                except:
                    print('Encounter exception, skip sp.signal.filtfilt operation!')
                    Te_LowPass['Te'] = Te_DataFrame['Te'].copy()
                # Calculate the difference between Tp_LowPass and Tp_DataFrame.
                Te_dif = pd.DataFrame(index = Te_DataFrame.index, columns = ['Te']) # Generate empty DataFrame.
                Te_dif['Te'] = Te_DataFrame['Te'] - Te_LowPass['Te']
                # Calculate the mean and standard deviation of Te_dif. Te_dif_std is a Series object, so [0] is added.
                Te_dif_std = Te_dif.std(skipna=True, numeric_only=True)[0]
                Te_dif_mean = Te_dif.mean(skipna=True, numeric_only=True)[0]
                # Set the values fall outside n*std to np.nan.
                n_dif_std = 3.89 # 99.99%.
                Te_remove = (Te_dif['Te']<(Te_dif_mean-n_dif_std*Te_dif_std))|(Te_dif['Te']>(Te_dif_mean+n_dif_std*Te_dif_std))
                Te_DataFrame[Te_remove] = np.nan
                
                Te_dif_lower_boundary = Te_dif_mean-n_dif_std*Te_dif_std
                Te_dif_upper_boundary = Te_dif_mean+n_dif_std*Te_dif_std
                
                n_removed_Te = sum(Te_remove)
                n_removed_Te_total += n_removed_Te
                
                if isVerbose:
                    print('Te_dif_std:', Te_dif_std)
                    print('Te_dif_mean:', Te_dif_mean)
                    print('The Te_dif value range within 3 std is [{}, {}]'.format(Te_dif_lower_boundary, Te_dif_upper_boundary))
                    print('In Te, this operation removed {} records!'.format(n_removed_Te))
                    print('Till now, in Te, {} records have been removed!'.format(n_removed_Te_total))
                    print('\n')
                
            # There is no need to convert Te to Kelvin. Te from WIND is already in Kelvin.

        # ===================================== Plot Np, N_alpha, Tp and Te filter process =====================================
        if isPlotFilterProcess:
            fig_line_width = 0.5
            fig_ylabel_fontsize = 9
            fig_xtick_fontsize = 8
            fig_ytick_fontsize = 8
            fig_legend_size = 5
            fig,ax = plt.subplots(8,1, sharex=True,figsize=(18, 12))
            Np_plot = ax[0]
            Np_dif_plot = ax[1]
            N_alpha_plot = ax[2]
            N_alpha_dif_plot = ax[3]
            Tp_plot = ax[4]
            Tp_dif_plot = ax[5]
            Te_plot = ax[6]
            Te_dif_plot = ax[7]

            # Plotting Np filter process.
            print('Plotting Np filtering process...')
            Np_plot.plot(Np_DataFrame0.index, Np_DataFrame0['Np'].fillna(0),\
                         color = 'red', linewidth=fig_line_width, label='Np_original') # Original data.
            Np_plot.plot(Np_DataFrame.index, Np_DataFrame['Np'].fillna(0),\
                         color = 'blue', linewidth=fig_line_width, label='Np_processed') # Filtered data.
            Np_plot.plot(Np_LowPass.index, Np_LowPass['Np'].fillna(0),\
                         color = 'black', linewidth=fig_line_width, label='Np_LowPass') # Low pass curve.
            Np_plot.set_ylabel('Np', fontsize=fig_ylabel_fontsize)
            Np_plot.legend(loc='upper left',prop={'size':fig_legend_size})
            Np_plot.tick_params(axis='y', which='major', labelsize=fig_ytick_fontsize)
            Np_dif_plot.plot(Np_dif.index, Np_dif['Np'].fillna(0), color = 'green', linewidth=fig_line_width) # Difference data.
            Np_dif_plot.tick_params(axis='y', which='major', labelsize=fig_ytick_fontsize)
            Np_dif_plot.set_ylabel('Np_dif', fontsize=fig_ylabel_fontsize)
            # Plotting N_alpha filter process.
            print('Plotting N_alpha filtering process...')
            N_alpha_plot.plot(N_alpha_DataFrame0.index, N_alpha_DataFrame0['N_alpha'].fillna(0),\
                         color = 'red', linewidth=fig_line_width, label='N_alpha_original') # Original data.
            N_alpha_plot.plot(N_alpha_DataFrame.index, N_alpha_DataFrame['N_alpha'].fillna(0),\
                         color = 'blue', linewidth=fig_line_width, label='N_alpha_processed') # Filtered data.
            N_alpha_plot.plot(N_alpha_LowPass.index, N_alpha_LowPass['N_alpha'].fillna(0),\
                         color = 'black', linewidth=fig_line_width, label='N_alpha_LowPass') # Low pass curve.
            N_alpha_plot.set_ylabel('N_alpha', fontsize=fig_ylabel_fontsize)
            N_alpha_plot.legend(loc='upper left',prop={'size':fig_legend_size})
            N_alpha_plot.tick_params(axis='y', which='major', labelsize=fig_ytick_fontsize)
            N_alpha_dif_plot.plot(N_alpha_dif.index, N_alpha_dif['N_alpha'].fillna(0), color = 'green', linewidth=fig_line_width) # Difference data.
            N_alpha_dif_plot.tick_params(axis='y', which='major', labelsize=fig_ytick_fontsize)
            N_alpha_dif_plot.set_ylabel('N_alpha_dif', fontsize=fig_ylabel_fontsize)
            # Plotting Tp filter process.
            print('Plotting Tp filtering process...')
            Tp_plot.plot(Tp_DataFrame0.index, Tp_DataFrame0['Tp'].fillna(0),\
                         color = 'red', linewidth=fig_line_width, label='Tp_original') # Original data.
            Tp_plot.plot(Tp_DataFrame.index, Tp_DataFrame['Tp'].fillna(0),\
                         color = 'blue', linewidth=fig_line_width, label='Tp_processed') # Filtered data.
            Tp_plot.plot(Tp_LowPass.index, Tp_LowPass['Tp'].fillna(0),\
                         color = 'black', linewidth=fig_line_width, label='Tp_LowPass') # Low pass curve.
            Tp_plot.set_ylabel('Tp', fontsize=fig_ylabel_fontsize)
            Tp_plot.legend(loc='upper left',prop={'size':fig_legend_size})
            Tp_plot.tick_params(axis='y', which='major', labelsize=fig_ytick_fontsize)
            Tp_dif_plot.plot(Tp_dif.index, Tp_dif['Tp'].fillna(0),\
                             color = 'green', linewidth=fig_line_width) # Difference data.
            Tp_dif_plot.tick_params(axis='y', which='major', labelsize=fig_ytick_fontsize)
            Tp_dif_plot.set_ylabel('Tp_dif', fontsize=fig_ylabel_fontsize)
            
            # Plotting Te filter process.
            if 'Te' in locals():
                print('Plotting Te filtering process...')
                Te_plot.plot(Te_DataFrame0.index, Te_DataFrame0['Te'].fillna(0),\
                color = 'red', linewidth=fig_line_width, label='Te_original') # Original data.
                Te_plot.plot(Te_DataFrame.index, Te_DataFrame['Te'].fillna(0),\
                color = 'blue', linewidth=fig_line_width, label='Te_processed') # Filtered data.
                Te_plot.plot(Te_LowPass.index, Te_LowPass['Te'].fillna(0),\
                color = 'black', linewidth=fig_line_width, label='Te_LowPass') # Low pass curve.
                Te_plot.set_ylabel('Te', fontsize=fig_ylabel_fontsize)
                Te_plot.legend(loc='upper left',prop={'size':fig_legend_size})
                Te_plot.tick_params(axis='y', which='major', labelsize=fig_ytick_fontsize)
                Te_dif_plot.plot(Te_dif.index, Te_dif['Te'].fillna(0),\
                color = 'green', linewidth=fig_line_width) # Difference data.
                Te_dif_plot.set_ylabel('Te_dif', fontsize=fig_ylabel_fontsize)
                Te_dif_plot.tick_params(axis='y', which='major', labelsize=fig_ytick_fontsize)
                # This is a shared axis for all subplot
                Te_dif_plot.tick_params(axis='x', which='major', labelsize=fig_xtick_fontsize)
            else:
                # This is a shared axis for all subplot
                Tp_dif_plot.tick_params(axis='x', which='major', labelsize=fig_xtick_fontsize)
            # Save plot.
            fig.savefig(data_pickle_dir + '/ACE_filter_process_Np_N_alpha_Tp_Te_' + timeStart_str + '_' + timeEnd_str + '.eps',format='eps', dpi=500, bbox_inches='tight')

        # ============================== Resample data to 1min resolution ==============================.

        n_interp_limit = 10

        # Resample BGSE data into one minute resolution.
        # Linear fit first, to make sure there is no NaN. NaN will propagate by resample.
        # Interpolate according to timestamps. Cannot handle boundary. Do not interpolate NaN longer than 10.
        BGSE_DataFrame.interpolate(method='time', inplace=True, limit=n_interp_limit)
        print('\nResampling BGSE data into 1 minute resolution...')
        BGSE_DataFrame.drop_duplicates(keep='first', inplace=True) # Drop duplicated records, keep first one.
        # Resample to 1 minute resolution. New added records will be filled with NaN.
        BGSE_DataFrame = BGSE_DataFrame.resample('1T').mean()
        # Interpolate according to timestamps. Cannot handle boundary.
        BGSE_DataFrame.interpolate(method='time', inplace=True, limit=n_interp_limit)

        if not VGSE_DataFrame.empty:
            # Resample VGSE data into one minute resolution.
            # Interpolate according to timestamps. Cannot handle boundary.
            VGSE_DataFrame.interpolate(method='time', inplace=True, limit=n_interp_limit)
            print('Resampling VGSE data into 1 minute resolution...')
            VGSE_DataFrame.drop_duplicates(keep='first', inplace=True) # Drop duplicated records, keep first one.
            # Resample to 1 minute resolution. New added records will be filled with NaN.
            VGSE_DataFrame = VGSE_DataFrame.resample('1T').mean()
            # Interpolate according to timestamps. Cannot handle boundary.
            VGSE_DataFrame.interpolate(method='time', inplace=True, limit=n_interp_limit)
        if not Np_DataFrame.empty:
            # Resample Np data into one minute resolution.
            # Linear fit first, to make sure there is no NaN. NaN will propagate by resample.
            # Interpolate according to timestamps. Cannot handle boundary.
            Np_DataFrame.interpolate(method='time', inplace=True, limit=n_interp_limit)
            print('Resampling Np data into 1 minute resolution...')
            Np_DataFrame.drop_duplicates(keep='first', inplace=True) # Drop duplicated records, keep first one.
            # Resample to 1 minute resolution. New added records will be filled with NaN.
            Np_DataFrame = Np_DataFrame.resample('1T').mean()
            # Interpolate according to timestamps. Cannot handle boundary.
            Np_DataFrame.interpolate(method='time', inplace=True, limit=n_interp_limit)
        if not Tp_DataFrame.empty:
            # Resample Tp data into one minute resolution.
            # Linear fit first, to make sure there is no NaN. NaN will propagate by resample.
            # Interpolate according to timestamps. Cannot handle boundary.
            Tp_DataFrame.interpolate(method='time', inplace=True)
            print('Resampling Tp data into 1 minute resolution...')
            Tp_DataFrame.drop_duplicates(keep='first', inplace=True) # Drop duplicated records, keep first one.
            # Resample to 1 minute resolution. New added records will be filled with NaN.
            # Resample to 30 second first, or data points will shift too much, as large as 1 min.
            Tp_DataFrame = Tp_DataFrame.resample('1T').mean()
            # Interpolate according to timestamps. Cannot handle boundary.
            Tp_DataFrame.interpolate(method='time', inplace=True, limit=n_interp_limit)
            
        if not N_alpha_DataFrame.empty:
            # Resample N_alpha data into one minute resolution.
            # Linear fit first, to make sure there is no NaN. NaN will propagate by resample.
            # Interpolate according to timestamps. Cannot handle boundary.
            N_alpha_DataFrame.interpolate(method='time', inplace=True, limit=n_interp_limit)
            print('Resampling N_alpha data into 1 minute resolution...')
            N_alpha_DataFrame.drop_duplicates(keep='first', inplace=True) # Drop duplicated records, keep first one.
            # Resample to 1 minute resolution. New added records will be filled with NaN.
            N_alpha_DataFrame = N_alpha_DataFrame.resample('1T').mean()
            # Interpolate according to timestamps. Cannot handle boundary.
            N_alpha_DataFrame.interpolate(method='time', inplace=True, limit=n_interp_limit)

        if 'Te' in locals():
            if not Te_DataFrame.empty:
                # Resample Te data into one minute resolution.
                # Linear fit first, to make sure there is no NaN. NaN will propagate by resample.
                # Interpolate according to timestamps. Cannot handle boundary.
                Te_DataFrame.interpolate(method='time', inplace=True, limit=n_interp_limit)
                print('Resampling Te data into 1 minute resolution...')
                Te_DataFrame.drop_duplicates(keep='first', inplace=True) # Drop duplicated records, keep first one.
                # Resample to 1 minute resolution. New added records will be filled with NaN.
                # Resample to 30 second first, or data points will shift too much, as large as 1 min.
                Te_DataFrame = Te_DataFrame.resample('1T').mean()
                # Interpolate according to timestamps. Cannot handle boundary.
                Te_DataFrame.interpolate(method='time', inplace=True, limit=n_interp_limit)

        # Merge all DataFrames into one according to time index.
        # Calculate time range in minutes.
        #timeRangeInMinutes = int((BGSE_DataFrame.index[-1] - BGSE_DataFrame.index[0]).total_seconds())//60
        timeRangeInMinutes = int((timeEnd - timeStart).total_seconds())//60
        # Generate timestamp index.
        index_datetime = np.asarray([timeStart + timedelta(minutes=x) for x in range(0, timeRangeInMinutes+1)])
        # Generate empty DataFrame according using index_datetime as index.
        GS_AllData_DataFrame = pd.DataFrame(index=index_datetime)
        # Merge all DataFrames.
        if 'Te' in locals():
            GS_AllData_DataFrame = pd.concat([GS_AllData_DataFrame, \
            BGSE_DataFrame, VGSE_DataFrame, Np_DataFrame, N_alpha_DataFrame, Tp_DataFrame, Te_DataFrame], axis=1)
        else:
            GS_AllData_DataFrame = pd.concat([GS_AllData_DataFrame, \
            BGSE_DataFrame, VGSE_DataFrame, Np_DataFrame, N_alpha_DataFrame, Tp_DataFrame], axis=1)
        # Save merged DataFrame into pickle file.
        GS_AllData_DataFrame.to_pickle(data_pickle_dir + '/WIND_' + timeStart_str + '-' + timeEnd_str + '_preprocessed.p')

        if isCheckDataIntegrity:
            print('Checking the number of NaNs in GS_AllData_DataFrame...')
            len_GS_AllData_DataFrame = len(GS_AllData_DataFrame)
            for key in GS_AllData_DataFrame.keys():
                num_notNaN = GS_AllData_DataFrame[key].isnull().values.sum()
                percent_notNaN = 100.0 - num_notNaN * 100.0 / len_GS_AllData_DataFrame
                print('The number of NaNs in {} is {}, integrity is {}%'.format(key, num_notNaN, round(percent_notNaN, 2)))
        
        if isSaveOriginalData:
            # Save original data into dataframe.
            # Interpolate original data, remove NaN.
            BGSE_DataFrame0.interpolate(method='time', inplace=True, limit=5)
            VGSE_DataFrame0.interpolate(method='time', inplace=True, limit=5)
            Np_DataFrame0.interpolate(method='time', inplace=True, limit=5)
            N_alpha_DataFrame.interpolate(method='time', inplace=True, limit=5)
            Tp_DataFrame0.interpolate(method='time', inplace=True, limit=5)
            if 'Te' in locals():
                Te_DataFrame0.interpolate(method='time', inplace=True, limit=5)
            # Resample original data to 1T.
            BGSE_DataFrame0 = BGSE_DataFrame0.resample('1T').mean()
            VGSE_DataFrame0 = VGSE_DataFrame0.resample('1T').mean()
            Np_DataFrame0 = Np_DataFrame0.resample('1T').mean()
            N_alpha_DataFrame0 = N_alpha_DataFrame0.resample('1T').mean()
            Tp_DataFrame0 = Tp_DataFrame0.resample('1T').mean()
            if 'Te' in locals():
                Te_DataFrame0 = Te_DataFrame0.resample('1T').mean()
            # Reindex, or will raise ValueError: Shape of passed values is (), indices imply ().
            BGSE_DataFrame0.reindex(index_datetime)
            VGSE_DataFrame0.reindex(index_datetime)
            Np_DataFrame0.reindex(index_datetime)
            N_alpha_DataFrame0.reindex(index_datetime)
            Tp_DataFrame0.reindex(index_datetime)
            if 'Te' in locals():
                Te_DataFrame0.reindex(index_datetime)
            # Merge all original DataFrames.
            if 'Te' in locals():
                GS_AllData_DataFrame_original = pd.concat([BGSE_DataFrame0, VGSE_DataFrame0, Np_DataFrame0, N_alpha_DataFrame0, Tp_DataFrame0, Te_DataFrame0], axis=1)
            else:
                GS_AllData_DataFrame_original = pd.concat([BGSE_DataFrame0, VGSE_DataFrame0, Np_DataFrame0, N_alpha_DataFrame0, Tp_DataFrame0], axis=1)
            # Interpolate 1T original data, remove NaN.
            GS_AllData_DataFrame_original.interpolate(method='time', inplace=True, limit=5)

            if isCheckDataIntegrity:
                print('Checking the number of NaNs in WIND_AllData_DataFrame_original...')
                len_GS_AllData_DataFrame_original = len(GS_AllData_DataFrame_original)
                for key in GS_AllData_DataFrame_original.keys():
                    num_notNaN = GS_AllData_DataFrame_original[key].isnull().values.sum()
                    percent_notNaN = 100.0 - num_notNaN * 100.0 / len_GS_AllData_DataFrame_original
                    print('The number of NaNs in {} is {}, integrity is {}%'.format(key, num_notNaN, round(percent_notNaN, 2)))

            # Save merged DataFrame into pickle file.
            GS_AllData_DataFrame_original.to_pickle(data_pickle_dir + '/WIND_' + timeStart_str + '_' + timeEnd_str + '_original.p')

        print('\nData preprocessing is done!')

        return GS_AllData_DataFrame

    elif data_dict['ID']=='ACE':
        print('\nSpacecraft ID: WIND')
        
        # Extract data dict file.
        print('Extracting BGSE_Epoch...')
        if data_dict['AC_H0_MFI'] is not None:
            BGSE_Epoch = data_dict['AC_H0_MFI']['Epoch']
        else:
            BGSE_Epoch = None

        print('Extracting BGSEc...')
        if data_dict['AC_H0_MFI'] is not None:
            BGSE = data_dict['AC_H0_MFI']['BGSEc'] # Magnetic file in GSE coordinate.
        else:
            BGSE = None
        
        print('Extracting SW_Epoch...')
        if data_dict['AC_H0_SWE'] is not None:
            SW_Epoch = data_dict['AC_H0_SWE']['Epoch']
        else:
            SW_Epoch = None
        
        print('Extracting V_GSE...')
        if data_dict['AC_H0_SWE'] is not None:
            VGSE = data_dict['AC_H0_SWE']['V_GSE'] # Solar wind speed in GSE coordinate.
        else:
            VGSE = None
        
        print('Extracting Np...')
        if data_dict['AC_H0_SWE'] is not None:
            Np = data_dict['AC_H0_SWE']['Np'] # Proton number density.
        else:
            Np = None  
        
        print('Extracting Tpr...')
        if data_dict['AC_H0_SWE'] is not None:
            Tpr = data_dict['AC_H0_SWE']['Tpr'] # Proton thermal speed.
        else:
            Tpr = None
        
        print('Extracting alpha_ratio...')
        if data_dict['AC_H0_SWE'] is not None:
            Alpha2Proton_ratio = data_dict['AC_H0_SWE']['alpha_ratio'] # Na/Np.
        else:
            Alpha2Proton_ratio = None
        
        print('Extracting LEMS_Epoch...')
        if data_dict['AC_H3_EPM'] is not None:
            LEMS_Epoch = data_dict['AC_H3_EPM']['Epoch']
        else:
            LEMS_Epoch = None
        
        print('Extracting LEMS30 P1 ~ P8...') # Low energetic ions from LEMS30.
        if data_dict['AC_H3_EPM'] is not None:
            LEMS30_P1 = data_dict['AC_H3_EPM']['P1']
            LEMS30_P2 = data_dict['AC_H3_EPM']['P2']
            LEMS30_P3 = data_dict['AC_H3_EPM']['P3']
            LEMS30_P4 = data_dict['AC_H3_EPM']['P4']
            LEMS30_P5 = data_dict['AC_H3_EPM']['P5']
            LEMS30_P6 = data_dict['AC_H3_EPM']['P6']
            LEMS30_P7 = data_dict['AC_H3_EPM']['P7']
            LEMS30_P8 = data_dict['AC_H3_EPM']['P8']
        else:
            LEMS30_P1, LEMS30_P2, LEMS30_P3, LEMS30_P4, LEMS30_P5, LEMS30_P6, LEMS30_P7, LEMS30_P8 = (None,None,None,None,None,None,None,None)
        print('Extracting LEMS120 P1p ~ P8p...') # Low energetic ions from LEMS120.
        if data_dict['AC_H3_EPM'] is not None:
            LEMS120_P1 = data_dict['AC_H3_EPM']['P1p']
            LEMS120_P2 = data_dict['AC_H3_EPM']['P2p']
            LEMS120_P3 = data_dict['AC_H3_EPM']['P3p']
            LEMS120_P4 = data_dict['AC_H3_EPM']['P4p']
            LEMS120_P5 = data_dict['AC_H3_EPM']['P5p']
            LEMS120_P6 = data_dict['AC_H3_EPM']['P6p']
            LEMS120_P7 = data_dict['AC_H3_EPM']['P7p']
            LEMS120_P8 = data_dict['AC_H3_EPM']['P8p']
        else:
            LEMS120_P1, LEMS120_P2, LEMS120_P3, LEMS120_P4, LEMS120_P5, LEMS120_P6, LEMS120_P7, LEMS120_P8 = (None,None,None,None,None,None,None,None)
        
        # Process missing value. missing value = -9.9999998e+30.
        print('Processing missing value...')
        if BGSE is not None:
            BGSE[abs(BGSE) > 80] = np.nan # B field.
        if Np is not None:
            Np[Np < -1e+10] = np.nan # Proton number density.
        if VGSE is not None:
            VGSE[abs(VGSE) > 1500] = np.nan # Solar wind speed.
        if Tpr is not None:
            Tpr[Tpr < -1e+10] = np.nan # Proton temperature, radial component of T tensor.
        if Alpha2Proton_ratio is not None:
            Alpha2Proton_ratio[Alpha2Proton_ratio < -1e+10] = np.nan # Na/Np.
            
        # Particle counts must be greater than 1.
        if data_dict['AC_H3_EPM'] is not None:
            LEMS30_P1[LEMS30_P1 < 1] = np.nan
            LEMS30_P2[LEMS30_P2 < 1] = np.nan
            LEMS30_P3[LEMS30_P3 < 1] = np.nan
            LEMS30_P4[LEMS30_P4 < 1] = np.nan
            LEMS30_P5[LEMS30_P5 < 1] = np.nan
            LEMS30_P6[LEMS30_P6 < 1] = np.nan
            LEMS30_P7[LEMS30_P7 < 1] = np.nan
            LEMS30_P8[LEMS30_P8 < 1] = np.nan
            LEMS120_P1[LEMS120_P1 < 1] = np.nan
            LEMS120_P2[LEMS120_P2 < 1] = np.nan
            LEMS120_P3[LEMS120_P3 < 1] = np.nan
            LEMS120_P4[LEMS120_P4 < 1] = np.nan
            LEMS120_P5[LEMS120_P5 < 1] = np.nan
            LEMS120_P6[LEMS120_P6 < 1] = np.nan
            LEMS120_P7[LEMS120_P7 < 1] = np.nan
            LEMS120_P8[LEMS120_P8 < 1] = np.nan

        # Put data into DataFrame.
        print('Putting data into DataFrame...')
        if BGSE_Epoch is None:
            return None
        else:
            BGSE_DataFrame = pd.DataFrame(BGSE, index = BGSE_Epoch, columns = ['Bx', 'By', 'Bz'])
        
        if SW_Epoch is not None:
            VGSE_DataFrame = pd.DataFrame(VGSE, index = SW_Epoch, columns = ['Vx', 'Vy', 'Vz'])
            Np_DataFrame = pd.DataFrame(Np, index = SW_Epoch, columns = ['Np'])
            Tp_DataFrame = pd.DataFrame(Tpr, index = SW_Epoch, columns = ['Tp'])
            Alpha2Proton_ratio_DataFrame = pd.DataFrame(Alpha2Proton_ratio, index = SW_Epoch, columns = ['Alpha2Proton_ratio'])
        else:
            VGSE_DataFrame = pd.DataFrame(None, columns = ['Vx', 'Vy', 'Vz'])
            Np_DataFrame = pd.DataFrame(None, columns = ['Np'])
            Tp_DataFrame = pd.DataFrame(None, columns = ['Tp'])
            Alpha2Proton_ratio_DataFrame = pd.DataFrame(None, columns = ['Alpha2Proton_ratio'])
        
        if LEMS_Epoch is not None:
            LEMS_DataFrame = pd.DataFrame({'LEMS30_P1': LEMS30_P1, 'LEMS30_P2': LEMS30_P2, 'LEMS30_P3': LEMS30_P3, 'LEMS30_P4': LEMS30_P4, 'LEMS30_P5': LEMS30_P5, 'LEMS30_P6': LEMS30_P6, 'LEMS30_P7': LEMS30_P7, 'LEMS30_P8': LEMS30_P8,'LEMS120_P1': LEMS120_P1, 'LEMS120_P2': LEMS120_P2, 'LEMS120_P3': LEMS120_P3, 'LEMS120_P4': LEMS120_P4, 'LEMS120_P5': LEMS120_P5, 'LEMS120_P6': LEMS120_P6, 'LEMS120_P7': LEMS120_P7, 'LEMS120_P8': LEMS120_P8}, index = LEMS_Epoch)
        else:
            LEMS_DataFrame = pd.DataFrame(None, columns = ['LEMS30_P1', 'LEMS30_P2' , 'LEMS30_P3', 'LEMS30_P4', 'LEMS30_P5', 'LEMS30_P6', 'LEMS30_P7', 'LEMS30_P8','LEMS120_P1', 'LEMS120_P2', 'LEMS120_P3', 'LEMS120_P4', 'LEMS120_P5', 'LEMS120_P6', 'LEMS120_P7', 'LEMS120_P8'])
        
        # Trim data. Some times cdas API will download wrong time range.
        if BGSE_Epoch is not None:
            BGSE_DataFrame = BGSE_DataFrame[(BGSE_DataFrame.index>=timeStart)&(BGSE_DataFrame.index<=timeEnd)]
        if SW_Epoch is not None:
            VGSE_DataFrame = VGSE_DataFrame[(VGSE_DataFrame.index>=timeStart)&(VGSE_DataFrame.index<=timeEnd)]
            Np_DataFrame = Np_DataFrame[(Np_DataFrame.index>=timeStart)&(Np_DataFrame.index<=timeEnd)]
            Tp_DataFrame = Tp_DataFrame[(Tp_DataFrame.index>=timeStart)&(Tp_DataFrame.index<=timeEnd)]
            Alpha2Proton_ratio_DataFrame = Alpha2Proton_ratio_DataFrame[(Alpha2Proton_ratio_DataFrame.index>=timeStart)&(Alpha2Proton_ratio_DataFrame.index<=timeEnd)]
        if LEMS_Epoch is not None: 
            LEMS_DataFrame = LEMS_DataFrame[(LEMS_DataFrame.index>=timeStart)&(LEMS_DataFrame.index<=timeEnd)]

        # Drop duplicated records. This is the flaw of the source data.
        print('Dropping duplicated records...')
        BGSE_DataFrame.drop_duplicates(keep='first', inplace=True) # Drop duplicated records, keep first one.
        VGSE_DataFrame.drop_duplicates(keep='first', inplace=True) # Drop duplicated records, keep first one.
        Np_DataFrame.drop_duplicates(keep='first', inplace=True) # Drop duplicated records, keep first one.
        Tp_DataFrame.drop_duplicates(keep='first', inplace=True) # Drop duplicated records, keep first one.
        Alpha2Proton_ratio_DataFrame.drop_duplicates(keep='first', inplace=True) # Drop duplicated records, keep first one.
        LEMS_DataFrame.drop_duplicates(keep='first', inplace=True) # Drop duplicated records, keep first one.

        # Sort data by time index. Time series data may be not in order, This is the flaw of the source data.
        print('Sorting data...')
        BGSE_DataFrame.sort_index(axis=0, ascending=True, inplace=True, kind='quicksort')
        VGSE_DataFrame.sort_index(axis=0, ascending=True, inplace=True, kind='quicksort')
        Np_DataFrame.sort_index(axis=0, ascending=True, inplace=True, kind='quicksort')
        Tp_DataFrame.sort_index(axis=0, ascending=True, inplace=True, kind='quicksort')
        Alpha2Proton_ratio_DataFrame.sort_index(axis=0, ascending=True, inplace=True, kind='quicksort')
        LEMS_DataFrame.sort_index(axis=0, ascending=True, inplace=True, kind='quicksort')
        
        #========================================= Process BGSE value =========================================
        print('\nProcessing BGSE...')
        # Keep original data.
        BGSE_DataFrame0 = BGSE_DataFrame.copy(deep=True)
        #print('BGSE_DataFrame.shape = {}'.format(BGSE_DataFrame.shape))

        n_removed_Bx_total = 0
        n_removed_By_total = 0
        n_removed_Bz_total = 0

        # Apply Butterworth filter to BGSE.
        for Wn in [0.45]:
            print('Applying Butterworth filter with cutoff frequency = {}, remove spikes...'.format(Wn))
            # Note that, sp.signal.butter cannot handle np.nan. Fill the nan before using it.
            BGSE_DataFrame.fillna(method='ffill', inplace=True) # Fill missing data, forward copy.
            BGSE_DataFrame.fillna(method='bfill', inplace=True) # Fill missing data, backward copy.
            # Create an empty DataFrame to store the filtered data.
            BGSE_LowPass = pd.DataFrame(index = BGSE_DataFrame.index, columns = ['Bx', 'By', 'Bz'])
            # Design the Buterworth filter.
            N  = 2    # Filter order
            B, A = sp.signal.butter(N, Wn, output='ba')
            # Apply the filter.
            try:
                BGSE_LowPass['Bx'] = sp.signal.filtfilt(B, A, BGSE_DataFrame['Bx'])
            except:
                print('Encounter exception, skip sp.signal.filtfilt operation!')
                BGSE_LowPass['Bx'] = BGSE_DataFrame['Bx'].copy()
                
            try:
                BGSE_LowPass['By'] = sp.signal.filtfilt(B, A, BGSE_DataFrame['By'])
            except:
                print('Encounter exception, skip sp.signal.filtfilt operation!')
                BGSE_LowPass['By'] = BGSE_DataFrame['By'].copy()
            
            try:
                BGSE_LowPass['Bz'] = sp.signal.filtfilt(B, A, BGSE_DataFrame['Bz'])
            except:
                print('Encounter exception, skip sp.signal.filtfilt operation!')
                BGSE_LowPass['Bz'] = BGSE_DataFrame['Bz'].copy()
            
            # Calculate the difference between BGSE_LowPass and BGSE_DataFrame.
            BGSE_dif = pd.DataFrame(index = BGSE_DataFrame.index, columns = ['Bx', 'By', 'Bz']) # Generate empty DataFrame.
            BGSE_dif['Bx'] = BGSE_DataFrame['Bx'] - BGSE_LowPass['Bx']
            BGSE_dif['By'] = BGSE_DataFrame['By'] - BGSE_LowPass['By']
            BGSE_dif['Bz'] = BGSE_DataFrame['Bz'] - BGSE_LowPass['Bz']
            # Calculate the mean and standard deviation of BGSE_dif.
            Bx_dif_std, By_dif_std, Bz_dif_std = BGSE_dif.std(skipna=True, numeric_only=True)
            Bx_dif_mean, By_dif_mean, Bz_dif_mean = BGSE_dif.mean(skipna=True, numeric_only=True)
            # Set the values fall outside n*std to np.nan.
            n_dif_std = 4.417 # 99.999%
            Bx_remove = (BGSE_dif['Bx']<(Bx_dif_mean-n_dif_std*Bx_dif_std))|(BGSE_dif['Bx']>(Bx_dif_mean+n_dif_std*Bx_dif_std))
            By_remove = (BGSE_dif['By']<(By_dif_mean-n_dif_std*By_dif_std))|(BGSE_dif['By']>(By_dif_mean+n_dif_std*By_dif_std))
            Bz_remove = (BGSE_dif['Bz']<(Bz_dif_mean-n_dif_std*Bz_dif_std))|(BGSE_dif['Bz']>(Bz_dif_mean+n_dif_std*Bz_dif_std))
            BGSE_DataFrame['Bx'][Bx_remove] = np.nan
            BGSE_DataFrame['By'][By_remove] = np.nan
            BGSE_DataFrame['Bz'][Bz_remove] = np.nan
            
            Bx_dif_lower_boundary = Bx_dif_mean-n_dif_std*Bx_dif_std
            Bx_dif_upper_boundary = Bx_dif_mean+n_dif_std*Bx_dif_std
            By_dif_lower_boundary = By_dif_mean-n_dif_std*By_dif_std
            By_dif_upper_boundary = By_dif_mean+n_dif_std*By_dif_std
            Bz_dif_lower_boundary = Bz_dif_mean-n_dif_std*Bz_dif_std
            Bz_dif_upper_boundary = Bz_dif_mean+n_dif_std*Bz_dif_std
            
            n_removed_Bx = sum(Bx_remove)
            n_removed_By = sum(By_remove)
            n_removed_Bz = sum(Bz_remove)
            n_removed_Bx_total += n_removed_Bx
            n_removed_By_total += n_removed_By
            n_removed_Bz_total += n_removed_Bz
            
            if isVerbose:
                print('B_dif_std:', Bx_dif_std, By_dif_std, Bz_dif_std)
                print('B_dif_mean:', Bx_dif_mean, By_dif_mean, Bz_dif_mean)
                print('The BGSE Bx_dif value range within {} std is [{}, {}]'.format(n_dif_std, Bx_dif_lower_boundary, Bx_dif_upper_boundary))
                print('The BGSE By_dif value range within {} std is [{}, {}]'.format(n_dif_std, By_dif_lower_boundary, By_dif_upper_boundary))
                print('The BGSE Bz_dif value range within {} std is [{}, {}]'.format(n_dif_std, Bz_dif_lower_boundary, Bz_dif_upper_boundary))
                print('In Bx, this operation removed {} records!'.format(n_removed_Bx))
                print('In By, this operation removed {} records!'.format(n_removed_By))
                print('In Bz, this operation removed {} records!!'.format(n_removed_Bz))
                print('Till now, in Bx, {} records have been removed!'.format(n_removed_Bx_total))
                print('Till now, in By, {} records have been removed!'.format(n_removed_By_total))
                print('Till now, in Bz, {} records have been removed!'.format(n_removed_Bz_total))
                print('\n')
        
        # If plot filter process of BGSE or not.
        if isPlotFilterProcess:
            # Plot BGSE filter process.
            print('Plotting BGSE filtering process...')
            fig_line_width = 0.5
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
            Bx_plot.plot(BGSE_DataFrame0.index, BGSE_DataFrame0['Bx'].fillna(0),\
                         color = 'red', linewidth=fig_line_width, label='Bx_original') # Original data.
            Bx_plot.plot(BGSE_DataFrame.index, BGSE_DataFrame['Bx'].fillna(0),\
                         color = 'blue', linewidth=fig_line_width, label='Bx_processed') # Filtered data.
            Bx_plot.plot(BGSE_LowPass.index, BGSE_LowPass['Bx'].fillna(0),\
                         color = 'black', linewidth=fig_line_width, label='Bx_LowPass') # Low pass curve.
            Bx_plot.set_ylabel('Bx', fontsize=fig_ylabel_fontsize)
            Bx_plot.legend(loc='upper left',prop={'size':fig_legend_size})
            Bx_plot.tick_params(axis='y', which='major', labelsize=fig_ytick_fontsize)
            Bx_dif.plot(BGSE_dif.index, BGSE_dif['Bx'].fillna(0),\
                        color = 'green', linewidth=fig_line_width) # Difference data.
            Bx_dif.tick_params(axis='y', which='major', labelsize=fig_ytick_fontsize)
            Bx_dif.set_ylabel('Bx_dif', fontsize=fig_ylabel_fontsize)
            # Plotting By filter process.
            By_plot.plot(BGSE_DataFrame0.index, BGSE_DataFrame0['By'].fillna(0),\
                         color = 'red', linewidth=fig_line_width, label='By_original') # Original data.
            By_plot.plot(BGSE_DataFrame.index, BGSE_DataFrame['By'].fillna(0),\
                         color = 'blue', linewidth=fig_line_width, label='By_processed') # Filtered data.
            By_plot.plot(BGSE_LowPass.index, BGSE_LowPass['By'].fillna(0),\
                         color = 'black', linewidth=fig_line_width, label='By_LowPass') # Low pass curve.
            By_plot.set_ylabel('By', fontsize=fig_ylabel_fontsize)
            By_plot.tick_params(axis='y', which='major', labelsize=fig_ytick_fontsize)
            By_plot.legend(loc='upper left',prop={'size':fig_legend_size})
            By_dif.plot(BGSE_dif.index, BGSE_dif['By'].fillna(0),\
                        color = 'green', linewidth=fig_line_width) # Difference data.
            By_dif.tick_params(axis='y', which='major', labelsize=fig_ytick_fontsize)
            By_dif.set_ylabel('By_dif', fontsize=fig_ylabel_fontsize)
            # Plotting Bz filter process.
            Bz_plot.plot(BGSE_DataFrame0.index, BGSE_DataFrame0['Bz'].fillna(0),\
                         color = 'red', linewidth=fig_line_width, label='Bz_original') # Original data.
            Bz_plot.plot(BGSE_DataFrame.index, BGSE_DataFrame['Bz'].fillna(0),\
                         color = 'blue', linewidth=fig_line_width, label='Bz_processed') # Filtered data.
            Bz_plot.plot(BGSE_LowPass.index, BGSE_LowPass['Bz'].fillna(0),\
                         color = 'black', linewidth=fig_line_width, label='Bz_LowPass') # Low pass curve.
            Bz_plot.set_ylabel('Bz', fontsize=fig_ylabel_fontsize)
            Bz_plot.tick_params(axis='y', which='major', labelsize=fig_ytick_fontsize)
            Bz_plot.legend(loc='upper left',prop={'size':fig_legend_size})
            Bz_dif.plot(BGSE_dif.index, BGSE_dif['Bz'].fillna(0),\
                        color = 'green', linewidth=fig_line_width) # Difference data.
            Bz_dif.tick_params(axis='y', which='major', labelsize=fig_ytick_fontsize)
            Bz_dif.set_ylabel('Bz_dif', fontsize=fig_ylabel_fontsize)
            # This is a shared axis for all subplot
            Bz_dif.tick_params(axis='x', which='major', labelsize=fig_xtick_fontsize)
            # Save plot.
            fig.savefig(data_pickle_dir + '/ACE_filter_process_BGSE_' + timeStart_str + '_' + timeEnd_str + '.eps',format='eps', dpi=500, bbox_inches='tight')
        
        # ========================================= Process VGSE =========================================
        if not VGSE_DataFrame.empty:
            print('\nProcessing VGSE...')
            # Keep original data.
            VGSE_DataFrame0 = VGSE_DataFrame.copy(deep=True)
            # Remove all data which fall outside three standard deviations.
            n_removed_Vx_total = 0
            n_removed_Vy_total = 0
            n_removed_Vz_total = 0

            n_std = 3.89 # 99.99%.
            Vx_std, Vy_std, Vz_std = VGSE_DataFrame.std(skipna=True, numeric_only=True)
            Vx_mean, Vy_mean, Vz_mean = VGSE_DataFrame.mean(skipna=True, numeric_only=True)
            Vx_remove = (VGSE_DataFrame['Vx']<(Vx_mean-n_std*Vx_std))|(VGSE_DataFrame['Vx']>(Vx_mean+n_std*Vx_std))
            Vy_remove = (VGSE_DataFrame['Vy']<(Vy_mean-n_std*Vy_std))|(VGSE_DataFrame['Vy']>(Vy_mean+n_std*Vy_std))
            Vz_remove = (VGSE_DataFrame['Vz']<(Vz_mean-n_std*Vz_std))|(VGSE_DataFrame['Vz']>(Vz_mean+n_std*Vz_std))
            VGSE_DataFrame['Vx'][Vx_remove] = np.nan
            VGSE_DataFrame['Vy'][Vy_remove] = np.nan
            VGSE_DataFrame['Vz'][Vz_remove] = np.nan

            Vx_lower_boundary = Vx_mean-n_std*Vx_std
            Vx_upper_boundary = Vx_mean+n_std*Vx_std
            Vy_lower_boundary = Vy_mean-n_std*Vy_std
            Vy_upper_boundary = Vy_mean+n_std*Vy_std
            Vz_lower_boundary = Vz_mean-n_std*Vz_std
            Vz_upper_boundary = Vz_mean+n_std*Vz_std

            n_removed_Vx = sum(Vx_remove)
            n_removed_Vy = sum(Vy_remove)
            n_removed_Vz = sum(Vz_remove)
            n_removed_Vx_total += n_removed_Vx
            n_removed_Vy_total += n_removed_Vy
            n_removed_Vz_total += n_removed_Vz

            if isVerbose:
                print('\nRemove all VGSE data which fall outside 3.89 standard deviations...')
                print('V_std:', Vx_std, Vy_std, Vz_std)
                print('V_mean:', Vx_mean, Vy_mean, Vz_mean)
                print('The VGSE Vx value range within 3 std is [{}, {}]'.format(Vx_lower_boundary, Vx_upper_boundary))
                print('The VGSE Vy value range within 3 std is [{}, {}]'.format(Vy_lower_boundary, Vy_upper_boundary))
                print('The VGSE Vz value range within 3 std is [{}, {}]'.format(Vz_lower_boundary, Vz_upper_boundary))
                print('In Vx, {} data has been removed!'.format(n_removed_Vx))
                print('In Vy, {} data has been removed!'.format(n_removed_Vy))
                print('In Vz, {} data has been removed!'.format(n_removed_Vz))
                print('Until now, in Vx, {} records have been removed!'.format(n_removed_Vx_total))
                print('Until now, in Vy, {} records have been removed!'.format(n_removed_Vy_total))
                print('Until now, in Vz, {} records have been removed!'.format(n_removed_Vz_total))
                print('\n')

            # Apply Butterworth filter to VGSE two times.
            for Wn in [0.05]: #[0.005, 0.05]
                if Wn==0.005:
                    print('Applying Butterworth filter with cutoff frequency = 0.005, remove large outliers...')
                if Wn==0.05:
                    print('Applying Butterworth filter with cutoff frequency = 0.05, remove spikes...')
                # Note that, sp.signal.butter cannot handle np.nan. Fill the nan before using it.
                VGSE_DataFrame.fillna(method='ffill', inplace=True) # Fill missing data, forward copy.
                VGSE_DataFrame.fillna(method='bfill', inplace=True) # Fill missing data, backward copy.
                # Create an empty DataFrame to store the filtered data.
                VGSE_LowPass = pd.DataFrame(index = VGSE_DataFrame.index, columns = ['Vx', 'Vy', 'Vz'])
                # Design the Buterworth filter.
                N  = 2    # Filter order
                B, A = sp.signal.butter(N, Wn, output='ba')
                # Apply the filter.
                try:
                    VGSE_LowPass['Vx'] = sp.signal.filtfilt(B, A, VGSE_DataFrame['Vx'])
                except:
                    print('Encounter exception, skip sp.signal.filtfilt operation!')
                    VGSE_LowPass['Vx'] = VGSE_DataFrame['Vx'].copy()
                    
                try:
                    VGSE_LowPass['Vy'] = sp.signal.filtfilt(B, A, VGSE_DataFrame['Vy'])
                except:
                    print('Encounter exception, skip sp.signal.filtfilt operation!')
                    VGSE_LowPass['Vy'] = VGSE_DataFrame['Vy'].copy()
                    
                try:
                    VGSE_LowPass['Vz'] = sp.signal.filtfilt(B, A, VGSE_DataFrame['Vz'])
                except:
                    print('Encounter exception, skip sp.signal.filtfilt operation!')
                    VGSE_LowPass['Vz'] = VGSE_DataFrame['Vz'].copy()
                
                # Calculate the difference between VGSE_LowPass and VGSE_DataFrame.
                VGSE_dif = pd.DataFrame(index = VGSE_DataFrame.index, columns = ['Vx', 'Vy', 'Vz']) # Generate empty DataFrame.
                VGSE_dif['Vx'] = VGSE_DataFrame['Vx'] - VGSE_LowPass['Vx']
                VGSE_dif['Vy'] = VGSE_DataFrame['Vy'] - VGSE_LowPass['Vy']
                VGSE_dif['Vz'] = VGSE_DataFrame['Vz'] - VGSE_LowPass['Vz']
                # Calculate the mean and standard deviation of VGSE_dif.
                Vx_dif_std, Vy_dif_std, Vz_dif_std = VGSE_dif.std(skipna=True, numeric_only=True)
                Vx_dif_mean, Vy_dif_mean, Vz_dif_mean = VGSE_dif.mean(skipna=True, numeric_only=True)
                # Set the values fall outside n*std to np.nan.
                n_dif_std = 3.89 # 99.99%
                Vx_remove = (VGSE_dif['Vx']<(Vx_dif_mean-n_dif_std*Vx_dif_std))|(VGSE_dif['Vx']>(Vx_dif_mean+n_dif_std*Vx_dif_std))
                Vy_remove = (VGSE_dif['Vy']<(Vy_dif_mean-n_dif_std*Vy_dif_std))|(VGSE_dif['Vy']>(Vy_dif_mean+n_dif_std*Vy_dif_std))
                Vz_remove = (VGSE_dif['Vz']<(Vz_dif_mean-n_dif_std*Vz_dif_std))|(VGSE_dif['Vz']>(Vz_dif_mean+n_dif_std*Vz_dif_std))
                VGSE_DataFrame['Vx'][Vx_remove] = np.nan
                VGSE_DataFrame['Vy'][Vy_remove] = np.nan
                VGSE_DataFrame['Vz'][Vz_remove] = np.nan
                
                Vx_dif_lower_boundary = Vx_dif_mean-n_dif_std*Vx_dif_std
                Vx_dif_upper_boundary = Vx_dif_mean+n_dif_std*Vx_dif_std
                Vy_dif_lower_boundary = Vy_dif_mean-n_dif_std*Vy_dif_std
                Vy_dif_upper_boundary = Vy_dif_mean+n_dif_std*Vy_dif_std
                Vz_dif_lower_boundary = Vz_dif_mean-n_dif_std*Vz_dif_std
                Vz_dif_upper_boundary = Vz_dif_mean+n_dif_std*Vz_dif_std
                
                n_removed_Vx = sum(Vx_remove)
                n_removed_Vy = sum(Vy_remove)
                n_removed_Vz = sum(Vz_remove)
                n_removed_Vx_total += n_removed_Vx
                n_removed_Vy_total += n_removed_Vy
                n_removed_Vz_total += n_removed_Vz
                
                if isVerbose:
                    print('V_dif_std:', Vx_dif_std, Vy_dif_std, Vz_dif_std)
                    print('V_dif_mean:', Vx_dif_mean, Vy_dif_mean, Vz_dif_mean)
                    print('The VGSE Vx_dif value range within 3.89 std is [{}, {}]'.format(Vx_dif_lower_boundary, Vx_dif_upper_boundary))
                    print('The VGSE Vy_dif value range within 3.89 std is [{}, {}]'.format(Vy_dif_lower_boundary, Vy_dif_upper_boundary))
                    print('The VGSE Vz_dif value range within 3.89 std is [{}, {}]'.format(Vz_dif_lower_boundary, Vz_dif_upper_boundary))
                    print('In Vx, this operation removed {} records!'.format(n_removed_Vx))
                    print('In Vy, this operation removed {} records!'.format(n_removed_Vy))
                    print('In Vz, this operation removed {} records!!'.format(n_removed_Vz))
                    print('Until now, in Vx, {} records have been removed!'.format(n_removed_Vx_total))
                    print('Until now, in Vy, {} records have been removed!'.format(n_removed_Vy_total))
                    print('Until now, in Vz, {} records have been removed!'.format(n_removed_Vz_total))
                    print('\n')
            
            # If plot filter process of VGSE or not.
            if isPlotFilterProcess:
                # Plot VGSE filter process.
                print('Plotting VGSE filtering process...')
                fig_line_width = 0.5
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
                Vx_plot.plot(VGSE_DataFrame0.index, VGSE_DataFrame0['Vx'].fillna(0),\
                             color = 'red', linewidth=fig_line_width, label='Vx_original') # Original data.
                Vx_plot.plot(VGSE_DataFrame.index, VGSE_DataFrame['Vx'].fillna(0),\
                             color = 'blue', linewidth=fig_line_width, label='Vx_processed') # Filtered data.
                Vx_plot.plot(VGSE_LowPass.index, VGSE_LowPass['Vx'].fillna(0),\
                             color = 'black', linewidth=fig_line_width, label='Vx_LowPass') # Low pass curve.
                Vx_plot.set_ylabel('Vx', fontsize=fig_ylabel_fontsize)
                Vx_plot.legend(loc='upper left',prop={'size':fig_legend_size})
                Vx_plot.tick_params(axis='y', which='major', labelsize=fig_ytick_fontsize)
                Vx_dif.plot(VGSE_dif.index, VGSE_dif['Vx'].fillna(0),\
                            color = 'green', linewidth=fig_line_width) # Difference data.
                Vx_dif.tick_params(axis='y', which='major', labelsize=fig_ytick_fontsize)
                Vx_dif.set_ylabel('Vx_dif', fontsize=fig_ylabel_fontsize)
                # Plotting Vy filter process.
                Vy_plot.plot(VGSE_DataFrame0.index, VGSE_DataFrame0['Vy'].fillna(0),\
                             color = 'red', linewidth=fig_line_width, label='Vy_original') # Original data.
                Vy_plot.plot(VGSE_DataFrame.index, VGSE_DataFrame['Vy'].fillna(0),\
                             color = 'blue', linewidth=fig_line_width, label='Vy_processed') # Filtered data.
                Vy_plot.plot(VGSE_LowPass.index, VGSE_LowPass['Vy'].fillna(0),\
                             color = 'black', linewidth=fig_line_width, label='Vy_LowPass') # Low pass curve.
                Vy_plot.set_ylabel('Vy', fontsize=fig_ylabel_fontsize)
                Vy_plot.legend(loc='upper left',prop={'size':fig_legend_size})
                Vy_plot.tick_params(axis='y', which='major', labelsize=fig_ytick_fontsize)
                Vy_dif.plot(VGSE_dif.index, VGSE_dif['Vy'].fillna(0),\
                            color = 'green', linewidth=fig_line_width) # Difference data.
                Vy_dif.tick_params(axis='y', which='major', labelsize=fig_ytick_fontsize)
                Vy_dif.set_ylabel('Vy_dif', fontsize=fig_ylabel_fontsize)
                # Plotting Vz filter process.
                Vz_plot.plot(VGSE_DataFrame0.index, VGSE_DataFrame0['Vz'].fillna(0),\
                             color = 'red', linewidth=fig_line_width, label='Vz_original') # Original data.
                Vz_plot.plot(VGSE_DataFrame.index, VGSE_DataFrame['Vz'].fillna(0),\
                             color = 'blue', linewidth=fig_line_width, label='Vz_processed') # Filtered data.
                Vz_plot.plot(VGSE_LowPass.index, VGSE_LowPass['Vz'].fillna(0),\
                             color = 'black', linewidth=fig_line_width, label='Vz_LowPass') # Low pass curve.
                Vz_plot.set_ylabel('Vz', fontsize=fig_ylabel_fontsize)
                Vz_plot.legend(loc='upper left',prop={'size':fig_legend_size})
                Vz_plot.tick_params(axis='y', which='major', labelsize=fig_ytick_fontsize)
                Vz_dif.plot(VGSE_dif.index, VGSE_dif['Vz'].fillna(0),\
                            color = 'green', linewidth=fig_line_width) # Difference data.
                Vz_dif.tick_params(axis='y', which='major', labelsize=fig_ytick_fontsize)
                Vz_dif.set_ylabel('Vz_dif', fontsize=fig_ylabel_fontsize)
                # This is a shared axis for all subplot
                Vz_dif.tick_params(axis='x', which='major', labelsize=fig_xtick_fontsize)
                # Save plot.
                fig.savefig(data_pickle_dir + '/ACE_filter_process_VGSE_' + timeStart_str + '_' + timeEnd_str + '.eps',format='eps', dpi=500, bbox_inches='tight')
        else:
            # Keep original data.
            VGSE_DataFrame0 = VGSE_DataFrame.copy(deep=True)
        # ========================================= Process Np missing value =========================================
        if not Np_DataFrame.empty:
            print('\nProcessing Np...')
            # Keep original data.
            Np_DataFrame0 = Np_DataFrame.copy(deep=True)
            #print('Np_DataFrame.shape = {}'.format(Np_DataFrame.shape))

            # Remove all data which fall outside 4 standard deviations.
            n_removed_Np_total = 0
            print('Remove all Np data which fall outside 3.89 standard deviations...')
            n_std = 3.89 # 99.99%.
            Np_std = Np_DataFrame.std(skipna=True, numeric_only=True)[0]
            Np_mean = Np_DataFrame.mean(skipna=True, numeric_only=True)[0]
            Np_remove = (Np_DataFrame['Np']<(Np_mean-n_std*Np_std))|(Np_DataFrame['Np']>(Np_mean+n_std*Np_std))
            Np_DataFrame['Np'][Np_remove] = np.nan

            Np_lower_boundary = Np_mean-n_std*Np_std
            Np_upper_boundary = Np_mean+n_std*Np_std

            n_removed_Np = sum(Np_remove)
            n_removed_Np_total += n_removed_Np

            if isVerbose:
                print('Np_std:', Np_std)
                print('Np_mean:', Np_mean)
                print('The Np value range within {} std is [{}, {}]'.format(n_std, Np_lower_boundary, Np_upper_boundary))
                print('In Np, {} data has been removed!'.format(n_removed_Np))
                print('Till now, in Np, {} records have been removed!'.format(n_removed_Np_total))
                print('\n')

            # Apply Butterworth filter to Np.
            for Wn in [0.05, 0.7]: # Np
                if Wn==0.05:
                    print('Applying Butterworth filter with cutoff frequency = 0.05, remove large outliers...')
                if Wn==0.7:
                    print('Applying Butterworth filter with cutoff frequency = 0.7, remove spikes...')
                # Note that, sp.signal.butter cannot handle np.nan. Fill the nan before using it.
                Np_DataFrame.fillna(method='ffill', inplace=True) # Fill missing data, forward copy.
                Np_DataFrame.fillna(method='bfill', inplace=True) # Fill missing data, backward copy.
                # Create an empty DataFrame to store the filtered data.
                Np_LowPass = pd.DataFrame(index = Np_DataFrame.index, columns = ['Np'])
                # Design the Buterworth filter.
                N  = 2    # Filter order
                B, A = sp.signal.butter(N, Wn, output='ba')
                # Apply the filter.
                try:
                    Np_LowPass['Np'] = sp.signal.filtfilt(B, A, Np_DataFrame['Np'])
                except:
                    print('Encounter exception, skip sp.signal.filtfilt operation!')
                    Np_LowPass['Np'] = Np_DataFrame['Np'].copy()
                # Calculate the difference between Np_LowPass and Np_DataFrame.
                Np_dif = pd.DataFrame(index = Np_DataFrame.index, columns = ['Np']) # Generate empty DataFrame.
                Np_dif['Np'] = Np_DataFrame['Np'] - Np_LowPass['Np']
                # Calculate the mean and standard deviation of Np_dif. Np_dif_std is a Series object, so [0] is added.
                Np_dif_std = Np_dif.std(skipna=True, numeric_only=True)[0]
                Np_dif_mean = Np_dif.mean(skipna=True, numeric_only=True)[0]
                # Set the values fall outside n*std to np.nan.
                n_dif_std = 3.89 # 99.99%.
                Np_remove = (Np_dif['Np']<(Np_dif_mean-n_dif_std*Np_dif_std))|(Np_dif['Np']>(Np_dif_mean+n_dif_std*Np_dif_std))
                Np_DataFrame[Np_remove] = np.nan
                
                Np_dif_lower_boundary = Np_dif_mean-n_dif_std*Np_dif_std
                Np_dif_upper_boundary = Np_dif_mean+n_dif_std*Np_dif_std
                
                n_removed_Np = sum(Np_remove)
                n_removed_Np_total += n_removed_Np
                
                if isVerbose:
                    print('Np_dif_std:', Np_dif_std)
                    print('Np_dif_mean:', Np_dif_mean)
                    print('The Np_dif value range within {} std is [{}, {}]'.format(n_std, Np_dif_lower_boundary, Np_dif_upper_boundary))
                    print('In Np, this operation removed {} records!'.format(n_removed_Np))
                    print('Till now, in Np, {} records have been removed!'.format(n_removed_Np_total))
        else:
            # Keep original data.
            Np_DataFrame0 = Np_DataFrame.copy(deep=True)   
        # ========================================= Process Alpha2Proton_ratio missing value =========================================
        if not Alpha2Proton_ratio_DataFrame.empty:
            print('\nProcessing Alpha2Proton_ratio...')
            # Keep original data.
            Alpha2Proton_ratio_DataFrame0 = Alpha2Proton_ratio_DataFrame.copy(deep=True)
            #print('Alpha2Proton_ratio_DataFrame.shape = {}'.format(Alpha2Proton_ratio_DataFrame.shape))

            # Remove all data which fall outside 4 standard deviations.
            n_removed_Alpha2Proton_ratio_total = 0
            print('Remove all Alpha2Proton_ratio data which fall outside 3.89 standard deviations...')
            n_std = 3.89 # 99.99%.
            Alpha2Proton_ratio_std = Alpha2Proton_ratio_DataFrame.std(skipna=True, numeric_only=True)[0]
            Alpha2Proton_ratio_mean = Alpha2Proton_ratio_DataFrame.mean(skipna=True, numeric_only=True)[0]
            Alpha2Proton_ratio_remove = (Alpha2Proton_ratio_DataFrame['Alpha2Proton_ratio']<(Alpha2Proton_ratio_mean-n_std*Alpha2Proton_ratio_std))|(Alpha2Proton_ratio_DataFrame['Alpha2Proton_ratio']>(Alpha2Proton_ratio_mean+n_std*Alpha2Proton_ratio_std))
            Alpha2Proton_ratio_DataFrame['Alpha2Proton_ratio'][Alpha2Proton_ratio_remove] = np.nan

            Alpha2Proton_ratio_lower_boundary = Alpha2Proton_ratio_mean-n_std*Alpha2Proton_ratio_std
            Alpha2Proton_ratio_upper_boundary = Alpha2Proton_ratio_mean+n_std*Alpha2Proton_ratio_std

            n_removed_Alpha2Proton_ratio = sum(Alpha2Proton_ratio_remove)
            n_removed_Alpha2Proton_ratio_total += n_removed_Alpha2Proton_ratio

            if isVerbose:
                print('Alpha2Proton_ratio_std:', Alpha2Proton_ratio_std)
                print('Alpha2Proton_ratio_mean:', Alpha2Proton_ratio_mean)
                print('The Alpha2Proton_ratio value range within {} std is [{}, {}]'.format(n_std, Alpha2Proton_ratio_lower_boundary, Alpha2Proton_ratio_upper_boundary))
                print('In Alpha2Proton_ratio, {} data has been removed!'.format(n_removed_Alpha2Proton_ratio))
                print('Till now, in Alpha2Proton_ratio, {} records have been removed!'.format(n_removed_Alpha2Proton_ratio_total))
                print('\n')

            # Apply Butterworth filter to Alpha2Proton_ratio.
            for Wn in [0.05, 0.7]: # Alpha2Proton_ratio
                if Wn==0.05:
                    print('Applying Butterworth filter with cutoff frequency = 0.05, remove large outliers...')
                if Wn==0.7:
                    print('Applying Butterworth filter with cutoff frequency = 0.7, remove spikes...')
                # Note that, sp.signal.butter cannot handle np.nan. Fill the nan before using it.
                Alpha2Proton_ratio_DataFrame.fillna(method='ffill', inplace=True) # Fill missing data, forward copy.
                Alpha2Proton_ratio_DataFrame.fillna(method='bfill', inplace=True) # Fill missing data, backward copy.
                # Create an empty DataFrame to store the filtered data.
                Alpha2Proton_ratio_LowPass = pd.DataFrame(index = Alpha2Proton_ratio_DataFrame.index, columns = ['Alpha2Proton_ratio'])
                # Design the Buterworth filter.
                N  = 2    # Filter order
                B, A = sp.signal.butter(N, Wn, output='ba')
                # Apply the filter.
                try:
                    Alpha2Proton_ratio_LowPass['Alpha2Proton_ratio'] = sp.signal.filtfilt(B, A, Alpha2Proton_ratio_DataFrame['Alpha2Proton_ratio'])
                except:
                    print('Encounter exception, skip sp.signal.filtfilt operation!')
                    Alpha2Proton_ratio_LowPass['Alpha2Proton_ratio'] = Alpha2Proton_ratio_DataFrame['Alpha2Proton_ratio'].copy()
                # Calculate the difference between Alpha2Proton_ratio_LowPass and Alpha2Proton_ratio_DataFrame.
                Alpha2Proton_ratio_dif = pd.DataFrame(index = Alpha2Proton_ratio_DataFrame.index, columns = ['Alpha2Proton_ratio']) # Generate empty DataFrame.
                Alpha2Proton_ratio_dif['Alpha2Proton_ratio'] = Alpha2Proton_ratio_DataFrame['Alpha2Proton_ratio'] - Alpha2Proton_ratio_LowPass['Alpha2Proton_ratio']
                # Calculate the mean and standard deviation of Alpha2Proton_ratio_dif. Alpha2Proton_ratio_dif_std is a Series object, so [0] is added.
                Alpha2Proton_ratio_dif_std = Alpha2Proton_ratio_dif.std(skipna=True, numeric_only=True)[0]
                Alpha2Proton_ratio_dif_mean = Alpha2Proton_ratio_dif.mean(skipna=True, numeric_only=True)[0]
                # Set the values fall outside n*std to np.nan.
                n_dif_std = 3.89 # 99.99%.
                Alpha2Proton_ratio_remove = (Alpha2Proton_ratio_dif['Alpha2Proton_ratio']<(Alpha2Proton_ratio_dif_mean-n_dif_std*Alpha2Proton_ratio_dif_std))|(Alpha2Proton_ratio_dif['Alpha2Proton_ratio']>(Alpha2Proton_ratio_dif_mean+n_dif_std*Alpha2Proton_ratio_dif_std))
                Alpha2Proton_ratio_DataFrame[Alpha2Proton_ratio_remove] = np.nan
                
                Alpha2Proton_ratio_dif_lower_boundary = Alpha2Proton_ratio_dif_mean-n_dif_std*Alpha2Proton_ratio_dif_std
                Alpha2Proton_ratio_dif_upper_boundary = Alpha2Proton_ratio_dif_mean+n_dif_std*Alpha2Proton_ratio_dif_std
                
                n_removed_Alpha2Proton_ratio = sum(Alpha2Proton_ratio_remove)
                n_removed_Alpha2Proton_ratio_total += n_removed_Alpha2Proton_ratio
                
                if isVerbose:
                    print('Alpha2Proton_ratio_dif_std:', Alpha2Proton_ratio_dif_std)
                    print('Alpha2Proton_ratio_dif_mean:', Alpha2Proton_ratio_dif_mean)
                    print('The Alpha2Proton_ratio_dif value range within {} std is [{}, {}]'.format(n_std, Alpha2Proton_ratio_dif_lower_boundary, Alpha2Proton_ratio_dif_upper_boundary))
                    print('In Alpha2Proton_ratio, this operation removed {} records!'.format(n_removed_Alpha2Proton_ratio))
                    print('Till now, in Alpha2Proton_ratio, {} records have been removed!'.format(n_removed_Alpha2Proton_ratio_total))
        else:
            # Keep original data.
            Alpha2Proton_ratio_DataFrame0 = Alpha2Proton_ratio_DataFrame.copy(deep=True)
        # ========================================= Process Tp value =========================================
        if not Tp_DataFrame.empty:    
            print('\nProcessing Tp...')
            # Keep original data.
            Tp_DataFrame0 = Tp_DataFrame.copy(deep=True)

            # Remove all data which fall outside 3.89 standard deviations.
            n_removed_Tp_total = 0
            n_std = 3.89 # 99.99%.
            print('Remove all Tp data which fall outside {} standard deviations...'.format(n_std))
            Tp_std = Tp_DataFrame.std(skipna=True, numeric_only=True)[0]
            Tp_mean = Tp_DataFrame.mean(skipna=True, numeric_only=True)[0]
            Tp_remove = (Tp_DataFrame['Tp']<(Tp_mean-n_std*Tp_std))|(Tp_DataFrame['Tp']>(Tp_mean+n_std*Tp_std))
            Tp_DataFrame['Tp'][Tp_remove] = np.nan

            Tp_lower_boundary = Tp_mean-n_std*Tp_std
            Tp_upper_boundary = Tp_mean+n_std*Tp_std

            n_removed_Tp = sum(Tp_remove)
            n_removed_Tp_total += n_removed_Tp

            if isVerbose:
                print('Tp_std:', Tp_std)
                print('Tp_mean:', Tp_mean)
                print('The Tp value range within {} std is [{}, {}]'.format(n_std, Tp_lower_boundary, Tp_upper_boundary))
                print('In Tp, {} data has been removed!'.format(n_removed_Tp))
                print('Till now, in Tp, {} records have been removed!'.format(n_removed_Tp_total))
                print('\n')

            # Apply Butterworth filter to Tp.
            for Wn in [0.05, 0.7]: # Tp
                if Wn==0.05:
                    print('Applying Butterworth filter with cutoff frequency = 0.05, remove large outliers...')
                if Wn==0.7:
                    print('Applying Butterworth filter with cutoff frequency = 0.7, remove spikes...')
                # Note that, sp.signal.butter cannot handle np.nan. Fill the nan before using it.
                Tp_DataFrame.fillna(method='ffill', inplace=True) # Fill missing data, forward copy.
                Tp_DataFrame.fillna(method='bfill', inplace=True) # Fill missing data, backward copy.
                # Create an empty DataFrame to store the filtered data.
                Tp_LowPass = pd.DataFrame(index = Tp_DataFrame.index, columns = ['Tp'])
                # Design the Buterworth filter.
                N  = 2    # Filter order
                B, A = sp.signal.butter(N, Wn, output='ba')
                # Apply the filter.
                try:
                    Tp_LowPass['Tp'] = sp.signal.filtfilt(B, A, Tp_DataFrame['Tp'])
                except:
                    print('Encounter exception, skip sp.signal.filtfilt operation!')
                    Tp_LowPass['Tp'] = Tp_DataFrame['Tp'].copy()
                # Calculate the difference between Tp_LowPass and Tp_DataFrame.
                Tp_dif = pd.DataFrame(index = Tp_DataFrame.index, columns = ['Tp']) # Generate empty DataFrame.
                Tp_dif['Tp'] = Tp_DataFrame['Tp'] - Tp_LowPass['Tp']
                # Calculate the mean and standard deviation of Tp_dif. Tp_dif_std is a Series object, so [0] is added.
                Tp_dif_std = Tp_dif.std(skipna=True, numeric_only=True)[0]
                Tp_dif_mean = Tp_dif.mean(skipna=True, numeric_only=True)[0]
                # Set the values fall outside n*std to np.nan.
                n_dif_std = 3.89 # 99.99%.
                Tp_remove = (Tp_dif['Tp']<(Tp_dif_mean-n_dif_std*Tp_dif_std))|(Tp_dif['Tp']>(Tp_dif_mean+n_dif_std*Tp_dif_std))
                Tp_DataFrame[Tp_remove] = np.nan
                
                Tp_dif_lower_boundary = Tp_dif_mean-n_dif_std*Tp_dif_std
                Tp_dif_upper_boundary = Tp_dif_mean+n_dif_std*Tp_dif_std
                
                n_removed_Tp = sum(Tp_remove)
                n_removed_Tp_total += n_removed_Tp
                
                if isVerbose:
                    print('Tp_dif_std:', Tp_dif_std)
                    print('Tp_dif_mean:', Tp_dif_mean)
                    print('The Tp_dif value range within {} std is [{}, {}]'.format(n_dif_std, Tp_dif_lower_boundary, Tp_dif_upper_boundary))
                    print('In Tp, this operation removed {} records!'.format(n_removed_Tp))
                    print('Till now, in Tp, {} records have been removed!'.format(n_removed_Tp_total))
                    print('\n')
            
            # There is no need to convert Tp. Tp from ACE is already in Kelvin. Tp from ACE is radial component of the proton temperature.
            # ACE has no Te.
        else:
            # Keep original data.
            Np_DataFrame0 = Np_DataFrame.copy(deep=True)  

        # ===================================== Plot Np, Tp filter process =====================================
        # If plot filter process of Np, Tp or not.
        if isPlotFilterProcess:
            fig_line_width = 0.5
            fig_ylabel_fontsize = 9
            fig_xtick_fontsize = 8
            fig_ytick_fontsize = 8
            fig_legend_size = 5
            fig,ax = plt.subplots(6,1, sharex=True,figsize=(18, 12))
            Np_plot = ax[0]
            Np_dif_plot = ax[1]
            Alpha2Proton_ratio_plot = ax[2]
            Alpha2Proton_ratio_dif_plot = ax[3]
            Tp_plot = ax[4]
            Tp_dif_plot = ax[5]

            # Plotting Np filter process.
            print('Plotting Np filtering process...')
            Np_plot.plot(Np_DataFrame0.index, Np_DataFrame0['Np'].fillna(0),\
                         color = 'red', linewidth=fig_line_width, label='Np_original') # Original data.
            Np_plot.plot(Np_DataFrame.index, Np_DataFrame['Np'].fillna(0),\
                         color = 'blue', linewidth=fig_line_width, label='Np_processed') # Filtered data.
            Np_plot.plot(Np_LowPass.index, Np_LowPass['Np'].fillna(0),\
                         color = 'black', linewidth=fig_line_width, label='Np_LowPass') # Low pass curve.
            Np_plot.set_ylabel('Np', fontsize=fig_ylabel_fontsize)
            Np_plot.legend(loc='upper left',prop={'size':fig_legend_size})
            Np_plot.tick_params(axis='y', which='major', labelsize=fig_ytick_fontsize)
            Np_dif_plot.plot(Np_dif.index, Np_dif['Np'].fillna(0), color = 'green', linewidth=fig_line_width) # Difference data.
            Np_dif_plot.tick_params(axis='y', which='major', labelsize=fig_ytick_fontsize)
            Np_dif_plot.set_ylabel('Np_dif', fontsize=fig_ylabel_fontsize)
            
            # Plotting Alpha2Proton_ratio filter process.
            print('Plotting Alpha2Proton_ratio filtering process...')
            Alpha2Proton_ratio_plot.plot(Alpha2Proton_ratio_DataFrame0.index, Alpha2Proton_ratio_DataFrame0['Alpha2Proton_ratio'].fillna(0),\
                         color = 'red', linewidth=fig_line_width, label='Alpha2Proton_ratio_original') # Original data.
            Alpha2Proton_ratio_plot.plot(Alpha2Proton_ratio_DataFrame.index, Alpha2Proton_ratio_DataFrame['Alpha2Proton_ratio'].fillna(0),\
                         color = 'blue', linewidth=fig_line_width, label='Alpha2Proton_ratio_processed') # Filtered data.
            Alpha2Proton_ratio_plot.plot(Alpha2Proton_ratio_LowPass.index, Alpha2Proton_ratio_LowPass['Alpha2Proton_ratio'].fillna(0),\
                         color = 'black', linewidth=fig_line_width, label='Alpha2Proton_ratio_LowPass') # Low pass curve.
            Alpha2Proton_ratio_plot.set_ylabel('Alpha2Proton_ratio', fontsize=fig_ylabel_fontsize)
            Alpha2Proton_ratio_plot.legend(loc='upper left',prop={'size':fig_legend_size})
            Alpha2Proton_ratio_plot.tick_params(axis='y', which='major', labelsize=fig_ytick_fontsize)
            Alpha2Proton_ratio_dif_plot.plot(Alpha2Proton_ratio_dif.index, Alpha2Proton_ratio_dif['Alpha2Proton_ratio'].fillna(0), color = 'green', linewidth=fig_line_width) # Difference data.
            Alpha2Proton_ratio_dif_plot.tick_params(axis='y', which='major', labelsize=fig_ytick_fontsize)
            Alpha2Proton_ratio_dif_plot.set_ylabel('Alpha2Proton_ratio_dif', fontsize=fig_ylabel_fontsize)
            
            # Plotting Tp filter process.
            print('Plotting Tp filtering process...')
            Tp_plot.plot(Tp_DataFrame0.index, Tp_DataFrame0['Tp'].fillna(0),\
                         color = 'red', linewidth=fig_line_width, label='Tp_original') # Original data.
            Tp_plot.plot(Tp_DataFrame.index, Tp_DataFrame['Tp'].fillna(0),\
                         color = 'blue', linewidth=fig_line_width, label='Tp_processed') # Filtered data.
            Tp_plot.plot(Tp_LowPass.index, Tp_LowPass['Tp'].fillna(0),\
                         color = 'black', linewidth=fig_line_width, label='Tp_LowPass') # Low pass curve.
            Tp_plot.set_ylabel('Tp', fontsize=fig_ylabel_fontsize)
            Tp_plot.legend(loc='upper left',prop={'size':fig_legend_size})
            Tp_plot.tick_params(axis='y', which='major', labelsize=fig_ytick_fontsize)
            Tp_dif_plot.plot(Tp_dif.index, Tp_dif['Tp'].fillna(0),\
                             color = 'green', linewidth=fig_line_width) # Difference data.
            Tp_dif_plot.tick_params(axis='y', which='major', labelsize=fig_ytick_fontsize)
            Tp_dif_plot.set_ylabel('Tp_dif', fontsize=fig_ylabel_fontsize)
            # This is a shared axis for all subplot
            Tp_dif_plot.tick_params(axis='x', which='major', labelsize=fig_xtick_fontsize)
            # Save plot.
            fig.savefig(data_pickle_dir + '/ACE_filter_process_Np_AlphaRatio_Tp_VGSE_' + timeStart_str + '_' + timeEnd_str + '.eps',format='eps', dpi=500, bbox_inches='tight')

        # ========================================= Process LEMS value =========================================
        if not LEMS_DataFrame.empty:
            print('\nProcessing LEMS value...')
            # Keep original data.
            LEMS_DataFrame0 = LEMS_DataFrame.copy(deep=True)
            
            n_removed_LEMS30_P1_total = 0
            n_removed_LEMS30_P2_total = 0
            n_removed_LEMS30_P3_total = 0
            n_removed_LEMS30_P4_total = 0
            n_removed_LEMS30_P5_total = 0
            n_removed_LEMS30_P6_total = 0
            n_removed_LEMS30_P7_total = 0
            n_removed_LEMS30_P8_total = 0
            n_removed_LEMS120_P1_total = 0
            n_removed_LEMS120_P2_total = 0
            n_removed_LEMS120_P3_total = 0
            n_removed_LEMS120_P4_total = 0
            n_removed_LEMS120_P5_total = 0
            n_removed_LEMS120_P6_total = 0
            n_removed_LEMS120_P7_total = 0
            n_removed_LEMS120_P8_total = 0
            
            n_dif_std = 3 # 99.73%
            
            # Apply Butterworth filter to LEMS data.
            for Wn in [0.01, 0.03, 0.15]:# 
                if Wn==0.01:
                    print('Applying Butterworth filter with cutoff frequency = {}, remove large outliers...'.format(Wn))
                    n_dif_std = 3.89 # 99.99%
                elif Wn==0.03:
                    print('Applying Butterworth filter with cutoff frequency = {}, remove large outliers...'.format(Wn))
                    n_dif_std = 3.89 # 99.99%
                elif Wn==0.15:
                    print('Applying Butterworth filter with cutoff frequency = {}, remove spikes...'.format(Wn))
                    n_dif_std = 3 # 99.73%
                
                # Note that, sp.signal.butter cannot handle np.nan. Fill the nan before using it.
                LEMS_DataFrame.fillna(method='ffill', inplace=True) # Fill missing data, forward copy.
                LEMS_DataFrame.fillna(method='bfill', inplace=True) # Fill missing data, backward copy.
                # Create an empty DataFrame to store the filtered data.
                LEMS_LowPass = pd.DataFrame(index = LEMS_DataFrame.index, columns = ['LEMS30_P1','LEMS30_P2','LEMS30_P3','LEMS30_P4','LEMS30_P5','LEMS30_P6','LEMS30_P7','LEMS30_P8','LEMS120_P1','LEMS120_P2','LEMS120_P3','LEMS120_P4','LEMS120_P5','LEMS120_P6','LEMS120_P7','LEMS120_P8'])
                # Design the Buterworth filter.
                N  = 2    # Filter order
                B, A = sp.signal.butter(N, Wn, output='ba')
                # Apply the filter.
                try:
                    LEMS_LowPass['LEMS30_P1'] = sp.signal.filtfilt(B, A, LEMS_DataFrame['LEMS30_P1'])
                except:
                    print('Encounter exception, skip sp.signal.filtfilt operation!')
                    LEMS_LowPass['LEMS30_P1'] = LEMS_DataFrame['LEMS30_P1'].copy()
                
                try:
                    LEMS_LowPass['LEMS30_P2'] = sp.signal.filtfilt(B, A, LEMS_DataFrame['LEMS30_P2'])
                except:
                    print('Encounter exception, skip sp.signal.filtfilt operation!')
                    LEMS_LowPass['LEMS30_P2'] = LEMS_DataFrame['LEMS30_P2'].copy()
                
                try:
                    LEMS_LowPass['LEMS30_P3'] = sp.signal.filtfilt(B, A, LEMS_DataFrame['LEMS30_P3'])
                except:
                    print('Encounter exception, skip sp.signal.filtfilt operation!')
                    LEMS_LowPass['LEMS30_P3'] = LEMS_DataFrame['LEMS30_P3'].copy()
                
                try:
                    LEMS_LowPass['LEMS30_P4'] = sp.signal.filtfilt(B, A, LEMS_DataFrame['LEMS30_P4'])
                except:
                    print('Encounter exception, skip sp.signal.filtfilt operation!')
                    LEMS_LowPass['LEMS30_P4'] = LEMS_DataFrame['LEMS30_P4'].copy()
                
                try:
                    LEMS_LowPass['LEMS30_P5'] = sp.signal.filtfilt(B, A, LEMS_DataFrame['LEMS30_P5'])
                except:
                    print('Encounter exception, skip sp.signal.filtfilt operation!')
                    LEMS_LowPass['LEMS30_P5'] = LEMS_DataFrame['LEMS30_P5'].copy()
                
                try:
                    LEMS_LowPass['LEMS30_P6'] = sp.signal.filtfilt(B, A, LEMS_DataFrame['LEMS30_P6'])
                except:
                    print('Encounter exception, skip sp.signal.filtfilt operation!')
                    LEMS_LowPass['LEMS30_P6'] = LEMS_DataFrame['LEMS30_P6'].copy()
                
                try:
                    LEMS_LowPass['LEMS30_P7'] = sp.signal.filtfilt(B, A, LEMS_DataFrame['LEMS30_P7'])
                except:
                    print('Encounter exception, skip sp.signal.filtfilt operation!')
                    LEMS_LowPass['LEMS30_P7'] = LEMS_DataFrame['LEMS30_P7'].copy()
                
                try:
                    LEMS_LowPass['LEMS30_P8'] = sp.signal.filtfilt(B, A, LEMS_DataFrame['LEMS30_P8'])
                except:
                    print('Encounter exception, skip sp.signal.filtfilt operation!')
                    LEMS_LowPass['LEMS30_P8'] = LEMS_DataFrame['LEMS30_P8'].copy()
                
                try:
                    LEMS_LowPass['LEMS120_P1'] = sp.signal.filtfilt(B, A, LEMS_DataFrame['LEMS120_P1'])
                except:
                    print('Encounter exception, skip sp.signal.filtfilt operation!')
                    LEMS_LowPass['LEMS120_P1'] = LEMS_DataFrame['LEMS120_P1'].copy()
                
                try:
                    LEMS_LowPass['LEMS120_P2'] = sp.signal.filtfilt(B, A, LEMS_DataFrame['LEMS120_P2'])
                except:
                    print('Encounter exception, skip sp.signal.filtfilt operation!')
                    LEMS_LowPass['LEMS120_P2'] = LEMS_DataFrame['LEMS120_P2'].copy()
                
                try:
                    LEMS_LowPass['LEMS120_P3'] = sp.signal.filtfilt(B, A, LEMS_DataFrame['LEMS120_P3'])
                except:
                    print('Encounter exception, skip sp.signal.filtfilt operation!')
                    LEMS_LowPass['LEMS120_P3'] = LEMS_DataFrame['LEMS120_P3'].copy()
                
                try:
                    LEMS_LowPass['LEMS120_P4'] = sp.signal.filtfilt(B, A, LEMS_DataFrame['LEMS120_P4'])
                except:
                    print('Encounter exception, skip sp.signal.filtfilt operation!')
                    LEMS_LowPass['LEMS120_P4'] = LEMS_DataFrame['LEMS120_P4'].copy()
                
                try:
                    LEMS_LowPass['LEMS120_P5'] = sp.signal.filtfilt(B, A, LEMS_DataFrame['LEMS120_P5'])
                except:
                    print('Encounter exception, skip sp.signal.filtfilt operation!')
                    LEMS_LowPass['LEMS120_P5'] = LEMS_DataFrame['LEMS120_P5'].copy()
                
                try:
                    LEMS_LowPass['LEMS120_P6'] = sp.signal.filtfilt(B, A, LEMS_DataFrame['LEMS120_P6'])
                except:
                    print('Encounter exception, skip sp.signal.filtfilt operation!')
                    LEMS_LowPass['LEMS120_P6'] = LEMS_DataFrame['LEMS120_P6'].copy()
                
                try:
                    LEMS_LowPass['LEMS120_P7'] = sp.signal.filtfilt(B, A, LEMS_DataFrame['LEMS120_P7'])
                except:
                    print('Encounter exception, skip sp.signal.filtfilt operation!')
                    LEMS_LowPass['LEMS120_P7'] = LEMS_DataFrame['LEMS120_P7'].copy()
                
                try:
                    LEMS_LowPass['LEMS120_P8'] = sp.signal.filtfilt(B, A, LEMS_DataFrame['LEMS120_P8'])
                except:
                    print('Encounter exception, skip sp.signal.filtfilt operation!')
                    LEMS_LowPass['LEMS120_P8'] = LEMS_DataFrame['LEMS120_P8'].copy()
                
                # Calculate the difference between BGSE_LowPass and BGSE_DataFrame.
                LEMS_dif = pd.DataFrame(index = LEMS_DataFrame.index, columns = ['LEMS30_P1','LEMS30_P2','LEMS30_P3','LEMS30_P4','LEMS30_P5','LEMS30_P6','LEMS30_P7','LEMS30_P8','LEMS120_P1','LEMS120_P2','LEMS120_P3','LEMS120_P4','LEMS120_P5','LEMS120_P6','LEMS120_P7','LEMS120_P8']) # Generate empty DataFrame.
                LEMS_dif['LEMS30_P1'] = LEMS_DataFrame['LEMS30_P1'] - LEMS_LowPass['LEMS30_P1']
                LEMS_dif['LEMS30_P2'] = LEMS_DataFrame['LEMS30_P2'] - LEMS_LowPass['LEMS30_P2']
                LEMS_dif['LEMS30_P3'] = LEMS_DataFrame['LEMS30_P3'] - LEMS_LowPass['LEMS30_P3']
                LEMS_dif['LEMS30_P4'] = LEMS_DataFrame['LEMS30_P4'] - LEMS_LowPass['LEMS30_P4']
                LEMS_dif['LEMS30_P5'] = LEMS_DataFrame['LEMS30_P5'] - LEMS_LowPass['LEMS30_P5']
                LEMS_dif['LEMS30_P6'] = LEMS_DataFrame['LEMS30_P6'] - LEMS_LowPass['LEMS30_P6']
                LEMS_dif['LEMS30_P7'] = LEMS_DataFrame['LEMS30_P7'] - LEMS_LowPass['LEMS30_P7']
                LEMS_dif['LEMS30_P8'] = LEMS_DataFrame['LEMS30_P8'] - LEMS_LowPass['LEMS30_P8']
                LEMS_dif['LEMS120_P1'] = LEMS_DataFrame['LEMS120_P1'] - LEMS_LowPass['LEMS120_P1']
                LEMS_dif['LEMS120_P2'] = LEMS_DataFrame['LEMS120_P2'] - LEMS_LowPass['LEMS120_P2']
                LEMS_dif['LEMS120_P3'] = LEMS_DataFrame['LEMS120_P3'] - LEMS_LowPass['LEMS120_P3']
                LEMS_dif['LEMS120_P4'] = LEMS_DataFrame['LEMS120_P4'] - LEMS_LowPass['LEMS120_P4']
                LEMS_dif['LEMS120_P5'] = LEMS_DataFrame['LEMS120_P5'] - LEMS_LowPass['LEMS120_P5']
                LEMS_dif['LEMS120_P6'] = LEMS_DataFrame['LEMS120_P6'] - LEMS_LowPass['LEMS120_P6']
                LEMS_dif['LEMS120_P7'] = LEMS_DataFrame['LEMS120_P7'] - LEMS_LowPass['LEMS120_P7']
                LEMS_dif['LEMS120_P8'] = LEMS_DataFrame['LEMS120_P8'] - LEMS_LowPass['LEMS120_P8']
                
                # Calculate the mean and standard deviation of BGSE_dif.
                LEMS30_P1_dif_std, LEMS30_P2_dif_std, LEMS30_P3_dif_std, LEMS30_P4_dif_std, LEMS30_P5_dif_std, LEMS30_P6_dif_std, LEMS30_P7_dif_std, LEMS30_P8_dif_std, LEMS120_P1_dif_std, LEMS120_P2_dif_std, LEMS120_P3_dif_std, LEMS120_P4_dif_std, LEMS120_P5_dif_std, LEMS120_P6_dif_std, LEMS120_P7_dif_std, LEMS120_P8_dif_std = LEMS_dif.std(skipna=True, numeric_only=True)
                LEMS30_P1_dif_mean, LEMS30_P2_dif_mean, LEMS30_P3_dif_mean, LEMS30_P4_dif_mean, LEMS30_P5_dif_mean, LEMS30_P6_dif_mean, LEMS30_P7_dif_mean, LEMS30_P8_dif_mean, LEMS120_P1_dif_mean, LEMS120_P2_dif_mean, LEMS120_P3_dif_mean, LEMS120_P4_dif_mean, LEMS120_P5_dif_mean, LEMS120_P6_dif_mean, LEMS120_P7_dif_mean, LEMS120_P8_dif_mean = LEMS_dif.mean(skipna=True, numeric_only=True)
                # Set the values fall outside n*std to np.nan.
                LEMS30_P1_remove = (LEMS_dif['LEMS30_P1']<(LEMS30_P1_dif_mean-n_dif_std*LEMS30_P1_dif_std))|(LEMS_dif['LEMS30_P1']>(LEMS30_P1_dif_mean+n_dif_std*LEMS30_P1_dif_std))
                LEMS30_P2_remove = (LEMS_dif['LEMS30_P2']<(LEMS30_P2_dif_mean-n_dif_std*LEMS30_P2_dif_std))|(LEMS_dif['LEMS30_P2']>(LEMS30_P2_dif_mean+n_dif_std*LEMS30_P2_dif_std))
                LEMS30_P3_remove = (LEMS_dif['LEMS30_P3']<(LEMS30_P3_dif_mean-n_dif_std*LEMS30_P3_dif_std))|(LEMS_dif['LEMS30_P3']>(LEMS30_P3_dif_mean+n_dif_std*LEMS30_P3_dif_std))
                LEMS30_P4_remove = (LEMS_dif['LEMS30_P4']<(LEMS30_P4_dif_mean-n_dif_std*LEMS30_P4_dif_std))|(LEMS_dif['LEMS30_P4']>(LEMS30_P4_dif_mean+n_dif_std*LEMS30_P4_dif_std))
                LEMS30_P5_remove = (LEMS_dif['LEMS30_P5']<(LEMS30_P5_dif_mean-n_dif_std*LEMS30_P5_dif_std))|(LEMS_dif['LEMS30_P5']>(LEMS30_P5_dif_mean+n_dif_std*LEMS30_P5_dif_std))
                LEMS30_P6_remove = (LEMS_dif['LEMS30_P6']<(LEMS30_P6_dif_mean-n_dif_std*LEMS30_P6_dif_std))|(LEMS_dif['LEMS30_P6']>(LEMS30_P6_dif_mean+n_dif_std*LEMS30_P6_dif_std))
                LEMS30_P7_remove = (LEMS_dif['LEMS30_P7']<(LEMS30_P7_dif_mean-n_dif_std*LEMS30_P7_dif_std))|(LEMS_dif['LEMS30_P7']>(LEMS30_P7_dif_mean+n_dif_std*LEMS30_P7_dif_std))
                LEMS30_P8_remove = (LEMS_dif['LEMS30_P8']<(LEMS30_P8_dif_mean-n_dif_std*LEMS30_P8_dif_std))|(LEMS_dif['LEMS30_P8']>(LEMS30_P8_dif_mean+n_dif_std*LEMS30_P8_dif_std))
                LEMS120_P1_remove = (LEMS_dif['LEMS120_P1']<(LEMS120_P1_dif_mean-n_dif_std*LEMS120_P1_dif_std))|(LEMS_dif['LEMS120_P1']>(LEMS120_P1_dif_mean+n_dif_std*LEMS120_P1_dif_std))
                LEMS120_P2_remove = (LEMS_dif['LEMS120_P2']<(LEMS120_P2_dif_mean-n_dif_std*LEMS120_P2_dif_std))|(LEMS_dif['LEMS120_P2']>(LEMS120_P2_dif_mean+n_dif_std*LEMS120_P2_dif_std))
                LEMS120_P3_remove = (LEMS_dif['LEMS120_P3']<(LEMS120_P3_dif_mean-n_dif_std*LEMS120_P3_dif_std))|(LEMS_dif['LEMS120_P3']>(LEMS120_P3_dif_mean+n_dif_std*LEMS120_P3_dif_std))
                LEMS120_P4_remove = (LEMS_dif['LEMS120_P4']<(LEMS120_P4_dif_mean-n_dif_std*LEMS120_P4_dif_std))|(LEMS_dif['LEMS120_P4']>(LEMS120_P4_dif_mean+n_dif_std*LEMS120_P4_dif_std))
                LEMS120_P5_remove = (LEMS_dif['LEMS120_P5']<(LEMS120_P5_dif_mean-n_dif_std*LEMS120_P5_dif_std))|(LEMS_dif['LEMS120_P5']>(LEMS120_P5_dif_mean+n_dif_std*LEMS120_P5_dif_std))
                LEMS120_P6_remove = (LEMS_dif['LEMS120_P6']<(LEMS120_P6_dif_mean-n_dif_std*LEMS120_P6_dif_std))|(LEMS_dif['LEMS120_P6']>(LEMS120_P6_dif_mean+n_dif_std*LEMS120_P6_dif_std))
                LEMS120_P7_remove = (LEMS_dif['LEMS120_P7']<(LEMS120_P7_dif_mean-n_dif_std*LEMS120_P7_dif_std))|(LEMS_dif['LEMS120_P7']>(LEMS120_P7_dif_mean+n_dif_std*LEMS120_P7_dif_std))
                LEMS120_P8_remove = (LEMS_dif['LEMS120_P8']<(LEMS120_P8_dif_mean-n_dif_std*LEMS120_P8_dif_std))|(LEMS_dif['LEMS120_P8']>(LEMS120_P8_dif_mean+n_dif_std*LEMS120_P8_dif_std))
                
                # Remove data fall outside.
                LEMS_DataFrame['LEMS30_P1'][LEMS30_P1_remove] = np.nan
                LEMS_DataFrame['LEMS30_P2'][LEMS30_P2_remove] = np.nan
                LEMS_DataFrame['LEMS30_P3'][LEMS30_P3_remove] = np.nan
                LEMS_DataFrame['LEMS30_P4'][LEMS30_P4_remove] = np.nan
                LEMS_DataFrame['LEMS30_P5'][LEMS30_P5_remove] = np.nan
                LEMS_DataFrame['LEMS30_P6'][LEMS30_P6_remove] = np.nan
                LEMS_DataFrame['LEMS30_P7'][LEMS30_P7_remove] = np.nan
                LEMS_DataFrame['LEMS30_P8'][LEMS30_P8_remove] = np.nan
                LEMS_DataFrame['LEMS120_P1'][LEMS120_P1_remove] = np.nan
                LEMS_DataFrame['LEMS120_P2'][LEMS120_P2_remove] = np.nan
                LEMS_DataFrame['LEMS120_P3'][LEMS120_P3_remove] = np.nan
                LEMS_DataFrame['LEMS120_P4'][LEMS120_P4_remove] = np.nan
                LEMS_DataFrame['LEMS120_P5'][LEMS120_P5_remove] = np.nan
                LEMS_DataFrame['LEMS120_P6'][LEMS120_P6_remove] = np.nan
                LEMS_DataFrame['LEMS120_P7'][LEMS120_P7_remove] = np.nan
                LEMS_DataFrame['LEMS120_P8'][LEMS120_P8_remove] = np.nan

                # Calculate data ranges.
                LEMS30_P1_dif_lower_boundary = LEMS30_P1_dif_mean-n_dif_std*LEMS30_P1_dif_std
                LEMS30_P1_dif_upper_boundary = LEMS30_P1_dif_mean+n_dif_std*LEMS30_P1_dif_std
                LEMS30_P2_dif_lower_boundary = LEMS30_P2_dif_mean-n_dif_std*LEMS30_P2_dif_std
                LEMS30_P2_dif_upper_boundary = LEMS30_P2_dif_mean+n_dif_std*LEMS30_P2_dif_std
                LEMS30_P3_dif_lower_boundary = LEMS30_P3_dif_mean-n_dif_std*LEMS30_P3_dif_std
                LEMS30_P3_dif_upper_boundary = LEMS30_P3_dif_mean+n_dif_std*LEMS30_P3_dif_std
                LEMS30_P4_dif_lower_boundary = LEMS30_P4_dif_mean-n_dif_std*LEMS30_P4_dif_std
                LEMS30_P4_dif_upper_boundary = LEMS30_P4_dif_mean+n_dif_std*LEMS30_P4_dif_std
                LEMS30_P5_dif_lower_boundary = LEMS30_P5_dif_mean-n_dif_std*LEMS30_P5_dif_std
                LEMS30_P5_dif_upper_boundary = LEMS30_P5_dif_mean+n_dif_std*LEMS30_P5_dif_std
                LEMS30_P6_dif_lower_boundary = LEMS30_P6_dif_mean-n_dif_std*LEMS30_P6_dif_std
                LEMS30_P6_dif_upper_boundary = LEMS30_P6_dif_mean+n_dif_std*LEMS30_P6_dif_std
                LEMS30_P7_dif_lower_boundary = LEMS30_P7_dif_mean-n_dif_std*LEMS30_P7_dif_std
                LEMS30_P7_dif_upper_boundary = LEMS30_P7_dif_mean+n_dif_std*LEMS30_P7_dif_std
                LEMS30_P8_dif_lower_boundary = LEMS30_P8_dif_mean-n_dif_std*LEMS30_P8_dif_std
                LEMS30_P8_dif_upper_boundary = LEMS30_P8_dif_mean+n_dif_std*LEMS30_P8_dif_std
                LEMS120_P1_dif_lower_boundary = LEMS120_P1_dif_mean-n_dif_std*LEMS120_P1_dif_std
                LEMS120_P1_dif_upper_boundary = LEMS120_P1_dif_mean+n_dif_std*LEMS120_P1_dif_std
                LEMS120_P2_dif_lower_boundary = LEMS120_P2_dif_mean-n_dif_std*LEMS120_P2_dif_std
                LEMS120_P2_dif_upper_boundary = LEMS120_P2_dif_mean+n_dif_std*LEMS120_P2_dif_std
                LEMS120_P3_dif_lower_boundary = LEMS120_P3_dif_mean-n_dif_std*LEMS120_P3_dif_std
                LEMS120_P3_dif_upper_boundary = LEMS120_P3_dif_mean+n_dif_std*LEMS120_P3_dif_std
                LEMS120_P4_dif_lower_boundary = LEMS120_P4_dif_mean-n_dif_std*LEMS120_P4_dif_std
                LEMS120_P4_dif_upper_boundary = LEMS120_P4_dif_mean+n_dif_std*LEMS120_P4_dif_std
                LEMS120_P5_dif_lower_boundary = LEMS120_P5_dif_mean-n_dif_std*LEMS120_P5_dif_std
                LEMS120_P5_dif_upper_boundary = LEMS120_P5_dif_mean+n_dif_std*LEMS120_P5_dif_std
                LEMS120_P6_dif_lower_boundary = LEMS120_P6_dif_mean-n_dif_std*LEMS120_P6_dif_std
                LEMS120_P6_dif_upper_boundary = LEMS120_P6_dif_mean+n_dif_std*LEMS120_P6_dif_std
                LEMS120_P7_dif_lower_boundary = LEMS120_P7_dif_mean-n_dif_std*LEMS120_P7_dif_std
                LEMS120_P7_dif_upper_boundary = LEMS120_P7_dif_mean+n_dif_std*LEMS120_P7_dif_std
                LEMS120_P8_dif_lower_boundary = LEMS120_P8_dif_mean-n_dif_std*LEMS120_P8_dif_std
                LEMS120_P8_dif_upper_boundary = LEMS120_P8_dif_mean+n_dif_std*LEMS120_P8_dif_std

                # Calculate how many data points are removed.
                n_removed_LEMS30_P1 = sum(LEMS30_P1_remove)
                n_removed_LEMS30_P2 = sum(LEMS30_P2_remove)
                n_removed_LEMS30_P3 = sum(LEMS30_P3_remove)
                n_removed_LEMS30_P4 = sum(LEMS30_P4_remove)
                n_removed_LEMS30_P5 = sum(LEMS30_P5_remove)
                n_removed_LEMS30_P6 = sum(LEMS30_P6_remove)
                n_removed_LEMS30_P7 = sum(LEMS30_P7_remove)
                n_removed_LEMS30_P8 = sum(LEMS30_P8_remove)
                n_removed_LEMS120_P1 = sum(LEMS120_P1_remove)
                n_removed_LEMS120_P2 = sum(LEMS120_P2_remove)
                n_removed_LEMS120_P3 = sum(LEMS120_P3_remove)
                n_removed_LEMS120_P4 = sum(LEMS120_P4_remove)
                n_removed_LEMS120_P5 = sum(LEMS120_P5_remove)
                n_removed_LEMS120_P6 = sum(LEMS120_P6_remove)
                n_removed_LEMS120_P7 = sum(LEMS120_P7_remove)
                n_removed_LEMS120_P8 = sum(LEMS120_P8_remove)
                
                n_removed_LEMS30_P1_total += n_removed_LEMS30_P1
                n_removed_LEMS30_P2_total += n_removed_LEMS30_P2
                n_removed_LEMS30_P3_total += n_removed_LEMS30_P3
                n_removed_LEMS30_P4_total += n_removed_LEMS30_P4
                n_removed_LEMS30_P5_total += n_removed_LEMS30_P5
                n_removed_LEMS30_P6_total += n_removed_LEMS30_P6
                n_removed_LEMS30_P7_total += n_removed_LEMS30_P7
                n_removed_LEMS30_P8_total += n_removed_LEMS30_P8
                n_removed_LEMS120_P1_total += n_removed_LEMS120_P1
                n_removed_LEMS120_P2_total += n_removed_LEMS120_P2
                n_removed_LEMS120_P3_total += n_removed_LEMS120_P3
                n_removed_LEMS120_P4_total += n_removed_LEMS120_P4
                n_removed_LEMS120_P5_total += n_removed_LEMS120_P5
                n_removed_LEMS120_P6_total += n_removed_LEMS120_P6
                n_removed_LEMS120_P7_total += n_removed_LEMS120_P7
                n_removed_LEMS120_P8_total += n_removed_LEMS120_P8

                if isVerbose:
                    print('LEMS30_P1_dif_std:', LEMS30_P1_dif_std)
                    print('LEMS30_P2_dif_std:', LEMS30_P2_dif_std)
                    print('LEMS30_P3_dif_std:', LEMS30_P3_dif_std)
                    print('LEMS30_P4_dif_std:', LEMS30_P4_dif_std)
                    print('LEMS30_P5_dif_std:', LEMS30_P5_dif_std)
                    print('LEMS30_P6_dif_std:', LEMS30_P6_dif_std)
                    print('LEMS30_P7_dif_std:', LEMS30_P7_dif_std)
                    print('LEMS30_P8_dif_std:', LEMS30_P8_dif_std)
                    print('LEMS120_P1_dif_std:', LEMS120_P1_dif_std)
                    print('LEMS120_P2_dif_std:', LEMS120_P2_dif_std)
                    print('LEMS120_P3_dif_std:', LEMS120_P3_dif_std)
                    print('LEMS120_P4_dif_std:', LEMS120_P4_dif_std)
                    print('LEMS120_P5_dif_std:', LEMS120_P5_dif_std)
                    print('LEMS120_P6_dif_std:', LEMS120_P6_dif_std)
                    print('LEMS120_P7_dif_std:', LEMS120_P7_dif_std)
                    print('LEMS120_P8_dif_std:', LEMS120_P8_dif_std)
                    
                    print('LEMS30_P1_dif_mean:', LEMS30_P1_dif_mean)
                    print('LEMS30_P2_dif_mean:', LEMS30_P2_dif_mean)
                    print('LEMS30_P3_dif_mean:', LEMS30_P3_dif_mean)
                    print('LEMS30_P4_dif_mean:', LEMS30_P4_dif_mean)
                    print('LEMS30_P5_dif_mean:', LEMS30_P5_dif_mean)
                    print('LEMS30_P6_dif_mean:', LEMS30_P6_dif_mean)
                    print('LEMS30_P7_dif_mean:', LEMS30_P7_dif_mean)
                    print('LEMS30_P8_dif_mean:', LEMS30_P8_dif_mean)
                    print('LEMS120_P1_dif_mean:', LEMS120_P1_dif_mean)
                    print('LEMS120_P2_dif_mean:', LEMS120_P2_dif_mean)
                    print('LEMS120_P3_dif_mean:', LEMS120_P3_dif_mean)
                    print('LEMS120_P4_dif_mean:', LEMS120_P4_dif_mean)
                    print('LEMS120_P5_dif_mean:', LEMS120_P5_dif_mean)
                    print('LEMS120_P6_dif_mean:', LEMS120_P6_dif_mean)
                    print('LEMS120_P7_dif_mean:', LEMS120_P7_dif_mean)
                    print('LEMS120_P8_dif_mean:', LEMS120_P8_dif_mean)
                    
                    print('The LEMS LEMS30_P1_dif value range within {} std is [{}, {}]'.format(n_dif_std, LEMS30_P1_dif_lower_boundary, LEMS30_P1_dif_upper_boundary))
                    print('The LEMS LEMS30_P2_dif value range within {} std is [{}, {}]'.format(n_dif_std, LEMS30_P2_dif_lower_boundary, LEMS30_P2_dif_upper_boundary))
                    print('The LEMS LEMS30_P3_dif value range within {} std is [{}, {}]'.format(n_dif_std, LEMS30_P3_dif_lower_boundary, LEMS30_P3_dif_upper_boundary))
                    print('The LEMS LEMS30_P4_dif value range within {} std is [{}, {}]'.format(n_dif_std, LEMS30_P4_dif_lower_boundary, LEMS30_P4_dif_upper_boundary))
                    print('The LEMS LEMS30_P5_dif value range within {} std is [{}, {}]'.format(n_dif_std, LEMS30_P5_dif_lower_boundary, LEMS30_P5_dif_upper_boundary))
                    print('The LEMS LEMS30_P6_dif value range within {} std is [{}, {}]'.format(n_dif_std, LEMS30_P6_dif_lower_boundary, LEMS30_P6_dif_upper_boundary))
                    print('The LEMS LEMS30_P7_dif value range within {} std is [{}, {}]'.format(n_dif_std, LEMS30_P7_dif_lower_boundary, LEMS30_P7_dif_upper_boundary))
                    print('The LEMS LEMS30_P8_dif value range within {} std is [{}, {}]'.format(n_dif_std, LEMS30_P8_dif_lower_boundary, LEMS30_P8_dif_upper_boundary))
                    print('The LEMS LEMS120_P1_dif value range within {} std is [{}, {}]'.format(n_dif_std, LEMS120_P1_dif_lower_boundary, LEMS120_P1_dif_upper_boundary))
                    print('The LEMS LEMS120_P2_dif value range within {} std is [{}, {}]'.format(n_dif_std, LEMS120_P2_dif_lower_boundary, LEMS120_P2_dif_upper_boundary))
                    print('The LEMS LEMS120_P3_dif value range within {} std is [{}, {}]'.format(n_dif_std, LEMS120_P3_dif_lower_boundary, LEMS120_P3_dif_upper_boundary))
                    print('The LEMS LEMS120_P4_dif value range within {} std is [{}, {}]'.format(n_dif_std, LEMS120_P4_dif_lower_boundary, LEMS120_P4_dif_upper_boundary))
                    print('The LEMS LEMS120_P5_dif value range within {} std is [{}, {}]'.format(n_dif_std, LEMS120_P5_dif_lower_boundary, LEMS120_P5_dif_upper_boundary))
                    print('The LEMS LEMS120_P6_dif value range within {} std is [{}, {}]'.format(n_dif_std, LEMS120_P6_dif_lower_boundary, LEMS120_P6_dif_upper_boundary))
                    print('The LEMS LEMS120_P7_dif value range within {} std is [{}, {}]'.format(n_dif_std, LEMS120_P7_dif_lower_boundary, LEMS120_P7_dif_upper_boundary))
                    print('The LEMS LEMS120_P8_dif value range within {} std is [{}, {}]'.format(n_dif_std, LEMS120_P8_dif_lower_boundary, LEMS120_P8_dif_upper_boundary))
            
                    print('In LEMS30_P1, this operation removed {} records!'.format(n_removed_LEMS30_P1))
                    print('In LEMS30_P2, this operation removed {} records!'.format(n_removed_LEMS30_P2))
                    print('In LEMS30_P3, this operation removed {} records!'.format(n_removed_LEMS30_P3))
                    print('In LEMS30_P4, this operation removed {} records!'.format(n_removed_LEMS30_P4))
                    print('In LEMS30_P5, this operation removed {} records!'.format(n_removed_LEMS30_P5))
                    print('In LEMS30_P6, this operation removed {} records!'.format(n_removed_LEMS30_P6))
                    print('In LEMS30_P7, this operation removed {} records!'.format(n_removed_LEMS30_P7))
                    print('In LEMS30_P8, this operation removed {} records!'.format(n_removed_LEMS30_P8))
                    print('In LEMS120_P1, this operation removed {} records!'.format(n_removed_LEMS120_P1))
                    print('In LEMS120_P2, this operation removed {} records!'.format(n_removed_LEMS120_P2))
                    print('In LEMS120_P3, this operation removed {} records!'.format(n_removed_LEMS120_P3))
                    print('In LEMS120_P4, this operation removed {} records!'.format(n_removed_LEMS120_P4))
                    print('In LEMS120_P5, this operation removed {} records!'.format(n_removed_LEMS120_P5))
                    print('In LEMS120_P6, this operation removed {} records!'.format(n_removed_LEMS120_P6))
                    print('In LEMS120_P7, this operation removed {} records!'.format(n_removed_LEMS120_P7))
                    print('In LEMS120_P8, this operation removed {} records!'.format(n_removed_LEMS120_P8))
                    
                    print('Until now, in LEMS30_P1, {} records have been removed!'.format(n_removed_LEMS30_P1_total))
                    print('Until now, in LEMS30_P2, {} records have been removed!'.format(n_removed_LEMS30_P2_total))
                    print('Until now, in LEMS30_P3, {} records have been removed!'.format(n_removed_LEMS30_P3_total))
                    print('Until now, in LEMS30_P4, {} records have been removed!'.format(n_removed_LEMS30_P4_total))
                    print('Until now, in LEMS30_P5, {} records have been removed!'.format(n_removed_LEMS30_P5_total))
                    print('Until now, in LEMS30_P6, {} records have been removed!'.format(n_removed_LEMS30_P6_total))
                    print('Until now, in LEMS30_P7, {} records have been removed!'.format(n_removed_LEMS30_P7_total))
                    print('Until now, in LEMS30_P8, {} records have been removed!'.format(n_removed_LEMS30_P8_total))
                    print('Until now, in LEMS120_P1, {} records have been removed!'.format(n_removed_LEMS120_P1_total))
                    print('Until now, in LEMS120_P2, {} records have been removed!'.format(n_removed_LEMS120_P2_total))
                    print('Until now, in LEMS120_P3, {} records have been removed!'.format(n_removed_LEMS120_P3_total))
                    print('Until now, in LEMS120_P4, {} records have been removed!'.format(n_removed_LEMS120_P4_total))
                    print('Until now, in LEMS120_P5, {} records have been removed!'.format(n_removed_LEMS120_P5_total))
                    print('Until now, in LEMS120_P6, {} records have been removed!'.format(n_removed_LEMS120_P6_total))
                    print('Until now, in LEMS120_P7, {} records have been removed!'.format(n_removed_LEMS120_P7_total))
                    print('Until now, in LEMS120_P8, {} records have been removed!'.format(n_removed_LEMS120_P8_total))
                    
                    print('\n')
            
            # If plot filter process of LEMS or not.
            if isPlotFilterProcess:
                # Plot LEMS30 filter process.
                print('Plotting LEMS30 filtering process...')
                fig_line_width = 0.1
                fig_ylabel_fontsize = 9
                fig_xtick_fontsize = 8
                fig_ytick_fontsize = 8
                fig_legend_size = 5
                fig,ax = plt.subplots(16,1, sharex=True,figsize=(18, 30))
                LEMS30_P1_plot = ax[0]
                LEMS30_P1_dif = ax[1]
                LEMS30_P2_plot = ax[2]
                LEMS30_P2_dif = ax[3]
                LEMS30_P3_plot = ax[4]
                LEMS30_P3_dif = ax[5]
                LEMS30_P4_plot = ax[6]
                LEMS30_P4_dif = ax[7]
                LEMS30_P5_plot = ax[8]
                LEMS30_P5_dif = ax[9]
                LEMS30_P6_plot = ax[10]
                LEMS30_P6_dif = ax[11]
                LEMS30_P7_plot = ax[12]
                LEMS30_P7_dif = ax[13]
                LEMS30_P8_plot = ax[14]
                LEMS30_P8_dif = ax[15]
                
                # Plotting LEMS30_P1 filter process.
                LEMS30_P1_plot.plot(LEMS_DataFrame0.index, LEMS_DataFrame0['LEMS30_P1'].fillna(0),color = 'red', linewidth=fig_line_width, label='LEMS30_P1_original') # Original data.
                LEMS30_P1_plot.plot(LEMS_DataFrame.index, LEMS_DataFrame['LEMS30_P1'].fillna(0),color = 'blue', linewidth=fig_line_width, label='LEMS30_P1_processed') # Filtered data.
                LEMS30_P1_plot.plot(LEMS_LowPass.index, LEMS_LowPass['LEMS30_P1'].fillna(0),color = 'black', linewidth=fig_line_width, label='LEMS30_P1_LowPass') # Low pass curve.
                LEMS30_P1_plot.set_ylabel('LEMS30_P1', fontsize=fig_ylabel_fontsize)
                LEMS30_P1_plot.legend(loc='upper left',prop={'size':fig_legend_size})
                LEMS30_P1_plot.tick_params(axis='y', which='major', labelsize=fig_ytick_fontsize)
                LEMS30_P1_dif.plot(LEMS_dif.index, LEMS_dif['LEMS30_P1'].fillna(0),color = 'green', linewidth=fig_line_width) # Difference data.
                LEMS30_P1_dif.tick_params(axis='y', which='major', labelsize=fig_ytick_fontsize)
                LEMS30_P1_dif.set_ylabel('LEMS30_P1_dif', fontsize=fig_ylabel_fontsize)
                # Plotting LEMS30_P2 filter process.
                LEMS30_P2_plot.plot(LEMS_DataFrame0.index, LEMS_DataFrame0['LEMS30_P2'].fillna(0),color = 'red', linewidth=fig_line_width, label='LEMS30_P2_original') # Original data.
                LEMS30_P2_plot.plot(LEMS_DataFrame.index, LEMS_DataFrame['LEMS30_P2'].fillna(0),color = 'blue', linewidth=fig_line_width, label='LEMS30_P2_processed') # Filtered data.
                LEMS30_P2_plot.plot(LEMS_LowPass.index, LEMS_LowPass['LEMS30_P2'].fillna(0),color = 'black', linewidth=fig_line_width, label='LEMS30_P2_LowPass') # Low pass curve.
                LEMS30_P2_plot.set_ylabel('LEMS30_P2', fontsize=fig_ylabel_fontsize)
                LEMS30_P2_plot.legend(loc='upper left',prop={'size':fig_legend_size})
                LEMS30_P2_plot.tick_params(axis='y', which='major', labelsize=fig_ytick_fontsize)
                LEMS30_P2_dif.plot(LEMS_dif.index, LEMS_dif['LEMS30_P2'].fillna(0),color = 'green', linewidth=fig_line_width) # Difference data.
                LEMS30_P2_dif.tick_params(axis='y', which='major', labelsize=fig_ytick_fontsize)
                LEMS30_P2_dif.set_ylabel('LEMS30_P2_dif', fontsize=fig_ylabel_fontsize)
                # Plotting LEMS30_P3 filter process.
                LEMS30_P3_plot.plot(LEMS_DataFrame0.index, LEMS_DataFrame0['LEMS30_P3'].fillna(0),color = 'red', linewidth=fig_line_width, label='LEMS30_P3_original') # Original data.
                LEMS30_P3_plot.plot(LEMS_DataFrame.index, LEMS_DataFrame['LEMS30_P3'].fillna(0),color = 'blue', linewidth=fig_line_width, label='LEMS30_P3_processed') # Filtered data.
                LEMS30_P3_plot.plot(LEMS_LowPass.index, LEMS_LowPass['LEMS30_P3'].fillna(0),color = 'black', linewidth=fig_line_width, label='LEMS30_P3_LowPass') # Low pass curve.
                LEMS30_P3_plot.set_ylabel('LEMS30_P3', fontsize=fig_ylabel_fontsize)
                LEMS30_P3_plot.legend(loc='upper left',prop={'size':fig_legend_size})
                LEMS30_P3_plot.tick_params(axis='y', which='major', labelsize=fig_ytick_fontsize)
                LEMS30_P3_dif.plot(LEMS_dif.index, LEMS_dif['LEMS30_P3'].fillna(0),color = 'green', linewidth=fig_line_width) # Difference data.
                LEMS30_P3_dif.tick_params(axis='y', which='major', labelsize=fig_ytick_fontsize)
                LEMS30_P3_dif.set_ylabel('LEMS30_P3_dif', fontsize=fig_ylabel_fontsize)
                # Plotting LEMS30_P4 filter process.
                LEMS30_P4_plot.plot(LEMS_DataFrame0.index, LEMS_DataFrame0['LEMS30_P4'].fillna(0),color = 'red', linewidth=fig_line_width, label='LEMS30_P4_original') # Original data.
                LEMS30_P4_plot.plot(LEMS_DataFrame.index, LEMS_DataFrame['LEMS30_P4'].fillna(0),color = 'blue', linewidth=fig_line_width, label='LEMS30_P4_processed') # Filtered data.
                LEMS30_P4_plot.plot(LEMS_LowPass.index, LEMS_LowPass['LEMS30_P4'].fillna(0),color = 'black', linewidth=fig_line_width, label='LEMS30_P4_LowPass') # Low pass curve.
                LEMS30_P4_plot.set_ylabel('LEMS30_P4', fontsize=fig_ylabel_fontsize)
                LEMS30_P4_plot.legend(loc='upper left',prop={'size':fig_legend_size})
                LEMS30_P4_plot.tick_params(axis='y', which='major', labelsize=fig_ytick_fontsize)
                LEMS30_P4_dif.plot(LEMS_dif.index, LEMS_dif['LEMS30_P4'].fillna(0),color = 'green', linewidth=fig_line_width) # Difference data.
                LEMS30_P4_dif.tick_params(axis='y', which='major', labelsize=fig_ytick_fontsize)
                LEMS30_P4_dif.set_ylabel('LEMS30_P4_dif', fontsize=fig_ylabel_fontsize)
                # Plotting LEMS30_P5 filter process.
                LEMS30_P5_plot.plot(LEMS_DataFrame0.index, LEMS_DataFrame0['LEMS30_P5'].fillna(0),color = 'red', linewidth=fig_line_width, label='LEMS30_P5_original') # Original data.
                LEMS30_P5_plot.plot(LEMS_DataFrame.index, LEMS_DataFrame['LEMS30_P5'].fillna(0),color = 'blue', linewidth=fig_line_width, label='LEMS30_P5_processed') # Filtered data.
                LEMS30_P5_plot.plot(LEMS_LowPass.index, LEMS_LowPass['LEMS30_P5'].fillna(0),color = 'black', linewidth=fig_line_width, label='LEMS30_P5_LowPass') # Low pass curve.
                LEMS30_P5_plot.set_ylabel('LEMS30_P5', fontsize=fig_ylabel_fontsize)
                LEMS30_P5_plot.legend(loc='upper left',prop={'size':fig_legend_size})
                LEMS30_P5_plot.tick_params(axis='y', which='major', labelsize=fig_ytick_fontsize)
                LEMS30_P5_dif.plot(LEMS_dif.index, LEMS_dif['LEMS30_P5'].fillna(0),color = 'green', linewidth=fig_line_width) # Difference data.
                LEMS30_P5_dif.tick_params(axis='y', which='major', labelsize=fig_ytick_fontsize)
                LEMS30_P5_dif.set_ylabel('LEMS30_P5_dif', fontsize=fig_ylabel_fontsize)
                # Plotting LEMS30_P6 filter process.
                LEMS30_P6_plot.plot(LEMS_DataFrame0.index, LEMS_DataFrame0['LEMS30_P6'].fillna(0),color = 'red', linewidth=fig_line_width, label='LEMS30_P6_original') # Original data.
                LEMS30_P6_plot.plot(LEMS_DataFrame.index, LEMS_DataFrame['LEMS30_P6'].fillna(0),color = 'blue', linewidth=fig_line_width, label='LEMS30_P6_processed') # Filtered data.
                LEMS30_P6_plot.plot(LEMS_LowPass.index, LEMS_LowPass['LEMS30_P6'].fillna(0),color = 'black', linewidth=fig_line_width, label='LEMS30_P6_LowPass') # Low pass curve.
                LEMS30_P6_plot.set_ylabel('LEMS30_P6', fontsize=fig_ylabel_fontsize)
                LEMS30_P6_plot.legend(loc='upper left',prop={'size':fig_legend_size})
                LEMS30_P6_plot.tick_params(axis='y', which='major', labelsize=fig_ytick_fontsize)
                LEMS30_P6_dif.plot(LEMS_dif.index, LEMS_dif['LEMS30_P6'].fillna(0),color = 'green', linewidth=fig_line_width) # Difference data.
                LEMS30_P6_dif.tick_params(axis='y', which='major', labelsize=fig_ytick_fontsize)
                LEMS30_P6_dif.set_ylabel('LEMS30_P6_dif', fontsize=fig_ylabel_fontsize)
                # Plotting LEMS30_P7 filter process.
                LEMS30_P7_plot.plot(LEMS_DataFrame0.index, LEMS_DataFrame0['LEMS30_P7'].fillna(0),color = 'red', linewidth=fig_line_width, label='LEMS30_P7_original') # Original data.
                LEMS30_P7_plot.plot(LEMS_DataFrame.index, LEMS_DataFrame['LEMS30_P7'].fillna(0),color = 'blue', linewidth=fig_line_width, label='LEMS30_P7_processed') # Filtered data.
                LEMS30_P7_plot.plot(LEMS_LowPass.index, LEMS_LowPass['LEMS30_P7'].fillna(0),color = 'black', linewidth=fig_line_width, label='LEMS30_P7_LowPass') # Low pass curve.
                LEMS30_P7_plot.set_ylabel('LEMS30_P7', fontsize=fig_ylabel_fontsize)
                LEMS30_P7_plot.legend(loc='upper left',prop={'size':fig_legend_size})
                LEMS30_P7_plot.tick_params(axis='y', which='major', labelsize=fig_ytick_fontsize)
                LEMS30_P7_dif.plot(LEMS_dif.index, LEMS_dif['LEMS30_P7'].fillna(0),color = 'green', linewidth=fig_line_width) # Difference data.
                LEMS30_P7_dif.tick_params(axis='y', which='major', labelsize=fig_ytick_fontsize)
                LEMS30_P7_dif.set_ylabel('LEMS30_P7_dif', fontsize=fig_ylabel_fontsize)
                # Plotting LEMS30_P8 filter process.
                LEMS30_P8_plot.plot(LEMS_DataFrame0.index, LEMS_DataFrame0['LEMS30_P8'].fillna(0),color = 'red', linewidth=fig_line_width, label='LEMS30_P8_original') # Original data.
                LEMS30_P8_plot.plot(LEMS_DataFrame.index, LEMS_DataFrame['LEMS30_P8'].fillna(0),color = 'blue', linewidth=fig_line_width, label='LEMS30_P8_processed') # Filtered data.
                LEMS30_P8_plot.plot(LEMS_LowPass.index, LEMS_LowPass['LEMS30_P8'].fillna(0),color = 'black', linewidth=fig_line_width, label='LEMS30_P8_LowPass') # Low pass curve.
                LEMS30_P8_plot.set_ylabel('LEMS30_P8', fontsize=fig_ylabel_fontsize)
                LEMS30_P8_plot.legend(loc='upper left',prop={'size':fig_legend_size})
                LEMS30_P8_plot.tick_params(axis='y', which='major', labelsize=fig_ytick_fontsize)
                LEMS30_P8_dif.plot(LEMS_dif.index, LEMS_dif['LEMS30_P8'].fillna(0),color = 'green', linewidth=fig_line_width) # Difference data.
                LEMS30_P8_dif.tick_params(axis='y', which='major', labelsize=fig_ytick_fontsize)
                LEMS30_P8_dif.set_ylabel('LEMS30_P8_dif', fontsize=fig_ylabel_fontsize)
                # This is a shared axis for all subplot
                LEMS30_P8_dif.tick_params(axis='x', which='major', labelsize=fig_xtick_fontsize)
                # Save plot.
                fig.savefig(data_pickle_dir + '/ACE_filter_process_LEMS30_' + timeStart_str + '_' + timeEnd_str + '.eps',format='eps', dpi=500, bbox_inches='tight')
                
                # Plot LEMS120 filter process.
                print('Plotting LEMS120 filtering process...')
                fig_line_width = 0.1
                fig_ylabel_fontsize = 9
                fig_xtick_fontsize = 8
                fig_ytick_fontsize = 8
                fig_legend_size = 5
                fig,ax = plt.subplots(16,1, sharex=True,figsize=(18, 30))
                LEMS120_P1_plot = ax[0]
                LEMS120_P1_dif = ax[1]
                LEMS120_P2_plot = ax[2]
                LEMS120_P2_dif = ax[3]
                LEMS120_P3_plot = ax[4]
                LEMS120_P3_dif = ax[5]
                LEMS120_P4_plot = ax[6]
                LEMS120_P4_dif = ax[7]
                LEMS120_P5_plot = ax[8]
                LEMS120_P5_dif = ax[9]
                LEMS120_P6_plot = ax[10]
                LEMS120_P6_dif = ax[11]
                LEMS120_P7_plot = ax[12]
                LEMS120_P7_dif = ax[13]
                LEMS120_P8_plot = ax[14]
                LEMS120_P8_dif = ax[15]
                
                # Plotting LEMS120_P1 filter process.
                LEMS120_P1_plot.plot(LEMS_DataFrame0.index, LEMS_DataFrame0['LEMS120_P1'].fillna(0),color = 'red', linewidth=fig_line_width, label='LEMS120_P1_original') # Original data.
                LEMS120_P1_plot.plot(LEMS_DataFrame.index, LEMS_DataFrame['LEMS120_P1'].fillna(0),color = 'blue', linewidth=fig_line_width, label='LEMS120_P1_processed') # Filtered data.
                LEMS120_P1_plot.plot(LEMS_LowPass.index, LEMS_LowPass['LEMS120_P1'].fillna(0),color = 'black', linewidth=fig_line_width, label='LEMS120_P1_LowPass') # Low pass curve.
                LEMS120_P1_plot.set_ylabel('LEMS120_P1', fontsize=fig_ylabel_fontsize)
                LEMS120_P1_plot.legend(loc='upper left',prop={'size':fig_legend_size})
                LEMS120_P1_plot.tick_params(axis='y', which='major', labelsize=fig_ytick_fontsize)
                LEMS120_P1_dif.plot(LEMS_dif.index, LEMS_dif['LEMS120_P1'].fillna(0),color = 'green', linewidth=fig_line_width) # Difference data.
                LEMS120_P1_dif.tick_params(axis='y', which='major', labelsize=fig_ytick_fontsize)
                LEMS120_P1_dif.set_ylabel('LEMS120_P1_dif', fontsize=fig_ylabel_fontsize)
                # Plotting LEMS120_P2 filter process.
                LEMS120_P2_plot.plot(LEMS_DataFrame0.index, LEMS_DataFrame0['LEMS120_P2'].fillna(0),color = 'red', linewidth=fig_line_width, label='LEMS120_P2_original') # Original data.
                LEMS120_P2_plot.plot(LEMS_DataFrame.index, LEMS_DataFrame['LEMS120_P2'].fillna(0),color = 'blue', linewidth=fig_line_width, label='LEMS120_P2_processed') # Filtered data.
                LEMS120_P2_plot.plot(LEMS_LowPass.index, LEMS_LowPass['LEMS120_P2'].fillna(0),color = 'black', linewidth=fig_line_width, label='LEMS120_P2_LowPass') # Low pass curve.
                LEMS120_P2_plot.set_ylabel('LEMS120_P2', fontsize=fig_ylabel_fontsize)
                LEMS120_P2_plot.legend(loc='upper left',prop={'size':fig_legend_size})
                LEMS120_P2_plot.tick_params(axis='y', which='major', labelsize=fig_ytick_fontsize)
                LEMS120_P2_dif.plot(LEMS_dif.index, LEMS_dif['LEMS120_P2'].fillna(0),color = 'green', linewidth=fig_line_width) # Difference data.
                LEMS120_P2_dif.tick_params(axis='y', which='major', labelsize=fig_ytick_fontsize)
                LEMS120_P2_dif.set_ylabel('LEMS120_P2_dif', fontsize=fig_ylabel_fontsize)
                # Plotting LEMS120_P3 filter process.
                LEMS120_P3_plot.plot(LEMS_DataFrame0.index, LEMS_DataFrame0['LEMS120_P3'].fillna(0),color = 'red', linewidth=fig_line_width, label='LEMS120_P3_original') # Original data.
                LEMS120_P3_plot.plot(LEMS_DataFrame.index, LEMS_DataFrame['LEMS120_P3'].fillna(0),color = 'blue', linewidth=fig_line_width, label='LEMS120_P3_processed') # Filtered data.
                LEMS120_P3_plot.plot(LEMS_LowPass.index, LEMS_LowPass['LEMS120_P3'].fillna(0),color = 'black', linewidth=fig_line_width, label='LEMS120_P3_LowPass') # Low pass curve.
                LEMS120_P3_plot.set_ylabel('LEMS120_P3', fontsize=fig_ylabel_fontsize)
                LEMS120_P3_plot.legend(loc='upper left',prop={'size':fig_legend_size})
                LEMS120_P3_plot.tick_params(axis='y', which='major', labelsize=fig_ytick_fontsize)
                LEMS120_P3_dif.plot(LEMS_dif.index, LEMS_dif['LEMS120_P3'].fillna(0),color = 'green', linewidth=fig_line_width) # Difference data.
                LEMS120_P3_dif.tick_params(axis='y', which='major', labelsize=fig_ytick_fontsize)
                LEMS120_P3_dif.set_ylabel('LEMS120_P3_dif', fontsize=fig_ylabel_fontsize)
                # Plotting LEMS120_P4 filter process.
                LEMS120_P4_plot.plot(LEMS_DataFrame0.index, LEMS_DataFrame0['LEMS120_P4'].fillna(0),color = 'red', linewidth=fig_line_width, label='LEMS120_P4_original') # Original data.
                LEMS120_P4_plot.plot(LEMS_DataFrame.index, LEMS_DataFrame['LEMS120_P4'].fillna(0),color = 'blue', linewidth=fig_line_width, label='LEMS120_P4_processed') # Filtered data.
                LEMS120_P4_plot.plot(LEMS_LowPass.index, LEMS_LowPass['LEMS120_P4'].fillna(0),color = 'black', linewidth=fig_line_width, label='LEMS120_P4_LowPass') # Low pass curve.
                LEMS120_P4_plot.set_ylabel('LEMS120_P4', fontsize=fig_ylabel_fontsize)
                LEMS120_P4_plot.legend(loc='upper left',prop={'size':fig_legend_size})
                LEMS120_P4_plot.tick_params(axis='y', which='major', labelsize=fig_ytick_fontsize)
                LEMS120_P4_dif.plot(LEMS_dif.index, LEMS_dif['LEMS120_P4'].fillna(0),color = 'green', linewidth=fig_line_width) # Difference data.
                LEMS120_P4_dif.tick_params(axis='y', which='major', labelsize=fig_ytick_fontsize)
                LEMS120_P4_dif.set_ylabel('LEMS120_P4_dif', fontsize=fig_ylabel_fontsize)
                # Plotting LEMS120_P5 filter process.
                LEMS120_P5_plot.plot(LEMS_DataFrame0.index, LEMS_DataFrame0['LEMS120_P5'].fillna(0),color = 'red', linewidth=fig_line_width, label='LEMS120_P5_original') # Original data.
                LEMS120_P5_plot.plot(LEMS_DataFrame.index, LEMS_DataFrame['LEMS120_P5'].fillna(0),color = 'blue', linewidth=fig_line_width, label='LEMS120_P5_processed') # Filtered data.
                LEMS120_P5_plot.plot(LEMS_LowPass.index, LEMS_LowPass['LEMS120_P5'].fillna(0),color = 'black', linewidth=fig_line_width, label='LEMS120_P5_LowPass') # Low pass curve.
                LEMS120_P5_plot.set_ylabel('LEMS120_P5', fontsize=fig_ylabel_fontsize)
                LEMS120_P5_plot.legend(loc='upper left',prop={'size':fig_legend_size})
                LEMS120_P5_plot.tick_params(axis='y', which='major', labelsize=fig_ytick_fontsize)
                LEMS120_P5_dif.plot(LEMS_dif.index, LEMS_dif['LEMS120_P5'].fillna(0),color = 'green', linewidth=fig_line_width) # Difference data.
                LEMS120_P5_dif.tick_params(axis='y', which='major', labelsize=fig_ytick_fontsize)
                LEMS120_P5_dif.set_ylabel('LEMS120_P5_dif', fontsize=fig_ylabel_fontsize)
                # Plotting LEMS120_P6 filter process.
                LEMS120_P6_plot.plot(LEMS_DataFrame0.index, LEMS_DataFrame0['LEMS120_P6'].fillna(0),color = 'red', linewidth=fig_line_width, label='LEMS120_P6_original') # Original data.
                LEMS120_P6_plot.plot(LEMS_DataFrame.index, LEMS_DataFrame['LEMS120_P6'].fillna(0),color = 'blue', linewidth=fig_line_width, label='LEMS120_P6_processed') # Filtered data.
                LEMS120_P6_plot.plot(LEMS_LowPass.index, LEMS_LowPass['LEMS120_P6'].fillna(0),color = 'black', linewidth=fig_line_width, label='LEMS120_P6_LowPass') # Low pass curve.
                LEMS120_P6_plot.set_ylabel('LEMS120_P6', fontsize=fig_ylabel_fontsize)
                LEMS120_P6_plot.legend(loc='upper left',prop={'size':fig_legend_size})
                LEMS120_P6_plot.tick_params(axis='y', which='major', labelsize=fig_ytick_fontsize)
                LEMS120_P6_dif.plot(LEMS_dif.index, LEMS_dif['LEMS120_P6'].fillna(0),color = 'green', linewidth=fig_line_width) # Difference data.
                LEMS120_P6_dif.tick_params(axis='y', which='major', labelsize=fig_ytick_fontsize)
                LEMS120_P6_dif.set_ylabel('LEMS120_P6_dif', fontsize=fig_ylabel_fontsize)
                # Plotting LEMS120_P7 filter process.
                LEMS120_P7_plot.plot(LEMS_DataFrame0.index, LEMS_DataFrame0['LEMS120_P7'].fillna(0),color = 'red', linewidth=fig_line_width, label='LEMS120_P7_original') # Original data.
                LEMS120_P7_plot.plot(LEMS_DataFrame.index, LEMS_DataFrame['LEMS120_P7'].fillna(0),color = 'blue', linewidth=fig_line_width, label='LEMS120_P7_processed') # Filtered data.
                LEMS120_P7_plot.plot(LEMS_LowPass.index, LEMS_LowPass['LEMS120_P7'].fillna(0),color = 'black', linewidth=fig_line_width, label='LEMS120_P7_LowPass') # Low pass curve.
                LEMS120_P7_plot.set_ylabel('LEMS120_P7', fontsize=fig_ylabel_fontsize)
                LEMS120_P7_plot.legend(loc='upper left',prop={'size':fig_legend_size})
                LEMS120_P7_plot.tick_params(axis='y', which='major', labelsize=fig_ytick_fontsize)
                LEMS120_P7_dif.plot(LEMS_dif.index, LEMS_dif['LEMS120_P7'].fillna(0),color = 'green', linewidth=fig_line_width) # Difference data.
                LEMS120_P7_dif.tick_params(axis='y', which='major', labelsize=fig_ytick_fontsize)
                LEMS120_P7_dif.set_ylabel('LEMS120_P7_dif', fontsize=fig_ylabel_fontsize)
                # Plotting LEMS120_P8 filter process.
                LEMS120_P8_plot.plot(LEMS_DataFrame0.index, LEMS_DataFrame0['LEMS120_P8'].fillna(0),color = 'red', linewidth=fig_line_width, label='LEMS120_P8_original') # Original data.
                LEMS120_P8_plot.plot(LEMS_DataFrame.index, LEMS_DataFrame['LEMS120_P8'].fillna(0),color = 'blue', linewidth=fig_line_width, label='LEMS120_P8_processed') # Filtered data.
                LEMS120_P8_plot.plot(LEMS_LowPass.index, LEMS_LowPass['LEMS120_P8'].fillna(0),color = 'black', linewidth=fig_line_width, label='LEMS120_P8_LowPass') # Low pass curve.
                LEMS120_P8_plot.set_ylabel('LEMS120_P8', fontsize=fig_ylabel_fontsize)
                LEMS120_P8_plot.legend(loc='upper left',prop={'size':fig_legend_size})
                LEMS120_P8_plot.tick_params(axis='y', which='major', labelsize=fig_ytick_fontsize)
                LEMS120_P8_dif.plot(LEMS_dif.index, LEMS_dif['LEMS120_P8'].fillna(0),color = 'green', linewidth=fig_line_width) # Difference data.
                LEMS120_P8_dif.tick_params(axis='y', which='major', labelsize=fig_ytick_fontsize)
                LEMS120_P8_dif.set_ylabel('LEMS120_P8_dif', fontsize=fig_ylabel_fontsize)
                # This is a shared axis for all subplot
                LEMS120_P8_dif.tick_params(axis='x', which='major', labelsize=fig_xtick_fontsize)
                # Save plot.
                fig.savefig(data_pickle_dir + '/ACE_filter_process_LEMS120_' + timeStart_str + '_' + timeEnd_str + '.eps',format='eps', dpi=500, bbox_inches='tight')
        else:
            # Keep original data.
            LEMS_DataFrame0 = LEMS_DataFrame.copy(deep=True)
        # ============================== Resample data to 1min resolution ==============================.

        n_interp_limit = 10

        # Resample BGSE data into one minute resolution.
        # Linear fit first, to make sure there is no NaN. NaN will propagate by resample.
        # Interpolate according to timestamps. Cannot handle boundary. Do not interpolate NaN longer than 10.
        BGSE_DataFrame.interpolate(method='time', inplace=True, limit=n_interp_limit)
        print('Resampling BGSE data into 1 minute resolution...')
        BGSE_DataFrame.drop_duplicates(keep='first', inplace=True) # Drop duplicated records, keep first one.
        # Resample to 1 minute resolution. New added records will be filled with NaN.
        BGSE_DataFrame = BGSE_DataFrame.resample('1T').mean()
        # Interpolate according to timestamps. Cannot handle boundary.
        BGSE_DataFrame.interpolate(method='time', inplace=True, limit=n_interp_limit)

        if not VGSE_DataFrame.empty:
            # Resample VGSE data into one minute resolution.
            # Interpolate according to timestamps. Cannot handle boundary.
            VGSE_DataFrame.interpolate(method='time', inplace=True, limit=n_interp_limit)
            print('Resampling VGSE data into 1 minute resolution...')
            VGSE_DataFrame.drop_duplicates(keep='first', inplace=True) # Drop duplicated records, keep first one.
            # Resample to 1 minute resolution. New added records will be filled with NaN.
            VGSE_DataFrame = VGSE_DataFrame.resample('1T').mean()
            # Interpolate according to timestamps. Cannot handle boundary.
            VGSE_DataFrame.interpolate(method='time', inplace=True, limit=n_interp_limit)

        if not Np_DataFrame.empty:
            # Resample Np data into one minute resolution.
            # Linear fit first, to make sure there is no NaN. NaN will propagate by resample.
            # Interpolate according to timestamps. Cannot handle boundary.
            Np_DataFrame.interpolate(method='time', inplace=True, limit=n_interp_limit)
            print('Resampling Np data into 1 minute resolution...')
            Np_DataFrame.drop_duplicates(keep='first', inplace=True) # Drop duplicated records, keep first one.
            # Resample to 1 minute resolution. New added records will be filled with NaN.
            Np_DataFrame = Np_DataFrame.resample('1T').mean()
            # Interpolate according to timestamps. Cannot handle boundary.
            Np_DataFrame.interpolate(method='time', inplace=True, limit=n_interp_limit)

        if not Tp_DataFrame.empty:
            # Resample Tp data into one minute resolution.
            # Linear fit first, to make sure there is no NaN. NaN will propagate by resample.
            # Interpolate according to timestamps. Cannot handle boundary.
            Tp_DataFrame.interpolate(method='time', inplace=True)
            print('Resampling Tp data into 1 minute resolution...')
            Tp_DataFrame.drop_duplicates(keep='first', inplace=True) # Drop duplicated records, keep first one.
            # Resample to 1 minute resolution. New added records will be filled with NaN.
            Tp_DataFrame = Tp_DataFrame.resample('1T').mean()
            # Interpolate according to timestamps. Cannot handle boundary.
            Tp_DataFrame.interpolate(method='time', inplace=True, limit=n_interp_limit)

        if not Alpha2Proton_ratio_DataFrame.empty:
            # Resample Alpha2Proton_ratio data into one minute resolution.
            # Interpolate according to timestamps. Cannot handle boundary.
            Alpha2Proton_ratio_DataFrame.interpolate(method='time', inplace=True, limit=n_interp_limit)
            print('Resampling Alpha2Proton_ratio data into 1 minute resolution...')
            Alpha2Proton_ratio_DataFrame.drop_duplicates(keep='first', inplace=True) # Drop duplicated records, keep first one.
            # Resample to 1 minute resolution. New added records will be filled with NaN.
            Alpha2Proton_ratio_DataFrame = Alpha2Proton_ratio_DataFrame.resample('1T').mean()
            # Interpolate according to timestamps. Cannot handle boundary.
            Alpha2Proton_ratio_DataFrame.interpolate(method='time', inplace=True, limit=n_interp_limit)

        if not LEMS_DataFrame.empty:
            # Resample P1_8 data into one minute resolution.
            # Linear fit first, to make sure there is no NaN. NaN will propagate by resample.
            # Interpolate according to timestamps. Cannot handle boundary.
            LEMS_DataFrame.interpolate(method='time', inplace=True)
            # Apply rolling mean to LEMS. The original resolution is 5 seconds, we want 1 minute, so use 5 point as window width.
            LEMS_DataFrame = LEMS_DataFrame.rolling(window=5,center=True).mean()
            print('Resampling P1_8 data into 1 minute resolution...')
            LEMS_DataFrame.drop_duplicates(keep='first', inplace=True) # Drop duplicated records, keep first one.
            # Resample to 1 minute resolution. New added records will be filled with NaN.
            LEMS_DataFrame = LEMS_DataFrame.resample('1T').mean()
            # Interpolate according to timestamps. Cannot handle boundary.
            LEMS_DataFrame.interpolate(method='time', inplace=True, limit=n_interp_limit)

        # Merge all DataFrames into one according to time index.
        # Calculate time range in minutes.
        timeRangeInMinutes = int((timeEnd - timeStart).total_seconds())//60
        # Generate timestamp index.
        index_datetime = np.asarray([timeStart + timedelta(minutes=x) for x in range(0, timeRangeInMinutes+1)])
        # Generate empty DataFrame according using index_datetime as index.
        GS_AllData_DataFrame = pd.DataFrame(index=index_datetime)
        # Merge all DataFrames.
        GS_AllData_DataFrame = pd.concat([GS_AllData_DataFrame, BGSE_DataFrame, VGSE_DataFrame, Np_DataFrame, Tp_DataFrame, Alpha2Proton_ratio_DataFrame, LEMS_DataFrame], axis=1)
        # Save merged DataFrame into pickle file.
        GS_AllData_DataFrame.to_pickle(data_pickle_dir + '/ACE_' + timeStart_str + '_' + timeEnd_str + '_preprocessed.p')
        
        if isCheckDataIntegrity:
            print('Checking the number of NaNs in GS_AllData_DataFrame...')
            len_GS_AllData_DataFrame = len(GS_AllData_DataFrame)
            for key in GS_AllData_DataFrame.keys():
                num_notNaN = GS_AllData_DataFrame[key].isnull().values.sum()
                percent_notNaN = 100.0 - num_notNaN * 100.0 / len_GS_AllData_DataFrame
                print('The number of NaNs in {} is {}, integrity is {}%'.format(key, num_notNaN, round(percent_notNaN, 2)))
    
        print('\nData preprocessing is done!')
        
        return GS_AllData_DataFrame

#############################################################################################################

# Choose root directory according to environment.
def setRootDir(ENV):
    return {
        'macbook'    : '/Users/jz0006/GoogleDrive/GS/',
        'bladerunner': '/home/jinlei/gs/',
        'blueshark'  : '/udrive/staff/lzhao/jinlei/gs/',
    }.get(ENV, 0) # 0 is default if ENV not found

#############################################################################################################

# Calculate the velocity of deHoffmann-Teller frame, VHT.
def findVHT(B_DF_inGSE, Vsw_DF_inGSE):
    N = len(B_DF_inGSE)
    B_square = np.square(B_DF_inGSE).sum(axis=1) # Take squre and sum row (axis=1 for row, axis=0 for column)
    KN = np.zeros((N,3,3)) # np.zeros((layer, row, column)). Right most index change first.
    for n in range(N):
        for i in range(3):
            for j in range(3):
                if i == j:
                    KN[n,i,j] = B_square.iloc[n] - B_DF_inGSE.iloc[n][i] * B_DF_inGSE.iloc[n][j]
                else:
                    KN[n,i,j] = - B_DF_inGSE.iloc[n][i] * B_DF_inGSE.iloc[n][j]
    K = np.mean(KN, axis=0) # Take average on layer (axis=1 for row, axis=2 for column).
    KVN = np.zeros((N,3)) # np.zeros((row, column)). Right most index change first.
    for n in range(N):
        KVN[n,:] = np.dot(KN[n,:], Vsw_DF_inGSE.iloc[n])
    # Average KVN over N to get KV.
    KV = np.mean(KVN, axis=0) # Take average on column.
    VHT = np.dot(np.linalg.inv(K), KV)
    return VHT
    
# Check VHT. Find correlation coefficient between E = -v X B and EHT = -VHT X B
def check_VHT(VHT, V, B):
    # Convert DataFrame to Array.
    VHT_array = np.array(VHT)
    V_array = np.array(V)
    B_array = np.array(B)
    # get EHT = B X VHT
    EHT = np.cross(B_array, VHT_array)
    # get E = B X v
    E = np.cross(B_array, V_array)
    # merge all component to 1-D array.
    EHT_1D = EHT.reshape(EHT.size)
    E_1D = E.reshape(E.size)
    mask = ~np.isnan(EHT_1D) & ~np.isnan(E_1D)
    if mask.sum()>=5:
        # slope, intercept, r_value, p_value, std_err = stats.linregress(A,B)
        # scipy.stats.linregress(x, y=None). Put VA on X-axis, V_remaining on Y-axis.
        slope, intercept, r_value, p_value, std_err = stats.linregress(EHT_1D[mask], E_1D[mask])
        # Return a numpy array.
        return slope, intercept, r_value
    else:
        return np.nan, np.nan, np.nan

#############################################################################################################

# Calculate the eignvectors and eigenvaluse of input matrix dataframe. This module is Python style.
def eigenMatrix(matrix_DataFrame, **kwargs):
    # Calculate the eigenvalues and eigenvectors of covariance matrix.
    eigenValue, eigenVector = la.eig(matrix_DataFrame) # eigen_arg are eigenvalues, and eigen_vec are eigenvectors.
    #print(eigenValue)
    #print(eigenVector)
    #print(type(eigenVector))

    # Sort the eigenvalues and arrange eigenvectors by sorted eigenvalues.
    eigenValue_i = np.argsort(eigenValue) # covM_B_eigenValue_i is sorted index of covM_B_eigenValue
    lambda3 = eigenValue[eigenValue_i[0]] # lambda3, minimum variance
    lambda2 = eigenValue[eigenValue_i[1]] # lambda2, intermediate variance.
    lambda1 = eigenValue[eigenValue_i[2]] # lambda1, maximum variance.
    eigenVector3 = pd.DataFrame(eigenVector[:, eigenValue_i[0]], columns=['minVar(lambda3)']) # Eigenvector 3, along minimum variance
    eigenVector2 = pd.DataFrame(eigenVector[:, eigenValue_i[1]], columns=['interVar(lambda2)']) # Eigenvector 2, along intermediate variance.
    eigenVector1 = pd.DataFrame(eigenVector[:, eigenValue_i[2]], columns=['maxVar(lambda1)']) # Eigenvector 1, along maximum variance.
    
    if kwargs['formXYZ']==True:
        # Form an eigenMatrix with the columns:
        # X = minimum variance direction, Y = Maximum variance direction, Z = intermediate variance direction.
        eigenMatrix = pd.concat([eigenVector3, eigenVector1, eigenVector2], axis=1)
        #print(eigenMatrix)
        eigenValues = pd.DataFrame([lambda3, lambda1, lambda2], index=['X1(min)', 'X2(max)', 'X3(inter)'], columns=['eigenValue'])
    else:
        # Form a sorted eigenMatrix using three sorted eigenvectors. Columns are eigenvectors.
        eigenMatrix = pd.concat([eigenVector3, eigenVector2, eigenVector1], axis=1)
        eigenValues = pd.DataFrame([lambda3, lambda2, lambda1], index=['lambda3', 'lambda2', 'lambda1'], columns=['eigenValue'])
    
    eigenVectorMaxVar_lambda1 = (eigenVector[:, eigenValue_i[2]])
    eigenVectorInterVar_lambda2 = (eigenVector[:, eigenValue_i[1]])
    eigenVectorMinVar_lambda3 = (eigenVector[:, eigenValue_i[0]])
    #print('eigenVectorMaxVar_lambda1 = ', eigenVectorMaxVar_lambda1) # maxVar(lambda1)
    #print('eigenVectorInterVar_lambda2 = ', eigenVectorInterVar_lambda2) # interVar(lambda2)
    #print('eigenVectorMinVar_lambda3 = ', eigenVectorMinVar_lambda3) # minVar(lambda3)

    #print(eigenMatrix)
    #exit()

    #return eigenValues, eigenMatrix
    return lambda1, lambda2, lambda3, eigenVectorMaxVar_lambda1, eigenVectorInterVar_lambda2, eigenVectorMinVar_lambda3

################################################################################################################

# Find X axis according to Z axis and V. The X axis is the projection of V on the plane perpendicular to Z axis.
# For this function, numba is slower than python.
def findXaxis(Z, V):
    #import numpy as np # Scientific calculation package.
    #from numpy import linalg as la
    Z = np.array(Z)
    V = np.array(V)
    # Both Z and V are unit vector representing the directions. They are numpy 1-D arrays.
    z1 = Z[0]; z2 = Z[1]; z3 = Z[2]; v1 = V[0]; v2 = V[1]; v3 = V[2]
    # V, Z, and X must satisfy two conditions. 1)The are co-plane. 2)X is perpendicular to Z. These two conditions
    # lead to two equations with three unknow. We can solve for x1, x2, and x3, in which x1 is arbitrary. Let x1
    # equals to 1, then normalize X.
    # 1) co-plane : (Z cross V) dot X = 0
    # 2) Z perpendicular to X : Z dot X = 0
    x1 = 1.0 # Arbitray.
    x2 = -((x1*(v2*z1*z1 - v1*z1*z2 - v3*z2*z3 + v2*z3*z3))/(v2*z1*z2 - v1*z2*z2 + v3*z1*z3 - v1*z3*z3))
    x3 = -((x1*(v3*z1*z1 + v3*z2*z2 - v1*z1*z3 - v2*z2*z3))/(v2*z1*z2 - v1*z2*z2 + v3*z1*z3 - v1*z3*z3))
    # Normalization.
    X = np.array([float(x1), float(x2), float(x3)])
    X = X/(la.norm(X))
    if X.dot(V) < 0:
        X = - X
    return X

################################################################################################################

# Given two orthnormal vectors(Z and X), find the third vector(Y) to form right-hand side frame.
# For this function, numba is slower than python.
def formRighHandFrame(X, Z): # Z cross X = Y in right hand frame.
    #import numpy as np # Scientific calculation package.
    #from numpy import linalg as la
    X = np.array(X)
    Z = np.array(Z)
    Y = np.cross(Z, X)
    Y = Y/(la.norm(Y)) # Normalize.
    return Y

################################################################################################################

# Find how many turning points in an array.
# For this function, numba is slower than python.
def turningPoints(array):
    array = np.array(array)
    dx = np.diff(array)
    dx = dx[dx != 0] # if don't remove duplicate points, will miss the turning points with duplicate values.
    return np.sum(dx[1:] * dx[:-1] < 0)

################################################################################################################

# Usage: B_inFR = B_inGSE.dot(matrix_transToFluxRopeFrame)
def angle2matrix(theta_deg, phi_deg, VHT_inGSE):
    factor_deg2rad = np.pi/180.0 # Convert degree to rad.
    # Direction cosines:
    # x = rcos(alpha) = rsin(theta)cos(phi) => cos(alpha) = sin(theta)cos(phi)
    # y = rcos(beta)  = rsin(theta)sin(phi) => cos(beta)  = sin(theta)sin(phi)
    # z = rcos(gamma) = rcos(theta)         => cos(gamma) = cos(theta)
    # Use direction cosines to construct a unit vector.
    theta_rad = factor_deg2rad * theta_deg
    phi_rad   = factor_deg2rad * phi_deg
    # Form new Z_unitVector according to direction cosines.
    Z_unitVector = np.array([np.sin(theta_rad)*np.cos(phi_rad), np.sin(theta_rad)*np.sin(phi_rad), np.cos(theta_rad)])
    # Find X axis from Z axis and -VHT.
    X_unitVector = findXaxis(Z_unitVector, -VHT_inGSE)
    # Find the Y axis to form a right-handed coordinater with X and Z.
    Y_unitVector = formRighHandFrame(X_unitVector, Z_unitVector)
    # Project B_inGSE into FluxRope Frame.
    matrix_transToFluxRopeFrame = np.array([X_unitVector, Y_unitVector, Z_unitVector]).T
    return matrix_transToFluxRopeFrame

################################################################################################################

def directionVector2angle(V):
    #print('V = {}'.format(V))
    Z = np.array([0,0,1])
    X = np.array([1,0,0])
    cos_theta = np.dot(V,Z)/la.norm(V)/la.norm(Z)
    #print('cos_theta = {}'.format(cos_theta))
    V_cast2XY = np.array([V[0], V[1], 0])
    cos_phi = np.dot(V_cast2XY,X)/la.norm(V_cast2XY)/la.norm(X)
    #print('cos_phi = {}'.format(cos_phi))
    theta_deg = np.arccos(np.clip(cos_theta, -1, 1))*180/np.pi
    phi_deg = np.arccos(np.clip(cos_phi, -1, 1))*180/np.pi
    if V[1]<0:
        phi_deg = 360 - phi_deg
    return (theta_deg, phi_deg)
    
################################################################################################################

def angle2directionVector(theta_deg, phi_deg):
    factor_deg2rad = np.pi/180.0 # Convert degree to rad.
    # Use direction cosines to construct a unit vector.
    theta_rad = factor_deg2rad * theta_deg
    phi_rad   = factor_deg2rad * phi_deg
    # Form new Z_unitVector according to direction cosines.
    Z_unitVector = np.array([np.sin(theta_rad)*np.cos(phi_rad), np.sin(theta_rad)*np.sin(phi_rad), np.cos(theta_rad)])
    return Z_unitVector
    
################################################################################################################

def flip_direction(theta_deg, phi_deg):
    new_theta_deg = 180 - theta_deg
    new_phi_deg = phi_deg + 180
    if new_phi_deg >= 360:
        new_phi_deg -= 360
    return (new_theta_deg, new_phi_deg)

################################################################################################################

# Walen test. Find the correlation coefficient and slop between the remainning velocity and Alfven speed.
# This function return the component-by-component correlation coefficient and slop of the plasma velocities
# and the Alfven velocities.
def walenTest(VA, V_remaining):
    # V_remaining reshaped time series of solar wind velocity. In km/s.
    # VA is the reshaped time series of Alfven wave. In km/s.
    # Make sure the input data is numpy.array.
    # Convert input to numpy array.
    V_remaining = np.array(V_remaining)
    VA = np.array(VA)
    mask = ~np.isnan(VA) & ~np.isnan(V_remaining)
    if mask.sum()>=5:
        # slope, intercept, r_value, p_value, std_err = stats.linregress(A,B)
        # scipy.stats.linregress(x, y=None). Put VA on X-axis, V_remaining on Y-axis.
        slope, intercept, r_value, p_value, std_err = stats.linregress(VA[mask], V_remaining[mask])
        # Return a numpy array.
        return slope, intercept, r_value
    else:
        return np.nan, np.nan, np.nan

################################################################################################################

# Loop for all directions to calculate residue, return the smallest residue and corresponding direction.
def searchFluxRopeInWindow_savgol_filter(B_DataFrame, VHT, n_theta_grid, minDuration, dt, flag_smoothA, savgol_filter_window):
    
    #t0 = datetime.now()
    print('{} - [{}~{} minutes] searching: ({} ~ {})'.format(time.ctime(), minDuration, len(B_DataFrame), B_DataFrame.index[0], B_DataFrame.index[-1]))
    #t1 = datetime.now()
    #print((t1-t0).total_seconds())
    
    # Initialization.
    # Caution: the type of return value will be different if the initial data is updated. If updated, timeRange_temp will become to tuple, plotData_dict_temp will becomes to dict, et, al.
    time_start_temp = np.nan
    time_end_temp = np.nan
    time_turn_temp = np.nan
    turnPointOnTop_temp = np.nan
    Residue_diff_temp = np.inf
    Residue_fit_temp = np.inf
    duration_temp = np.nan
    theta_temp = np.nan
    phi_temp = np.nan
    time_start, time_end, time_turn, turnPointOnTop, Residue_diff, Residue_fit, duration = getResidueForCurrentAxial(0, 0, minDuration, B_DataFrame, VHT, dt, flag_smoothA, savgol_filter_window)
    #print('For current orientation, the returned residue is {}'.format(Residue))
    #print('For current orientation, the returned duration is {}'.format(duration))
    if  Residue_diff < Residue_diff_temp:
        time_start_temp = time_start
        time_end_temp = time_end
        time_turn_temp = time_turn
        turnPointOnTop_temp = turnPointOnTop
        Residue_diff_temp = Residue_diff
        Residue_fit_temp = Residue_fit
        theta_temp = 0
        phi_temp = 0
        duration_temp = duration
    
    # This step loops all theta and phi except for theta = 0.
    thetaArray = np.linspace(0, 90, n_theta_grid+1)
    thetaArray = thetaArray[1:]
    phiArray = np.linspace(0, 360, n_theta_grid*2+1)
    phiArray = phiArray[1:]
    for theta_deg in thetaArray: # Not include theta = 0.
        for phi_deg in phiArray: # Include phi = 0.
            #print('theta_deg = {}, phi_deg = {}'.format(theta_deg, phi_deg))
            time_start, time_end, time_turn, turnPointOnTop, Residue_diff, Residue_fit, duration = getResidueForCurrentAxial(theta_deg, phi_deg, minDuration, B_DataFrame, VHT, dt, flag_smoothA, savgol_filter_window)
            #print('For current orientation, the returned residue is {}'.format(Residue))
            #print('For current orientation, the returned duration is {}'.format(duration))
            if Residue_diff < Residue_diff_temp:
                time_start_temp = time_start
                time_end_temp = time_end
                time_turn_temp = time_turn
                turnPointOnTop_temp = turnPointOnTop
                Residue_diff_temp = Residue_diff
                Residue_fit_temp = Residue_fit
                theta_temp = theta_deg
                phi_temp = phi_deg
                duration_temp = duration

    #print('Residue_diff = {}'.format(Residue_diff_temp))
    #print('Residue_fit  = {}\n'.format(Residue_fit_temp))
    # Round some results.
    #print((time_start_temp, time_turn_temp, time_end_temp, duration_temp, turnPointOnTop_temp, Residue_diff_temp, Residue_fit_temp, (theta_temp, phi_temp), (round(VHT[0],5),round(VHT[1],5),round(VHT[2],5))))
    return time_start_temp, time_turn_temp, time_end_temp, duration_temp, turnPointOnTop_temp, Residue_diff_temp, Residue_fit_temp, (theta_temp, phi_temp), (round(VHT[0],5),round(VHT[1],5),round(VHT[2],5))

################################################################################################################

# Loop for all directions to calculate residue, return the smallest residue and corresponding direction.
def searchFluxRopeInWindow(B_DataFrame, VHT, n_theta_grid, minDuration, dt, flag_smoothA):
    
    #t0 = datetime.now()
    print('{} - [{}~{} minutes] searching: ({} ~ {})'.format(time.ctime(), minDuration, len(B_DataFrame), B_DataFrame.index[0], B_DataFrame.index[-1]))
    #t1 = datetime.now()
    #print((t1-t0).total_seconds())
    
    # Initialization.
    # Caution: the type of return value will be different if the initial data is updated. If updated, timeRange_temp will become to tuple, plotData_dict_temp will becomes to dict, et, al.
    time_start_temp = np.nan
    time_end_temp = np.nan
    time_turn_temp = np.nan
    turnPointOnTop_temp = np.nan
    Residue_diff_temp = np.inf
    Residue_fit_temp = np.inf
    duration_temp = np.nan
    theta_temp = np.nan
    phi_temp = np.nan
    time_start, time_end, time_turn, turnPointOnTop, Residue_diff, Residue_fit, duration = getResidueForCurrentAxial(0, 0, minDuration, B_DataFrame, VHT, dt, flag_smoothA)
    #print('For current orientation, the returned residue is {}'.format(Residue))
    #print('For current orientation, the returned duration is {}'.format(duration))
    if  Residue_diff < Residue_diff_temp:
        time_start_temp = time_start
        time_end_temp = time_end
        time_turn_temp = time_turn
        turnPointOnTop_temp = turnPointOnTop
        Residue_diff_temp = Residue_diff
        Residue_fit_temp = Residue_fit
        theta_temp = 0
        phi_temp = 0
        duration_temp = duration
    
    # This step loops all theta and phi except for theta = 0.
    thetaArray = np.linspace(0, 90, n_theta_grid+1)
    thetaArray = thetaArray[1:]
    phiArray = np.linspace(0, 360, n_theta_grid*2+1)
    phiArray = phiArray[1:]
    for theta_deg in thetaArray: # Not include theta = 0.
        for phi_deg in phiArray: # Include phi = 0.
            #print('theta_deg = {}, phi_deg = {}'.format(theta_deg, phi_deg))
            time_start, time_end, time_turn, turnPointOnTop, Residue_diff, Residue_fit, duration = getResidueForCurrentAxial(theta_deg, phi_deg, minDuration, B_DataFrame, VHT, dt, flag_smoothA)
            #print('For current orientation, the returned residue is {}'.format(Residue))
            #print('For current orientation, the returned duration is {}'.format(duration))
            if Residue_diff < Residue_diff_temp:
                time_start_temp = time_start
                time_end_temp = time_end
                time_turn_temp = time_turn
                turnPointOnTop_temp = turnPointOnTop
                Residue_diff_temp = Residue_diff
                Residue_fit_temp = Residue_fit
                theta_temp = theta_deg
                phi_temp = phi_deg
                duration_temp = duration

    #print('Residue_diff = {}'.format(Residue_diff_temp))
    #print('Residue_fit  = {}\n'.format(Residue_fit_temp))
    # Round some results.
    #print((time_start_temp, time_turn_temp, time_end_temp, duration_temp, turnPointOnTop_temp, Residue_diff_temp, Residue_fit_temp, (theta_temp, phi_temp), (round(VHT[0],5),round(VHT[1],5),round(VHT[2],5))))
    return time_start_temp, time_turn_temp, time_end_temp, duration_temp, turnPointOnTop_temp, Residue_diff_temp, Residue_fit_temp, (theta_temp, phi_temp), (round(VHT[0],5),round(VHT[1],5),round(VHT[2],5))

################################################################################################################

# Calculate the residue for given theta and phi.
def getResidueForCurrentAxial_savgol_filter(theta_deg, phi_deg, minDuration, B_DataFrame, VHT, dt, flag_smoothA, savgol_filter_window):
    # Physics constants.
    mu0 = 4.0 * np.pi * 1e-7 #(N/A^2) magnetic constant permeability of free space vacuum permeability
    m_proton = 1.6726219e-27 # Proton mass. In kg.
    factor_deg2rad = np.pi/180.0 # Convert degree to rad.
    # Initialize
    time_start = np.nan
    time_end = np.nan
    time_turn = np.nan
    Residue_diff = np.inf
    Residue_fit = np.inf
    duration = np.nan
    turnPointOnTop = np.nan
    # Loop for half polar angle (theta(0~90 degree)), and azimuthal angle (phi(0~360 degree)) for Z axis orientations.
    # Direction cosines:
    # x = rcos(alpha) = rsin(theta)cos(phi) => cos(alpha) = sin(theta)cos(phi)
    # y = rcos(beta)  = rsin(theta)sin(phi) => cos(beta)  = sin(theta)sin(phi)
    # z = rcos(gamma) = rcos(theta)         => cos(gamma) = cos(theta)
    # Using direction cosines to form a unit vector.
    theta_rad = factor_deg2rad * theta_deg
    phi_rad   = factor_deg2rad * phi_deg
    # Form new Z_unitVector according to direction cosines.
    Z_unitVector = np.array([np.sin(theta_rad)*np.cos(phi_rad), np.sin(theta_rad)*np.sin(phi_rad), np.cos(theta_rad)])
    # Find X axis from Z axis and -VHT.
    X_unitVector = findXaxis(Z_unitVector, -VHT)
    # Find the Y axis to form a right-handed coordinater with X and Z.
    Y_unitVector = formRighHandFrame(X_unitVector, Z_unitVector)

    # Project B_DataFrame into new trial Frame.
    transToTrialFrame = np.array([X_unitVector, Y_unitVector, Z_unitVector]).T
    B_inTrialframe_DataFrame = B_DataFrame.dot(transToTrialFrame)
    # Project VHT into new trial Frame.
    VHT_inTrialframe = VHT.dot(transToTrialFrame)

    # Calculate A(x,0) by integrating By. A(x,0) = Integrate[-By(s,0)ds, {s, 0, x}], where ds = -Vht dot unit(X) dt.
    ds = - VHT_inTrialframe[0] * 1000.0 * dt # Space increment along X axis. Convert km/s to m/s.
    # Don't forget to convert km/s to m/s, and convert nT to T. By = B_inTrialframe_DataFrame[1]
    A = integrate.cumtrapz(-B_inTrialframe_DataFrame[1]*1e-9, dx=ds, initial=0)
    # Calculate Pt(x,0). Pt(x,0)=p(x,0)+Bz^2/(2mu0). By = B_inTrialframe_DataFrame[2]
    Pt = np.array((B_inTrialframe_DataFrame[2] * 1e-9)**2 / (2.0*mu0) * 1e9) # 1e9 convert unit form pa to npa.
    # Check how many turning points in original data.
    num_A_turningPoints = turningPoints(A)
    #print('num_A_turningPoints = {}'.format(num_A_turningPoints))

    if flag_smoothA == True:
        # Smooth A series to find the number of the turning point in main trend.
        # Because the small scale turning points are not important.
        # We only use A_smoothed to find turning points, when fitting Pt with A, original A is used.
        #savgol_filter_window = 9
        order = 3
        A_smoothed = savgol_filter(A, savgol_filter_window, order)
    else:
        A_smoothed = A

    # Check how many turning points in smoothed data.
    num_A_smoothed_turningPoints = turningPoints(A_smoothed)
    #print('num_A_smoothed_turningPoints = {}'.format(num_A_smoothed_turningPoints))

    # num_A_smoothed_turningPoints==0 means the A value is not double folded. It's monotonous. Skip.
    # num_A_smoothed_turningPoints > 1 means the A valuse is 3 or higher folded. Skip.
    # continue # Skip the rest commands in current iteration.
    if (num_A_smoothed_turningPoints==0)|(num_A_smoothed_turningPoints>1):
        #return timeRange, Residue, duration, plotData_dict, transToTrialFrame, turnPoint_dict # Skip the rest commands in current iteration.
        return time_start, time_end, time_turn, turnPointOnTop, Residue_diff, Residue_fit, duration # Skip the rest commands in current iteration.
    #print('Theta={}, Phi={}. Double-folding feature detected!\n'.format(theta_deg, phi_deg))
    
    # Find the boundary of A.
    A_smoothed_start = A_smoothed[0] # The first value of A.
    A_smoothed_end = A_smoothed[-1] # The last value of A.
    A_smoothed_max_index = A_smoothed.argmax() # The index of max A, return the index of first max(A).
    A_smoothed_max = A_smoothed[A_smoothed_max_index] # The max A.
    A_smoothed_min_index = A_smoothed.argmin() # The index of min A, return the index of first min(A).
    A_smoothed_min = A_smoothed[A_smoothed_min_index] # The min A.

    if (A_smoothed_min == min(A_smoothed_start, A_smoothed_end))&(A_smoothed_max == max(A_smoothed_start, A_smoothed_end)):
        # This means the A value is not double folded. It's monotonous. Skip.
        # Sometimes num_A_smoothed_turningPoints == 0 does not work well. This is double check.
        return time_start, time_end, time_turn, turnPointOnTop, Residue_diff, Residue_fit, duration
    elif abs(A_smoothed_min - ((A_smoothed_start + A_smoothed_end)/2)) < abs(A_smoothed_max - ((A_smoothed_start + A_smoothed_end)/2)):
        # This means the turning point is on the right side.
        A_turnPoint_index = A_smoothed_max_index
        turnPointOnRight = True
    elif abs(A_smoothed_min - ((A_smoothed_start + A_smoothed_end)/2)) > abs(A_smoothed_max - ((A_smoothed_start + A_smoothed_end)/2)):
        # This means the turning point is on the left side.
        A_turnPoint_index = A_smoothed_min_index
        turnPointOnLeft = True

    # Split A into two subarray from turning point.
    A_sub1 = A[:A_turnPoint_index+1]
    Pt_sub1 = Pt[:A_turnPoint_index+1] # Pick corresponding Pt according to index of A.
    A_sub2 = A[A_turnPoint_index:]
    Pt_sub2 = Pt[A_turnPoint_index:] # Pick corresponding Pt according to index of A.

    # Get time stamps.
    timeStamp = B_inTrialframe_DataFrame.index
    # Split time stamps into two subarray from turning point.
    timeStamp_sub1 = timeStamp[:A_turnPoint_index+1]
    timeStamp_sub2 = timeStamp[A_turnPoint_index:]
    
    # Keep the time of turn point and the value of Pt turn point.
    Pt_turnPoint = Pt[A_turnPoint_index]
    timeStamp_turnPoint = timeStamp[A_turnPoint_index]

    # This block is to find the time range.
    # Put two branches into DataFrame.
    Pt_vs_A_sub1_DataFrame = pd.DataFrame(np.array([Pt_sub1, timeStamp_sub1]).T, index=A_sub1, columns=['Pt_sub1','timeStamp_sub1'])
    Pt_vs_A_sub2_DataFrame = pd.DataFrame(np.array([Pt_sub2, timeStamp_sub2]).T, index=A_sub2, columns=['Pt_sub2','timeStamp_sub2'])
    # Sort by A. A is index in Pt_vs_A_sub1_DataFrame.
    Pt_vs_A_sub1_DataFrame.sort_index(ascending=True, inplace=True, kind='quicksort')
    Pt_vs_A_sub2_DataFrame.sort_index(ascending=True, inplace=True, kind='quicksort')

    # Trim two branches to get same boundary A value.
    # Note that, triming is by A value, not by lenght. After trimming, two branches may have different lengths.
    A_sub1_boundary_left = Pt_vs_A_sub1_DataFrame.index.min()
    A_sub1_boundary_right = Pt_vs_A_sub1_DataFrame.index.max()
    A_sub2_boundary_left = Pt_vs_A_sub2_DataFrame.index.min()
    A_sub2_boundary_right = Pt_vs_A_sub2_DataFrame.index.max()

    A_boundary_left = max(A_sub1_boundary_left, A_sub2_boundary_left)
    A_boundary_right = min(A_sub1_boundary_right, A_sub2_boundary_right)

    #Pt_vs_A_sub1_trimmed_DataFrame = Pt_vs_A_sub1_DataFrame.loc[A_boundary_left:A_boundary_right]
    #Pt_vs_A_sub2_trimmed_DataFrame = Pt_vs_A_sub2_DataFrame.loc[A_boundary_left:A_boundary_right]
    Pt_vs_A_sub1_trimmed_DataFrame = Pt_vs_A_sub1_DataFrame.iloc[Pt_vs_A_sub1_DataFrame.index.get_loc(A_boundary_left,method='nearest'):Pt_vs_A_sub1_DataFrame.index.get_loc(A_boundary_right,method='nearest')+1]
    Pt_vs_A_sub2_trimmed_DataFrame = Pt_vs_A_sub2_DataFrame.iloc[Pt_vs_A_sub2_DataFrame.index.get_loc(A_boundary_left,method='nearest'):Pt_vs_A_sub2_DataFrame.index.get_loc(A_boundary_right,method='nearest')+1]
    # Get the time range of trimmed A.
    timeStamp_start = min(Pt_vs_A_sub1_trimmed_DataFrame['timeStamp_sub1'].min(skipna=True), Pt_vs_A_sub2_trimmed_DataFrame['timeStamp_sub2'].min(skipna=True))
    timeStamp_end = max(Pt_vs_A_sub1_trimmed_DataFrame['timeStamp_sub1'].max(skipna=True), Pt_vs_A_sub2_trimmed_DataFrame['timeStamp_sub2'].max(skipna=True))
    #timeRange = [timeStamp_start, timeStamp_end]
    time_start = int(timeStamp_start.strftime('%Y%m%d%H%M'))
    time_end = int(timeStamp_end.strftime('%Y%m%d%H%M'))
    time_turn = int(timeStamp_turnPoint.strftime('%Y%m%d%H%M'))
    duration = int((timeStamp_end - timeStamp_start).total_seconds()/60)+1

    # Skip if shorter than minDuration.
    if duration < minDuration:
        return time_start, time_end, time_turn, turnPointOnTop, Residue_diff, Residue_fit, duration

    # Calculate two residues respectively. Residue_fit and Residue_diff.
    # Preparing for calculating Residue_fit, the residue of all data sample w.r.t. fitted PtA curve.
    # Combine two trimmed branches.
    A_sub1_array = np.array(Pt_vs_A_sub1_trimmed_DataFrame.index)
    A_sub2_array = np.array(Pt_vs_A_sub2_trimmed_DataFrame.index)
    Pt_sub1_array = np.array(Pt_vs_A_sub1_trimmed_DataFrame['Pt_sub1'])
    Pt_sub2_array = np.array(Pt_vs_A_sub2_trimmed_DataFrame['Pt_sub2'])
    # The order must be in accordance.
    Pt_array = np.concatenate((Pt_sub1_array, Pt_sub2_array))
    A_array = np.concatenate((A_sub1_array, A_sub2_array))
    # Sort index.
    sortedIndex = np.argsort(A_array)
    A_sorted_array = A_array[sortedIndex]
    Pt_sorted_array = Pt_array[sortedIndex]
    # Fit a polynomial function (3rd order). Use it to calculate residue.
    Pt_A_coeff = np.polyfit(A_array, Pt_array, 3)
    Pt_A = np.poly1d(Pt_A_coeff)

    # Preparing for calculating Residue_diff, the residue get by compare two branches.
    # Merge two subset into one DataFrame.
    Pt_vs_A_trimmed_DataFrame = pd.concat([Pt_vs_A_sub1_trimmed_DataFrame, Pt_vs_A_sub2_trimmed_DataFrame], axis=1)
    # Drop timeStamp.
    Pt_vs_A_trimmed_DataFrame.drop(['timeStamp_sub1', 'timeStamp_sub2'], axis=1, inplace=True) # axis=1 for column.
    #print('\n')
    #print('duration = {}'.format(duration))
    #print('A_boundary_left = {}'.format(A_boundary_left))
    #print('A_boundary_right = {}'.format(A_boundary_right))
    # Interpolation.
    # "TypeError: Cannot interpolate with all NaNs" can occur if the DataFrame contains columns of object dtype. Convert data to numeric type. Check data type by print(Pt_vs_A_trimmed_DataFrame.dtypes).
    for one_column in Pt_vs_A_trimmed_DataFrame:
        Pt_vs_A_trimmed_DataFrame[one_column] = pd.to_numeric(Pt_vs_A_trimmed_DataFrame[one_column], errors='coerce')
    # Interpolate according to index A.
    Pt_vs_A_trimmed_DataFrame.interpolate(method='index', axis=0, inplace=True) # axis=0:fill column-by-column
    # Drop leading and trailing NaNs. The leading NaN won't be filled by linear interpolation, however,
    # the trailing NaN will be filled by forward copy of the last non-NaN values. So, for leading NaN,
    # just use pd.dropna, and for trailing NaN, remove the duplicated values.
    Pt_vs_A_trimmed_DataFrame.dropna(inplace=True) # Drop leading NaNs.
    trailing_NaN_mask_DataFrame = (Pt_vs_A_trimmed_DataFrame.diff()!=0) # Get duplicate bool mask.
    trailing_NaN_mask = np.array(trailing_NaN_mask_DataFrame['Pt_sub1'] & trailing_NaN_mask_DataFrame['Pt_sub2'])
    Pt_vs_A_trimmed_DataFrame = Pt_vs_A_trimmed_DataFrame.iloc[trailing_NaN_mask]

    # Get Pt_max and Pt_min. They will be used to normalize Residue for both Residue_fit and Residue_diff.
    Pt_max = Pt_sorted_array.max()
    Pt_min = Pt_sorted_array.min()
    Pt_max_min_diff = abs(Pt_max - Pt_min)
    # Check if turn point is on top.
    turnPointOnTop = (Pt_turnPoint>(Pt_max-(Pt_max-Pt_min)*0.15))

    # Use two different defination to calculate Residues. # Note that, the definition of Residue_diff is different with Hu's paper. We divided it by 2 two make it comparable with Residue_fit. The definition of Residue_fit is same as that in Hu2004.
    if Pt_max_min_diff == 0:
        Residue_diff = np.inf
        Residue_fit = np.inf
    else:
        Residue_diff = 0.5 * np.sqrt((1.0/len(Pt_vs_A_trimmed_DataFrame))*((Pt_vs_A_trimmed_DataFrame['Pt_sub1'] - Pt_vs_A_trimmed_DataFrame['Pt_sub2']) ** 2).sum()) / Pt_max_min_diff
        Residue_fit = np.sqrt((1.0/len(A_array))*((Pt_sorted_array - Pt_A(A_sorted_array)) ** 2).sum()) / Pt_max_min_diff
        # Round results.
        Residue_diff = round(Residue_diff, 5)
        Residue_fit = round(Residue_fit, 5)
    
    return time_start, time_end, time_turn, turnPointOnTop, Residue_diff, Residue_fit, duration

################################################################################################################

# Calculate the residue for given theta and phi.
def getResidueForCurrentAxial(theta_deg, phi_deg, minDuration, B_DataFrame, VHT, dt, flag_smoothA):
    # Physics constants.
    global mu0 #(N/A^2) magnetic constant permeability of free space vacuum permeability
    global m_proton # Proton mass. In kg.
    global factor_deg2rad # Convert degree to rad.
    # Initialize
    time_start = np.nan
    time_end = np.nan
    time_turn = np.nan
    Residue_diff = np.inf
    Residue_fit = np.inf
    duration = np.nan
    turnPointOnTop = np.nan
    # Loop for half polar angle (theta(0~90 degree)), and azimuthal angle (phi(0~360 degree)) for Z axis orientations.
    # Direction cosines:
    # x = rcos(alpha) = rsin(theta)cos(phi) => cos(alpha) = sin(theta)cos(phi)
    # y = rcos(beta)  = rsin(theta)sin(phi) => cos(beta)  = sin(theta)sin(phi)
    # z = rcos(gamma) = rcos(theta)         => cos(gamma) = cos(theta)
    # Using direction cosines to form a unit vector.
    theta_rad = factor_deg2rad * theta_deg
    phi_rad   = factor_deg2rad * phi_deg
    # Form new Z_unitVector according to direction cosines.
    Z_unitVector = np.array([np.sin(theta_rad)*np.cos(phi_rad), np.sin(theta_rad)*np.sin(phi_rad), np.cos(theta_rad)])
    # Find X axis from Z axis and -VHT.
    X_unitVector = findXaxis(Z_unitVector, -VHT)
    # Find the Y axis to form a right-handed coordinater with X and Z.
    Y_unitVector = formRighHandFrame(X_unitVector, Z_unitVector)

    # Project B_DataFrame into new trial Frame.
    transToTrialFrame = np.array([X_unitVector, Y_unitVector, Z_unitVector]).T
    B_inTrialframe_DataFrame = B_DataFrame.dot(transToTrialFrame)
    # Project VHT into new trial Frame.
    VHT_inTrialframe = VHT.dot(transToTrialFrame)

    # Calculate A(x,0) by integrating By. A(x,0) = Integrate[-By(s,0)ds, {s, 0, x}], where ds = -Vht dot unit(X) dt.
    ds = - VHT_inTrialframe[0] * 1000.0 * dt # Space increment along X axis. Convert km/s to m/s.
    # Don't forget to convert km/s to m/s, and convert nT to T. By = B_inTrialframe_DataFrame[1]
    A = integrate.cumtrapz(-B_inTrialframe_DataFrame[1]*1e-9, dx=ds, initial=0)
    # Calculate Pt(x,0). Pt(x,0)=p(x,0)+Bz^2/(2mu0). By = B_inTrialframe_DataFrame[2]
    Pt = np.array((B_inTrialframe_DataFrame[2] * 1e-9)**2 / (2.0*mu0) * 1e9) # 1e9 convert unit form pa to npa.
    # Check how many turning points in original data.
    num_A_turningPoints = turningPoints(A)
    #print('num_A_turningPoints = {}'.format(num_A_turningPoints))

    '''
    if flag_smoothA == True:
        # Smooth A series to find the number of the turning point in main trend.
        # Because the small scale turning points are not important.
        # We only use A_smoothed to find turning points, when fitting Pt with A, original A is used.
        #savgol_filter_window = 9
        order = 3
        A_smoothed = savgol_filter(A, savgol_filter_window, order)
    else:
        A_smoothed = A
    '''
    
    if flag_smoothA == True:
        # Smooth A series to find the number of the turning point in main trend.
        # Because the small scale turning points are not important.
        # We only use A_smoothed to find turning points, when fitting Pt with A, original A is used.
        # Firstly, downsample A to 20 points, then apply savgol_filter, then upsample to original data points number.
        
        #t1 = time.time()
        
        index_A = range(len(A))
        # Downsample A to 20 points.
        index_downsample = np.linspace(index_A[0],index_A[-1], 20)
        A_downsample = np.interp(index_downsample, index_A, A)
        # Apply savgol_filter.
        A_downsample = savgol_filter(A_downsample, 7, 3) # 7 is smooth window size, 3 is polynomial order.
        # Upsample to original data points amount.
        A_upsample = np.interp(index_A, index_downsample, A_downsample)

        # The smoothed A is just upsampled A.
        A_smoothed = A_upsample
        
        #t2 = time.time()
        #print('dt = {} ms'.format((t2-t1)*1000.0))
    else:
        A_smoothed = A
    
    '''
    plt.ion()
    plt.plot(index_A, A, 'r.', index_A, A_smoothed, 'g.-')#, X0_interp_dense, A0_interp_dense, 'o')
    #plt.legend(['A', 'A_smoothed'])
    plt.pause(0.01)
    plt.gcf().clear()
    #plt.close()
    #exit()
    '''

    # Check how many turning points in smoothed data.
    num_A_smoothed_turningPoints = turningPoints(A_smoothed)
    #print('num_A_smoothed_turningPoints = {}'.format(num_A_smoothed_turningPoints))

    # num_A_smoothed_turningPoints==0 means the A value is not double folded. It's monotonous. Skip.
    # num_A_smoothed_turningPoints > 1 means the A valuse is 3 or higher folded. Skip.
    # continue # Skip the rest commands in current iteration.
    if (num_A_smoothed_turningPoints==0)|(num_A_smoothed_turningPoints>1):
        #return timeRange, Residue, duration, plotData_dict, transToTrialFrame, turnPoint_dict # Skip the rest commands in current iteration.
        return time_start, time_end, time_turn, turnPointOnTop, Residue_diff, Residue_fit, duration # Skip the rest commands in current iteration.
    #print('Theta={}, Phi={}. Double-folding feature detected!\n'.format(theta_deg, phi_deg))
    
    # Find the boundary of A.
    A_smoothed_start = A_smoothed[0] # The first value of A.
    A_smoothed_end = A_smoothed[-1] # The last value of A.
    A_smoothed_max_index = A_smoothed.argmax() # The index of max A, return the index of first max(A).
    A_smoothed_max = A_smoothed[A_smoothed_max_index] # The max A.
    A_smoothed_min_index = A_smoothed.argmin() # The index of min A, return the index of first min(A).
    A_smoothed_min = A_smoothed[A_smoothed_min_index] # The min A.

    if (A_smoothed_min == min(A_smoothed_start, A_smoothed_end))&(A_smoothed_max == max(A_smoothed_start, A_smoothed_end)):
        # This means the A value is not double folded. It's monotonous. Skip.
        # Sometimes num_A_smoothed_turningPoints == 0 does not work well. This is double check.
        return time_start, time_end, time_turn, turnPointOnTop, Residue_diff, Residue_fit, duration
    elif abs(A_smoothed_min - ((A_smoothed_start + A_smoothed_end)/2)) < abs(A_smoothed_max - ((A_smoothed_start + A_smoothed_end)/2)):
        # This means the turning point is on the right side.
        A_turnPoint_index = A_smoothed_max_index
        turnPointOnRight = True
    elif abs(A_smoothed_min - ((A_smoothed_start + A_smoothed_end)/2)) > abs(A_smoothed_max - ((A_smoothed_start + A_smoothed_end)/2)):
        # This means the turning point is on the left side.
        A_turnPoint_index = A_smoothed_min_index
        turnPointOnLeft = True

    # Split A into two subarray from turning point.
    A_sub1 = A[:A_turnPoint_index+1]
    Pt_sub1 = Pt[:A_turnPoint_index+1] # Pick corresponding Pt according to index of A.
    A_sub2 = A[A_turnPoint_index:]
    Pt_sub2 = Pt[A_turnPoint_index:] # Pick corresponding Pt according to index of A.

    # Get time stamps.
    timeStamp = B_inTrialframe_DataFrame.index
    # Split time stamps into two subarray from turning point.
    timeStamp_sub1 = timeStamp[:A_turnPoint_index+1]
    timeStamp_sub2 = timeStamp[A_turnPoint_index:]
    
    # Keep the time of turn point and the value of Pt turn point.
    Pt_turnPoint = Pt[A_turnPoint_index]
    timeStamp_turnPoint = timeStamp[A_turnPoint_index]

    # This block is to find the time range.
    # Put two branches into DataFrame.
    Pt_vs_A_sub1_DataFrame = pd.DataFrame(np.array([Pt_sub1, timeStamp_sub1]).T, index=A_sub1, columns=['Pt_sub1','timeStamp_sub1'])
    Pt_vs_A_sub2_DataFrame = pd.DataFrame(np.array([Pt_sub2, timeStamp_sub2]).T, index=A_sub2, columns=['Pt_sub2','timeStamp_sub2'])
    # Sort by A. A is index in Pt_vs_A_sub1_DataFrame.
    Pt_vs_A_sub1_DataFrame.sort_index(ascending=True, inplace=True, kind='quicksort')
    Pt_vs_A_sub2_DataFrame.sort_index(ascending=True, inplace=True, kind='quicksort')

    # Trim two branches to get same boundary A value.
    # Note that, triming is by A value, not by lenght. After trimming, two branches may have different lengths.
    A_sub1_boundary_left = Pt_vs_A_sub1_DataFrame.index.min()
    A_sub1_boundary_right = Pt_vs_A_sub1_DataFrame.index.max()
    A_sub2_boundary_left = Pt_vs_A_sub2_DataFrame.index.min()
    A_sub2_boundary_right = Pt_vs_A_sub2_DataFrame.index.max()

    A_boundary_left = max(A_sub1_boundary_left, A_sub2_boundary_left)
    A_boundary_right = min(A_sub1_boundary_right, A_sub2_boundary_right)

    #Pt_vs_A_sub1_trimmed_DataFrame = Pt_vs_A_sub1_DataFrame.loc[A_boundary_left:A_boundary_right]
    #Pt_vs_A_sub2_trimmed_DataFrame = Pt_vs_A_sub2_DataFrame.loc[A_boundary_left:A_boundary_right]
    Pt_vs_A_sub1_trimmed_DataFrame = Pt_vs_A_sub1_DataFrame.iloc[Pt_vs_A_sub1_DataFrame.index.get_loc(A_boundary_left,method='nearest'):Pt_vs_A_sub1_DataFrame.index.get_loc(A_boundary_right,method='nearest')+1]
    Pt_vs_A_sub2_trimmed_DataFrame = Pt_vs_A_sub2_DataFrame.iloc[Pt_vs_A_sub2_DataFrame.index.get_loc(A_boundary_left,method='nearest'):Pt_vs_A_sub2_DataFrame.index.get_loc(A_boundary_right,method='nearest')+1]
    # Get the time range of trimmed A.
    timeStamp_start = min(Pt_vs_A_sub1_trimmed_DataFrame['timeStamp_sub1'].min(skipna=True), Pt_vs_A_sub2_trimmed_DataFrame['timeStamp_sub2'].min(skipna=True))
    timeStamp_end = max(Pt_vs_A_sub1_trimmed_DataFrame['timeStamp_sub1'].max(skipna=True), Pt_vs_A_sub2_trimmed_DataFrame['timeStamp_sub2'].max(skipna=True))
    #timeRange = [timeStamp_start, timeStamp_end]
    time_start = int(timeStamp_start.strftime('%Y%m%d%H%M'))
    time_end = int(timeStamp_end.strftime('%Y%m%d%H%M'))
    time_turn = int(timeStamp_turnPoint.strftime('%Y%m%d%H%M'))
    duration = int((timeStamp_end - timeStamp_start).total_seconds()/60)+1

    # Skip if shorter than minDuration.
    if duration < minDuration:
        return time_start, time_end, time_turn, turnPointOnTop, Residue_diff, Residue_fit, duration

    # Calculate two residues respectively. Residue_fit and Residue_diff.
    # Preparing for calculating Residue_fit, the residue of all data sample w.r.t. fitted PtA curve.
    # Combine two trimmed branches.
    A_sub1_array = np.array(Pt_vs_A_sub1_trimmed_DataFrame.index)
    A_sub2_array = np.array(Pt_vs_A_sub2_trimmed_DataFrame.index)
    Pt_sub1_array = np.array(Pt_vs_A_sub1_trimmed_DataFrame['Pt_sub1'])
    Pt_sub2_array = np.array(Pt_vs_A_sub2_trimmed_DataFrame['Pt_sub2'])
    # The order must be in accordance.
    Pt_array = np.concatenate((Pt_sub1_array, Pt_sub2_array))
    A_array = np.concatenate((A_sub1_array, A_sub2_array))
    # Sort index.
    sortedIndex = np.argsort(A_array)
    A_sorted_array = A_array[sortedIndex]
    Pt_sorted_array = Pt_array[sortedIndex]
    # Fit a polynomial function (3rd order). Use it to calculate residue.
    Pt_A_coeff = np.polyfit(A_array, Pt_array, 3)
    Pt_A = np.poly1d(Pt_A_coeff)

    # Preparing for calculating Residue_diff, the residue get by compare two branches.
    # Merge two subset into one DataFrame.
    Pt_vs_A_trimmed_DataFrame = pd.concat([Pt_vs_A_sub1_trimmed_DataFrame, Pt_vs_A_sub2_trimmed_DataFrame], axis=1)
    # Drop timeStamp.
    Pt_vs_A_trimmed_DataFrame.drop(['timeStamp_sub1', 'timeStamp_sub2'], axis=1, inplace=True) # axis=1 for column.
    #print('\n')
    #print('duration = {}'.format(duration))
    #print('A_boundary_left = {}'.format(A_boundary_left))
    #print('A_boundary_right = {}'.format(A_boundary_right))
    # Interpolation.
    # "TypeError: Cannot interpolate with all NaNs" can occur if the DataFrame contains columns of object dtype. Convert data to numeric type. Check data type by print(Pt_vs_A_trimmed_DataFrame.dtypes).
    for one_column in Pt_vs_A_trimmed_DataFrame:
        Pt_vs_A_trimmed_DataFrame[one_column] = pd.to_numeric(Pt_vs_A_trimmed_DataFrame[one_column], errors='coerce')
    # Interpolate according to index A.
    Pt_vs_A_trimmed_DataFrame.interpolate(method='index', axis=0, inplace=True) # axis=0:fill column-by-column
    # Drop leading and trailing NaNs. The leading NaN won't be filled by linear interpolation, however,
    # the trailing NaN will be filled by forward copy of the last non-NaN values. So, for leading NaN,
    # just use pd.dropna, and for trailing NaN, remove the duplicated values.
    Pt_vs_A_trimmed_DataFrame.dropna(inplace=True) # Drop leading NaNs.
    trailing_NaN_mask_DataFrame = (Pt_vs_A_trimmed_DataFrame.diff()!=0) # Get duplicate bool mask.
    trailing_NaN_mask = np.array(trailing_NaN_mask_DataFrame['Pt_sub1'] & trailing_NaN_mask_DataFrame['Pt_sub2'])
    Pt_vs_A_trimmed_DataFrame = Pt_vs_A_trimmed_DataFrame.iloc[trailing_NaN_mask]

    # Get Pt_max and Pt_min. They will be used to normalize Residue for both Residue_fit and Residue_diff.
    Pt_max = Pt_sorted_array.max()
    Pt_min = Pt_sorted_array.min()
    Pt_max_min_diff = abs(Pt_max - Pt_min)
    # Check if turn point is on top.
    turnPointOnTop = (Pt_turnPoint>(Pt_max-(Pt_max-Pt_min)*0.15))

    # Use two different defination to calculate Residues. # Note that, the definition of Residue_diff is different with Hu's paper. We divided it by 2 two make it comparable with Residue_fit. The definition of Residue_fit is same as that in Hu2004.
    if Pt_max_min_diff == 0:
        Residue_diff = np.inf
        Residue_fit = np.inf
    else:
        Residue_diff = 0.5 * np.sqrt((1.0/len(Pt_vs_A_trimmed_DataFrame))*((Pt_vs_A_trimmed_DataFrame['Pt_sub1'] - Pt_vs_A_trimmed_DataFrame['Pt_sub2']) ** 2).sum()) / Pt_max_min_diff
        Residue_fit = np.sqrt((1.0/len(A_array))*((Pt_sorted_array - Pt_A(A_sorted_array)) ** 2).sum()) / Pt_max_min_diff
        # Round results.
        Residue_diff = round(Residue_diff, 5)
        Residue_fit = round(Residue_fit, 5)
    
    return time_start, time_end, time_turn, turnPointOnTop, Residue_diff, Residue_fit, duration

################################################################################################################

def detect_flux_rope_savgol_filter(data_DF, duration_range_tuple, search_result_dir, **kwargs):
    # Get Magnetic field slice.
    B_DataFrame = data_DF.ix[:,['Bx', 'By', 'Bz']] # Produce a reference.
    # Get the solar wind slice.
    Vsw_DataFrame = data_DF.ix[:,['Vx', 'Vy', 'Vz']] # Produce a reference.
    # Get the proton number density slice.
    #Np_DataFrame = GS_DataFrame.ix[:,['Np']] # Produce a reference.
    
    # Get start and end time.
    datetimeStart = data_DF.index[0]
    datetimeEnd = data_DF.index[-1]

    # Multiprocessing
    num_cpus = multiprocessing.cpu_count()
    max_processes = num_cpus
    print '\nTotol CPU cores on this node = ', num_cpus
    # Create a multiprocessing pool with safe_lock.
    pool = multiprocessing.Pool(processes=max_processes)
    # Create a list to save result.
    results = []

    # Apply GS detection in sliding window.
    # Set searching parameters.
    n_theta_grid = 9 # theta grid number. 90/9=10, d_theta=10(degree); 90/12=7.5, d_theta=7.5(degree)
    if 'n_theta_grid' in kwargs:
        n_theta_grid = kwargs['n_theta_grid']
    print('\nGrid size: d_theta_deg = {}, d_phi_deg = {}'.format(90/n_theta_grid, 180/n_theta_grid))
    # First integer in tuple is minimum duration threshold, second integer in tuple is searching window width.
    # duration_range_tuple = ((20,30), (30,40), (40,50), (50,60)) #
    print('\nDuration range tuple is: {}'.format(duration_range_tuple))
    search_result_raw_true = {}
    search_result_raw_false = {}
    totalStartTime = datetime.now()
    for duration_range in duration_range_tuple: # Loop different window width.
        startTime = datetime.now()
       
        print('\n{}'.format(time.ctime()))
        minDuration = duration_range[0]
        maxDuration = duration_range[1]
        print('Duration : {} ~ {} minutes.'.format(minDuration, maxDuration))

        # Choose a flexible savgol filter window width based on the length of minDuration.
        half_minDuration = minDuration//2
        half_maxDuration = maxDuration//2
        if (half_minDuration) % 2 == 0: # filter window must be odd.
            savgol_filter_window = half_minDuration + 1
        else:
            savgol_filter_window = half_minDuration
        print('savgol_filter_window = {}'.format(savgol_filter_window))

        flag_smoothA = True
        # The maximum gap tolerance is up to 30% of total points count.
        interp_limit = int(math.ceil(minDuration*3.0/10)) # Flexible interpolation limit based on window length.
        print('interp_limit = {}'.format(interp_limit))

        # Sliding window. If half_window=15, len(DataFrame)=60, range(15,45)=[15,...,44].
        for FluxRopeCenter in range(half_maxDuration, len(B_DataFrame) - half_maxDuration): # in minutes.
            indexFluxRopeStart = FluxRopeCenter - half_maxDuration
            indexFluxRopeEnd = FluxRopeCenter + half_maxDuration
            
            # Grab the B slice within the window. Change the slice will change the original DataFrame.
            B_inWindow = B_DataFrame.iloc[indexFluxRopeStart : indexFluxRopeEnd] # End is not included.
            
            # If there is any NaN in this range, try to interpolate.
            if B_inWindow.isnull().values.sum():
                B_inWindow_copy = B_inWindow.copy(deep=True)
                # For example, limit=3 means only interpolate the gap shorter than 4.
                B_inWindow_copy.interpolate(method='time', limit=interp_limit, inplace=True)
                if B_inWindow_copy.isnull().values.sum():
                    #print('Encounter NaN in B field data, skip this iteration.')
                    continue # If NaN still exists, skip this loop.
                else:
                    B_inWindow = B_inWindow_copy

            # Grab the Vsw slice within the window. Change the slice will change the original DataFrame.
            Vsw_inWindow = Vsw_DataFrame.iloc[indexFluxRopeStart : indexFluxRopeEnd+1]
            # If there is any NaN in this range, try to interpolate.
            if Vsw_inWindow.isnull().values.sum():
                Vsw_inWindow_copy = Vsw_inWindow.copy(deep=True)
                # limit=3 means only interpolate the gap shorter than 4.
                Vsw_inWindow_copy.interpolate(method='time', limit=interp_limit, inplace=True)
                if Vsw_inWindow_copy.isnull().values.sum():
                    #print('Encounter NaN in Vsw data, skip this iteration.')
                    continue # If NaN still exists, skip this loop.
                else:
                    Vsw_inWindow = Vsw_inWindow_copy
                    
            # Grab the Np slice within the window. Change the slice will change the original DataFrame.
            # Np_inWindow = Np_DataFrame.iloc[indexFluxRopeStart : indexFluxRopeEnd+1]
            
            # Calculate VHT in GSE frame.
            #VHT_inGSE = findVHT(B_inWindow, Vsw_inWindow) # Very slow.
            # Calculating VHT takes very long time(0.02748s for 14 data points), we use mean Vsw as VHT.
            VHT_inGSE = np.array(Vsw_inWindow.mean())
            
            # Return value: timeRange, Residue, orientation
            dt = 60.0 # For 1 minute resolution data, the time increment is 60 seconds.
            result_temp = pool.apply_async(searchFluxRopeInWindow, args=(B_inWindow, VHT_inGSE, n_theta_grid, minDuration, dt, flag_smoothA, savgol_filter_window,))
            # print(result_temp.get()) # This statement will cause IO very slow.
            results.append(result_temp)
            # DO NOT unpack result here. It will block IO. Unpack in bulk.

        # Next we are going to save file We have to wait for all worker processes to finish.
        # Block main process to wait for worker processes to finish. This while loop will execute almost immediately when the innner for loop goes through. The inner for loop is non-blocked, so it finish in seconds.
        while len(pool._cache)!=0:
            #print('{} - Waiting... There are {} worker processes in pool.'.format(time.ctime(), len(pool._cache)))
            time.sleep(1)
        print('{} - len(pool._cache) = {}'.format(time.ctime(), len(pool._cache)))
        print('{} - Duration range {}~{} minutes is completed!'.format(time.ctime(), minDuration, maxDuration))

        # Save result. One file per window size.
        results_true_tuple_list = []
        results_false_tuple_list = []
        # Unpack results. Convert to tuple, and put into list.
        for one_result in results:
            results_tuple_temp = (one_result.get())
            #print(results_tuple_temp)
            if not np.isinf(results_tuple_temp[5]): # Check residue.
                #print(results_tuple_temp)
                if results_tuple_temp[4]: #if True, turn point on top.
                    results_true_tuple_list.append(results_tuple_temp)
                else: # Turn point on bottom.
                    results_false_tuple_list.append(results_tuple_temp)
        # Save results to dictionary. One key per window size.
        key_temp = str(minDuration) + '~' + str(maxDuration)
        search_result_raw_true[key_temp] = results_true_tuple_list
        search_result_raw_false[key_temp] = results_false_tuple_list

        # Empty container results[].
        results = []

        endTime = datetime.now()
        time_spent_in_seconds = (endTime - startTime).total_seconds()
        print('Time spent on this window: {} seconds ({} minutes).'.format(time_spent_in_seconds, round(time_spent_in_seconds/60.0, 2)))

    # Close pool, prevent new worker process from joining.
    pool.close()
    # Block caller process until workder processes terminate.
    pool.join()

    totalEndTime = datetime.now()
    time_spent_in_seconds = (totalEndTime - totalStartTime).total_seconds()

    print('\n{} - All duration ranges are completed!'.format(time.ctime()))
    print('\nSaving search result...')
    search_result_raw = {'true':search_result_raw_true, 'false':search_result_raw_false, 'timeRange':{'datetimeStart':datetimeStart, 'datetimeEnd':datetimeEnd}}
    search_result_raw_filename = search_result_dir + '/search_result_raw.p'
    pickle.dump(search_result_raw, open(search_result_raw_filename, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
    print('Done!')
    print('\nTotal CPU cores: {}.'.format(num_cpus))
    print('Max number of workder process in pool: {}.'.format(max_processes))
    print('Total Time spent: {} seconds ({} minutes).\n'.format(time_spent_in_seconds, round(time_spent_in_seconds/60.0, 2)))
    
    return search_result_raw

################################################################################################################

def detect_flux_rope_single_processor(data_DF, duration_range_tuple, search_result_dir, **kwargs):
    
    flag_smoothA = True
    # Design the Buterworth filter.
    N  = 2    # Filter order
    Wn = 0.03 # Cutfrequency.
    filter_param = sp.signal.butter(N, Wn, btype='lowpass')
    #B = filter_param[0]
    #A = filter_param[1]
    
    # Get Magnetic field slice.
    B_DataFrame = data_DF.ix[:,['Bx', 'By', 'Bz']] # Produce a reference.
    # Get the solar wind slice.
    Vsw_DataFrame = data_DF.ix[:,['Vx', 'Vy', 'Vz']] # Produce a reference.
    # Get the proton number density slice.
    #Np_DataFrame = GS_DataFrame.ix[:,['Np']] # Produce a reference.
    
    # Get start and end time.
    datetimeStart = data_DF.index[0]
    datetimeEnd = data_DF.index[-1]

    # Multiprocessing
    num_cpus = multiprocessing.cpu_count()
    max_processes = num_cpus
    print '\nTotol CPU cores on this node = ', num_cpus
    # Create a multiprocessing pool with safe_lock.
    pool = multiprocessing.Pool(processes=max_processes)
    # Create a list to save result.
    results = []

    # Apply GS detection in sliding window.
    # Set searching parameters.
    n_theta_grid = 9 # theta grid number. 90/9=10, d_theta=10(degree); 90/12=7.5, d_theta=7.5(degree)
    if 'n_theta_grid' in kwargs:
        n_theta_grid = kwargs['n_theta_grid']
    print('\nGrid size: d_theta_deg = {}, d_phi_deg = {}'.format(90/n_theta_grid, 180/n_theta_grid))
    # First integer in tuple is minimum duration threshold, second integer in tuple is searching window width.
    # duration_range_tuple = ((20,30), (30,40), (40,50), (50,60)) #
    print('\nDuration range tuple is: {}'.format(duration_range_tuple))
    search_result_raw_true = {}
    search_result_raw_false = {}
    totalStartTime = datetime.now()
    for duration_range in duration_range_tuple: # Loop different window width.
        startTime = datetime.now()
       
        print('\n{}'.format(time.ctime()))
        minDuration = duration_range[0]
        maxDuration = duration_range[1]
        print('Duration : {} ~ {} minutes.'.format(minDuration, maxDuration))
        
        '''
        # Choose a flexible savgol filter window width based on the length of minDuration.
        half_minDuration = minDuration//2
        half_maxDuration = maxDuration//2
        if (half_minDuration) % 2 == 0: # filter window must be odd.
            savgol_filter_window = half_minDuration + 1
        else:
            savgol_filter_window = half_minDuration
        print('savgol_filter_window = {}'.format(savgol_filter_window))
        '''
        
        # The maximum gap tolerance is up to 30% of total points count.
        interp_limit = int(math.ceil(minDuration*3.0/10)) # Flexible interpolation limit based on window length.
        print('interp_limit = {}'.format(interp_limit))
        
        '''
        # Sliding window. If half_window=15, len(DataFrame)=60, range(15,45)=[15,...,44].
        for FluxRopeCenter in range(half_maxDuration, len(B_DataFrame) - half_maxDuration): # in minutes.
            indexFluxRopeStart = FluxRopeCenter - half_maxDuration
            indexFluxRopeEnd = FluxRopeCenter + half_maxDuration
        '''
        # Sliding window. If half_window=15, len(DataFrame)=60, range(15,45)=[15,...,44].
        for indexFluxRopeStart in xrange(len(B_DataFrame) - maxDuration): # in minutes.
            indexFluxRopeEnd = indexFluxRopeStart + maxDuration - 1 # The end point is included, so -1.   
            # Grab the B slice within the window. Change the slice will change the original DataFrame.
            B_inWindow = B_DataFrame.iloc[indexFluxRopeStart : indexFluxRopeEnd + 1] # End is not included.
            
            # If there is any NaN in this range, try to interpolate.
            if B_inWindow.isnull().values.sum():
                B_inWindow_copy = B_inWindow.copy(deep=True)
                # For example, limit=3 means only interpolate the gap shorter than 4.
                B_inWindow_copy.interpolate(method='time', limit=interp_limit, inplace=True)
                if B_inWindow_copy.isnull().values.sum():
                    #print('Encounter NaN in B field data, skip this iteration.')
                    continue # If NaN still exists, skip this loop.
                else:
                    B_inWindow = B_inWindow_copy

            # Grab the Vsw slice within the window. Change the slice will change the original DataFrame.
            Vsw_inWindow = Vsw_DataFrame.iloc[indexFluxRopeStart : indexFluxRopeEnd + 1]
            # If there is any NaN in this range, try to interpolate.
            if Vsw_inWindow.isnull().values.sum():
                Vsw_inWindow_copy = Vsw_inWindow.copy(deep=True)
                # limit=3 means only interpolate the gap shorter than 4.
                Vsw_inWindow_copy.interpolate(method='time', limit=interp_limit, inplace=True)
                if Vsw_inWindow_copy.isnull().values.sum():
                    #print('Encounter NaN in Vsw data, skip this iteration.')
                    continue # If NaN still exists, skip this loop.
                else:
                    Vsw_inWindow = Vsw_inWindow_copy
                    
            # Grab the Np slice within the window. Change the slice will change the original DataFrame.
            # Np_inWindow = Np_DataFrame.iloc[indexFluxRopeStart : indexFluxRopeEnd+1]
            
            # Calculate VHT in GSE frame.
            #VHT_inGSE = findVHT(B_inWindow, Vsw_inWindow) # Very slow.
            # Calculating VHT takes very long time(0.02748s for 14 data points), we use mean Vsw as VHT.
            VHT_inGSE = np.array(Vsw_inWindow.mean())
            
            # Return value: timeRange, Residue, orientation
            dt = 60.0 # For 1 minute resolution data, the time increment is 60 seconds.
            result_temp = searchFluxRopeInWindow(B_inWindow, VHT_inGSE, n_theta_grid, minDuration, dt, flag_smoothA)
            # print(result_temp.get()) # This statement will cause IO very slow.
            results.append(result_temp)
            # DO NOT unpack result here. It will block IO. Unpack in bulk.


        # Save result. One file per window size.
        results_true_tuple_list = []
        results_false_tuple_list = []
        # Unpack results. Convert to tuple, and put into list.
        for one_result in results:
            results_tuple_temp = one_result
            #print(results_tuple_temp)
            if not np.isinf(results_tuple_temp[5]): # Check residue.
                #print(results_tuple_temp)
                if results_tuple_temp[4]: #if True, turn point on top.
                    results_true_tuple_list.append(results_tuple_temp)
                else: # Turn point on bottom.
                    results_false_tuple_list.append(results_tuple_temp)
        # Save results to dictionary. One key per window size.
        key_temp = str(minDuration) + '~' + str(maxDuration)
        search_result_raw_true[key_temp] = results_true_tuple_list
        search_result_raw_false[key_temp] = results_false_tuple_list

        # Empty container results[].
        results = []

        endTime = datetime.now()
        time_spent_in_seconds = (endTime - startTime).total_seconds()
        print('Time spent on this window: {} seconds ({} minutes).'.format(time_spent_in_seconds, round(time_spent_in_seconds/60.0, 2)))


    totalEndTime = datetime.now()
    time_spent_in_seconds = (totalEndTime - totalStartTime).total_seconds()

    print('\n{} - All duration ranges are completed!'.format(time.ctime()))
    print('\nSaving search result...')
    search_result_raw = {'true':search_result_raw_true, 'false':search_result_raw_false, 'timeRange':{'datetimeStart':datetimeStart, 'datetimeEnd':datetimeEnd}}
    search_result_raw_filename = search_result_dir + '/search_result_raw.p'
    pickle.dump(search_result_raw, open(search_result_raw_filename, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
    print('Done!')
    print('\nTotal CPU cores: {}.'.format(num_cpus))
    print('Max number of workder process in pool: {}.'.format(max_processes))
    print('Total Time spent: {} seconds ({} minutes).\n'.format(time_spent_in_seconds, round(time_spent_in_seconds/60.0, 2)))
    
    return search_result_raw

################################################################################################################

def detect_flux_rope(data_DF, duration_range_tuple, search_result_dir, **kwargs):
    
    flag_smoothA = True
    
    # Get Magnetic field slice.
    B_DataFrame = data_DF.ix[:,['Bx', 'By', 'Bz']] # Produce a reference.
    # Get the solar wind slice.
    Vsw_DataFrame = data_DF.ix[:,['Vx', 'Vy', 'Vz']] # Produce a reference.
    # Get the proton number density slice.
    #Np_DataFrame = GS_DataFrame.ix[:,['Np']] # Produce a reference.
    
    # Get start and end time.
    datetimeStart = data_DF.index[0]
    datetimeEnd = data_DF.index[-1]

    # Multiprocessing
    num_cpus = multiprocessing.cpu_count()
    max_processes = num_cpus
    print '\nTotol CPU cores on this node = ', num_cpus
    # Create a multiprocessing pool with safe_lock.
    pool = multiprocessing.Pool(processes=max_processes)
    # Create a list to save result.
    results = []

    # Apply GS detection in sliding window.
    # Set searching parameters.
    n_theta_grid = 9 # theta grid number. 90/9=10, d_theta=10(degree); 90/12=7.5, d_theta=7.5(degree)
    if 'n_theta_grid' in kwargs:
        n_theta_grid = kwargs['n_theta_grid']
    print('\nGrid size: d_theta_deg = {}, d_phi_deg = {}'.format(90/n_theta_grid, 180/n_theta_grid))
    # First integer in tuple is minimum duration threshold, second integer in tuple is searching window width.
    # duration_range_tuple = ((20,30), (30,40), (40,50), (50,60)) #
    print('\nDuration range tuple is: {}'.format(duration_range_tuple))
    search_result_raw_true = {}
    search_result_raw_false = {}
    totalStartTime = datetime.now()
    for duration_range in duration_range_tuple: # Loop different window width.
        startTime = datetime.now()
       
        print('\n{}'.format(time.ctime()))
        minDuration = duration_range[0]
        maxDuration = duration_range[1]
        print('Duration : {} ~ {} minutes.'.format(minDuration, maxDuration))
        
        '''
        # Choose a flexible savgol filter window width based on the length of minDuration.
        half_minDuration = minDuration//2
        half_maxDuration = maxDuration//2
        if (half_minDuration) % 2 == 0: # filter window must be odd.
            savgol_filter_window = half_minDuration + 1
        else:
            savgol_filter_window = half_minDuration
        print('savgol_filter_window = {}'.format(savgol_filter_window))
        '''
        
        # The maximum gap tolerance is up to 30% of total points count.
        interp_limit = int(math.ceil(minDuration*3.0/10)) # Flexible interpolation limit based on window length.
        print('interp_limit = {}'.format(interp_limit))
        
        '''
        # Sliding window. If half_window=15, len(DataFrame)=60, range(15,45)=[15,...,44].
        for FluxRopeCenter in range(half_maxDuration, len(B_DataFrame) - half_maxDuration): # in minutes.
            indexFluxRopeStart = FluxRopeCenter - half_maxDuration
            indexFluxRopeEnd = FluxRopeCenter + half_maxDuration
        '''
        # Sliding window. If half_window=15, len(DataFrame)=60, range(15,45)=[15,...,44].
        for indexFluxRopeStart in xrange(len(B_DataFrame) - maxDuration): # in minutes.
            indexFluxRopeEnd = indexFluxRopeStart + maxDuration - 1  # The end point is included, so -1.
            # Grab the B slice within the window. Change the slice will change the original DataFrame.
            B_inWindow = B_DataFrame.iloc[indexFluxRopeStart : indexFluxRopeEnd + 1] # End is not included.
            
            # If there is any NaN in this range, try to interpolate.
            if B_inWindow.isnull().values.sum():
                B_inWindow_copy = B_inWindow.copy(deep=True)
                # For example, limit=3 means only interpolate the gap shorter than 4.
                B_inWindow_copy.interpolate(method='time', limit=interp_limit, inplace=True)
                if B_inWindow_copy.isnull().values.sum():
                    #print('Encounter NaN in B field data, skip this iteration.')
                    continue # If NaN still exists, skip this loop.
                else:
                    B_inWindow = B_inWindow_copy

            # Grab the Vsw slice within the window. Change the slice will change the original DataFrame.
            Vsw_inWindow = Vsw_DataFrame.iloc[indexFluxRopeStart : indexFluxRopeEnd + 1]
            # If there is any NaN in this range, try to interpolate.
            if Vsw_inWindow.isnull().values.sum():
                Vsw_inWindow_copy = Vsw_inWindow.copy(deep=True)
                # limit=3 means only interpolate the gap shorter than 4.
                Vsw_inWindow_copy.interpolate(method='time', limit=interp_limit, inplace=True)
                if Vsw_inWindow_copy.isnull().values.sum():
                    #print('Encounter NaN in Vsw data, skip this iteration.')
                    continue # If NaN still exists, skip this loop.
                else:
                    Vsw_inWindow = Vsw_inWindow_copy
                    
            # Grab the Np slice within the window. Change the slice will change the original DataFrame.
            # Np_inWindow = Np_DataFrame.iloc[indexFluxRopeStart : indexFluxRopeEnd+1]
            
            # Calculate VHT in GSE frame.
            #VHT_inGSE = findVHT(B_inWindow, Vsw_inWindow) # Very slow.
            # Calculating VHT takes very long time(0.02748s for 14 data points), we use mean Vsw as VHT.
            VHT_inGSE = np.array(Vsw_inWindow.mean())
            
            # Return value: timeRange, Residue, orientation
            dt = 60.0 # For 1 minute resolution data, the time increment is 60 seconds.
            result_temp = pool.apply_async(searchFluxRopeInWindow, args=(B_inWindow, VHT_inGSE, n_theta_grid, minDuration, dt, flag_smoothA))
            # print(result_temp.get()) # This statement will cause IO very slow.
            results.append(result_temp)
            # DO NOT unpack result here. It will block IO. Unpack in bulk.

        # Next we are going to save file We have to wait for all worker processes to finish.
        # Block main process to wait for worker processes to finish. This while loop will execute almost immediately when the innner for loop goes through. The inner for loop is non-blocked, so it finish in seconds.
        while len(pool._cache)!=0:
            #print('{} - Waiting... There are {} worker processes in pool.'.format(time.ctime(), len(pool._cache)))
            time.sleep(1)
        print('{} - len(pool._cache) = {}'.format(time.ctime(), len(pool._cache)))
        print('{} - Duration range {}~{} minutes is completed!'.format(time.ctime(), minDuration, maxDuration))

        # Save result. One file per window size.
        results_true_tuple_list = []
        results_false_tuple_list = []
        # Unpack results. Convert to tuple, and put into list.
        for one_result in results:
            results_tuple_temp = (one_result.get())
            #print(results_tuple_temp)
            if not np.isinf(results_tuple_temp[5]): # Check residue.
                #print(results_tuple_temp)
                if results_tuple_temp[4]: #if True, turn point on top.
                    results_true_tuple_list.append(results_tuple_temp)
                else: # Turn point on bottom.
                    results_false_tuple_list.append(results_tuple_temp)
        # Save results to dictionary. One key per window size.
        key_temp = str(minDuration) + '~' + str(maxDuration)
        search_result_raw_true[key_temp] = results_true_tuple_list
        search_result_raw_false[key_temp] = results_false_tuple_list

        # Empty container results[].
        results = []

        endTime = datetime.now()
        time_spent_in_seconds = (endTime - startTime).total_seconds()
        print('Time spent on this window: {} seconds ({} minutes).'.format(time_spent_in_seconds, round(time_spent_in_seconds/60.0, 2)))

    # Close pool, prevent new worker process from joining.
    pool.close()
    # Block caller process until workder processes terminate.
    pool.join()

    totalEndTime = datetime.now()
    time_spent_in_seconds = (totalEndTime - totalStartTime).total_seconds()

    print('\n{} - All duration ranges are completed!'.format(time.ctime()))
    print('\nSaving search result...')
    search_result_raw = {'true':search_result_raw_true, 'false':search_result_raw_false, 'timeRange':{'datetimeStart':datetimeStart, 'datetimeEnd':datetimeEnd}}
    search_result_raw_filename = search_result_dir + '/search_result_raw.p'
    pickle.dump(search_result_raw, open(search_result_raw_filename, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
    print('Done!')
    print('\nTotal CPU cores: {}.'.format(num_cpus))
    print('Max number of workder process in pool: {}.'.format(max_processes))
    print('Total Time spent: {} seconds ({} minutes).\n'.format(time_spent_in_seconds, round(time_spent_in_seconds/60.0, 2)))
    
    return search_result_raw

################################################################################################################

'''
# Remove the flux rope records with Vsw discontinuity, including shock.
def find_solar_wind_discontinuity(data_DF, **kwargs):
    # Default value.
    VswJumpThreshold = 50 #(km/s)
    smoothWindow = 1
    print('\nDefault parameters:')
    print('VswJumpThreshold = {} km/s'.format(VswJumpThreshold))
    print('smoothWindow = {}'.format(smoothWindow))
    # Check arguments.
    if 'VswJumpThreshold' in kwargs:
        VswJumpThreshold = kwargs['VswJumpThreshold']
        print('VswJumpThreshold is set to {} km/s.'.format(VswJumpThreshold))
    if 'smoothWindow' in kwargs:
        smoothWindow = kwargs['smoothWindow']
        print('smoothWindow is set to {}.'.format(smoothWindow))
        
    Vsw_vector = data_DF[['Vx', 'Vy', 'Vz']].copy()
    Vsw = np.sqrt(np.square(Vsw_vector).sum(axis=1))
    Vsw_DF = pd.DataFrame({'Vsw':Vsw, 'time':Vsw_vector.index})
    # Interpolate oneRecord_Vsw_temp if there is any NaNs in any column.
    Vsw_DF.interpolate(method='time', limit=None, inplace=True)
    # interpolate won't fill leading NaNs, so we use backward fill.
    Vsw_DF.bfill(inplace=True)
    Vsw_DF.ffill(inplace=True)
    
    discontinuity_DF = pd.DataFrame(columns=['time', 'speedJump'])
    # Calculate max Vsw jump.
    max_i_Vsw = len(Vsw_DF)-1 # max_i_Vsw is the max index.
    for i_temp in xrange(1, max_i_Vsw): # Do not include the last element.
        print('percent = {} %'.format(i_temp/max_i_Vsw*100.0))
        if (i_temp-(smoothWindow-1)>=0)&(i_temp+smoothWindow<=max_i_Vsw):
            speedJump_temp = abs(np.nanmean(Vsw_DF['Vsw'].iloc[i_temp-(smoothWindow-1):i_temp+1]) - np.nanmean(Vsw_DF['Vsw'].iloc[i_temp+1:i_temp+smoothWindow+1]))
            if (speedJump_temp>=VswJumpThreshold):
                discontinuity_DF = discontinuity_DF.append(pd.DataFrame({'time':[Vsw_DF.index[i_temp]], 'speedJump':[speedJump_temp]}))
                print(discontinuity_DF)
                plt.plot(np.array(Vsw_DF['Vsw'].iloc[i_temp-(smoothWindow-1):i_temp+smoothWindow+1]))
                plt.show()
        elif (i_temp-(smoothWindow-1)<0)&(i_temp+smoothWindow<=max_i_Vsw):
            speedJump_temp = abs(np.nanmean(Vsw_DF['Vsw'].iloc[:i_temp+1]) - np.nanmean(Vsw_DF['Vsw'].iloc[i_temp+1:i_temp+smoothWindow+1]))
            if (speedJump_temp>=VswJumpThreshold):
                discontinuity_DF = discontinuity_DF.append(pd.DataFrame({'time':[Vsw_DF.index[i_temp]], 'speedJump':[speedJump_temp]}))
                print(discontinuity_DF)
                plt.plot(np.array(Vsw_DF['Vsw'].iloc[:i_temp+smoothWindow+1]))
                plt.show()
        elif (i_temp-(smoothWindow-1)>=0)&(i_temp+smoothWindow>max_i_Vsw):
            speedJump_temp = abs(np.nanmean(Vsw_DF['Vsw'].iloc[i_temp-(smoothWindow-1):i_temp+1]) - np.nanmean(Vsw_DF['Vsw'].iloc[i_temp+1:]))
            if (speedJump_temp>=VswJumpThreshold):
                discontinuity_DF = discontinuity_DF.append(pd.DataFrame({'time':[Vsw_DF.index[i_temp]], 'speedJump':[speedJump_temp]}))
                print(discontinuity_DF)
                plt.plot(np.array(Vsw_DF['Vsw'].iloc[i_temp-(smoothWindow-1):]))
                plt.show()
        elif (i_temp-(smoothWindow-1)<0)&(i_temp+smoothWindow>max_i_Vsw):
            print('Smooth window is too large or the length of Vsw array is too short. Must 2*smoothWindow<=len(Array).')
    # Terminal output format.
    pd.set_option('display.max_rows', 1000)
    pd.set_option('display.max_columns', 100)
    pd.set_option('display.width', 300)
    print(discontinuity_DF.reset_index(drop=True))   
'''        
        
################################################################################################################  
        
# Calculate r_VHT (correlation coefficient of EHT and E).
def calculate_r_VHT(data_DF, fluxRopeList_DF):

    # Copy fluxRopeList_DF.
    fluxRopeList_with_r_VHT_DF = fluxRopeList_DF.copy()
    # Add 'r_VHT' column to fluxRopeList_DF.
    fluxRopeList_with_r_VHT_DF = fluxRopeList_with_r_VHT_DF.assign(r_VHT=[0.0]*len(fluxRopeList_with_r_VHT_DF))
    # Loop each record to calculate cv.
    recordLength = len(fluxRopeList_with_r_VHT_DF)
    for index_temp, oneRecord_temp in fluxRopeList_with_r_VHT_DF.iterrows():
        print('\nindex_temp = {}/{}'.format(index_temp, recordLength))
        startTime_temp = oneRecord_temp['startTime']
        endTime_temp = oneRecord_temp['endTime']
        VHT_temp = oneRecord_temp['VHT']
        # Grab the data for one FR_record.
        selectedRange_mask = (data_DF.index >= oneRecord_temp['startTime']) & (data_DF.index <= oneRecord_temp['endTime'])
        # The data of fluxrope candidate.
        oneRecord_data_temp = data_DF.iloc[selectedRange_mask]
        # Get solar wind slice copy.
        oneRecord_Vsw_temp = oneRecord_data_temp[['Vx', 'Vy', 'Vz']].copy()
        # Calculate Vsw magnitude.
        Vsw_temp = np.sqrt(np.square(oneRecord_Vsw_temp).sum(axis=1))
        # Get B data copy
        oneRecord_B_temp = oneRecord_data_temp[['Bx', 'By', 'Bz']].copy()
        # Calculate r_VHT.
        slope, intercept, r_VHT_temp = check_VHT(VHT_temp, oneRecord_Vsw_temp, oneRecord_B_temp)
        # Add r_VHT to fluxRopeList_with_r_VHT_DF.
        fluxRopeList_with_r_VHT_DF.loc[index_temp, 'r_VHT'] = r_VHT_temp
        
        if 0:
            if (r_VHT_temp<0.98):
                plt.plot(np.array(Vsw_temp))
                plt.ylim([300,700])
                plt.show()

    print(fluxRopeList_with_r_VHT_DF)
    fluxRopeList_with_r_VHT_DF['r_VHT'].plot.hist(bins=100)
    plt.show()

################################################################################################################

# Clean up raw_result.
def clean_up_raw_result(data_DF, dataObject_or_dataPath, **kwargs):
    # Check input datatype:
    # If dataObject_or_dataPath is an object(dict):
    if isinstance(dataObject_or_dataPath, dict):
        print('\nYour input is a dictionary data.')
        search_result_raw = dataObject_or_dataPath
    elif isinstance(dataObject_or_dataPath, str):
        print('\nYour input is a path. Load the dictionary data via this path.')
        search_result_raw = pd.read_pickle(open(dataObject_or_dataPath, 'rb'))
    else:
        print('\nPlease input the correct datatype!')
        return None

    #print(search_result_raw.keys())
    
    # Check keyword parameters.
    # Set default value.
    # Set turnTime tolerance dictionary.
    turnTime_tolerance = 5
    # Set minimum residue.
    min_residue_diff = 0.12
    min_residue_fit = 0.14
    # Set fitted curve quality parameters.
    max_tailPercentile = 0.3
    max_tailDiff = 0.3
    max_PtFitStd = 0.3
    # Remove discontinuity.
    isRemoveShock = False
    Vsw_std_threshold = 18 # Max allowed standard deviation for solar wind speed.
    Vsw_diff_threshold = 60 # Max allowed solar wind max-min difference.
    # walen test.
    walenTest_r_threshold = 0.5 # correlation coefficient.
    walenTest_k_threshold = 0.3 # slope.
    # Display control.
    isVerbose = False
    isPrintIntermediateDF = False
    # output filename.
    output_filename = 'search_result_no_overlap'
    # output dir.
    output_dir = os.getcwd()
    
    print('\nDefault parameters:')
    print('turnTime_tolerance    = {} minutes'.format(turnTime_tolerance))
    print('min_residue_diff      = {}'.format(min_residue_diff))
    print('min_residue_fit       = {}'.format(min_residue_fit))
    print('max_tailPercentile    = {}'.format(max_tailPercentile))
    print('max_tailDiff          = {}'.format(max_tailDiff))
    print('max_PtFitStd          = {}'.format(max_PtFitStd))
    print('isRemoveShock         = {}'.format(isRemoveShock))
    print('Vsw_std_threshold     = {} km/s'.format(Vsw_std_threshold))
    print('Vsw_diff_threshold    = {} km/s'.format(Vsw_diff_threshold))
    print('walenTest_r_threshold = {}'.format(walenTest_r_threshold))
    print('walenTest_k_threshold = {}'.format(walenTest_k_threshold))
    print('isVerbose             = {}'.format(isVerbose))
    print('isPrintIntermediateDF = {}'.format(isPrintIntermediateDF))
    print('output_dir            = {}.'.format(output_dir))
    print('output_filename       = {}.'.format(output_filename))
    #exit()
    
    # If keyword is specified, overwrite the default value.
    print('\nSetting parameters:')
    if 'turnTime_tolerance' in kwargs:
        turnTime_tolerance = kwargs['turnTime_tolerance']
        print('turnTime_tolerance is set to {}.'.format(turnTime_tolerance))
    if 'min_residue_diff' in kwargs:
        min_residue_diff = kwargs['min_residue_diff']
        print('min_residue_diff is set to {}.'.format(min_residue_diff))
    if 'min_residue_fit' in kwargs:
        min_residue_fit = kwargs['min_residue_fit']
        print('min_residue_fit is set to {}.'.format(min_residue_fit))
    if 'max_tailPercentile' in kwargs:
        max_tailPercentile = kwargs['max_tailPercentile']
        print('max_tailPercentile is set to {}.'.format(max_tailPercentile))
    if 'max_tailDiff' in kwargs:
        max_tailDiff = kwargs['max_tailDiff']
        print('max_tailDiff is set to {}.'.format(max_tailDiff))
    if 'max_PtFitStd' in kwargs:
        max_PtFitStd = kwargs['max_PtFitStd']
        print('max_PtFitStd is set to {}.'.format(max_PtFitStd))
    if 'isRemoveShock' in kwargs:
        isRemoveShock = kwargs['isRemoveShock']
        print('isRemoveShock set to {}.'.format(isRemoveShock))
        if isRemoveShock:
            if 'shockList_DF' in kwargs:
                shockList_DF = kwargs['shockList_DF']
                print('shockList_DF is loaded.')
            else:
                print('isRemoveShock is True, but shockList_DF is not provided.')
                return None
            if 'spacecraftID' in kwargs:
                spacecraftID = kwargs['spacecraftID']
                print('spacecraftID is set to {}'.format(spacecraftID))
            else:
                print('isRemoveShock is True, but spacecraftID is not provided.')
                return None
    if 'Vsw_std_threshold' in kwargs:
        Vsw_std_threshold = kwargs['Vsw_std_threshold']
        print('Vsw_std_threshold is set to {}.'.format(Vsw_std_threshold))
    if 'Vsw_diff_threshold' in kwargs:
        Vsw_diff_threshold = kwargs['Vsw_diff_threshold']
        print('Vsw_diff_threshold is set to {}.'.format(Vsw_diff_threshold))
    if 'walenTest_r_threshold' in kwargs:
        walenTest_r_threshold = kwargs['walenTest_r_threshold']
        print('walenTest_r_threshold is set to {}.'.format(walenTest_r_threshold))
    if 'walenTest_k_threshold' in kwargs:
        walenTest_k_threshold = kwargs['walenTest_k_threshold']
        print('walenTest_k_threshold is set to {}.'.format(walenTest_k_threshold))
    if 'isVerbose' in kwargs:
        isVerbose = kwargs['isVerbose']
        print('isVerbose is set to {}.'.format(isVerbose))
    if 'isPrintIntermediateDF' in kwargs:
        isPrintIntermediateDF = kwargs['isPrintIntermediateDF']
        print('isPrintIntermediateDF is set to {}.'.format(isPrintIntermediateDF))
    if 'output_dir' in kwargs:
        output_dir = kwargs['output_dir']
        print('output_dir is set to {}.'.format(output_dir))
    if 'output_filename' in kwargs:
        output_filename = kwargs['output_filename']
        print('output_filename is set to {}.'.format(output_filename))
    if 'spacecraftID' in kwargs:
        spacecraftID = kwargs['spacecraftID']
        print('spacecraftID is {}.'.format(spacecraftID))
    if 'shockList' in kwargs:
        shockList = kwargs['shockList']
        print('shockList is loaded.')
    
    # Set terminal display format.
    if isPrintIntermediateDF:
        pd.set_option('display.max_rows', 1000)
        pd.set_option('display.max_columns', 500)
        pd.set_option('display.width', 500)

    # Physics constants.
    mu0 = 4.0 * np.pi * 1e-7 #(N/A^2) magnetic constant permeability of free space vacuum permeability.
    m_proton = 1.6726219e-27 # Proton mass. In kg.
    factor_deg2rad = np.pi/180.0 # Convert degree to rad.
    k_Boltzmann = 1.3806488e-23 # Boltzmann constant, in J/K.
    # Parameters.
    dt = 60.0 # For 1 minute resolution data, the time increment is 60 seconds.

    # Get duration list.
    duration_list = search_result_raw['true'].keys()
    window_size_list = []
    for item in duration_list:
        window_size = int(item.split('~')[1])
        window_size_list.append(window_size)
    # Sort the duration_list with window_size_list by argsort, the duration is in descending order.
    sorted_index_window_size_array = np.argsort(window_size_list)
    sorted_index_window_size_array = sorted_index_window_size_array[::-1]
    duration_array = np.array(duration_list)
    duration_list = list(duration_array[sorted_index_window_size_array])
    print('\nduration_list:')
    print(duration_list)
    
    # Debug.
    #duration_list = ['160~180','140~160','120~140','100~120','80~100','60~80','50~60','40~50','30~40','20~30','10~20'] #
    #duration_list = ['140~160']
        
    # Get search_iteration.
    search_iteration = len(duration_list)

    # Get start and end time.
    datetimeStart = search_result_raw['timeRange']['datetimeStart']
    datetimeEnd = search_result_raw['timeRange']['datetimeEnd']

    # Create empty eventList_no_overlap DataFrame, the cleaned lists will be append to it.
    eventList_DF_noOverlap = pd.DataFrame(columns=['startTime', 'turnTime', 'endTime', 'duration', 'residue_diff', 'residue_fit', 'theta_phi', 'VHT', 'reduced_residue'])

    for i_iteration in xrange(search_iteration):
        print('\n======================================================================')
        print('\niteration = {}/{}'.format(i_iteration+1, search_iteration))
        
        # Get slots.
        # Create empty slotList DataFrame.
        slotList_DF_temp = pd.DataFrame(columns=['slotStart', 'slotEnd'])
        # If eventList_DF_noOverlap is empty, the entire time range is slot.
        if eventList_DF_noOverlap.empty:
            # Create slot.
            oneSlot_temp = pd.DataFrame(OrderedDict((('slotStart', [datetimeStart]),('slotEnd', [datetimeEnd]))))
            # Append this slot to slotList_DF_temp.
            slotList_DF_temp = slotList_DF_temp.append(oneSlot_temp, ignore_index=True)
        else: # If eventList_DF_noOverlap is not empty, extract slot from eventList_DF_noOverlap.
            # Add first slot: [datetimeStart:eventList_DF_noOverlap.iloc[0]['startTime']].
            if datetimeStart<eventList_DF_noOverlap.iloc[0]['startTime']:
                # An OrderedDict is a dictionary subclass that remembers the order in which its contents are added.
                oneSlot_temp = pd.DataFrame(OrderedDict((('slotStart', [datetimeStart]),('slotEnd', [eventList_DF_noOverlap.iloc[0]['startTime']]))))
                # Append first slot to slotList_DF_temp.
                slotList_DF_temp = slotList_DF_temp.append(oneSlot_temp, ignore_index=True)
            # Add last slot: [eventList_DF_noOverlap.iloc[-1]['endTime'] : datetimeEnd]
            if datetimeEnd>eventList_DF_noOverlap.iloc[-1]['endTime']:
                oneSlot_temp = pd.DataFrame(OrderedDict((('slotStart', [eventList_DF_noOverlap.iloc[-1]['endTime']]),('slotEnd', [datetimeEnd]))))
                # Append last slot to slotList_DF_temp.
                slotList_DF_temp = slotList_DF_temp.append(oneSlot_temp, ignore_index=True)
            # If eventList_DF_noOverlap has more than one record, add other slots besides first one and last one.
            if len(eventList_DF_noOverlap)>1:
                # Get slots from eventList_DF_noOverlap.
                multiSlot_temp = pd.DataFrame(OrderedDict((('slotStart', list(eventList_DF_noOverlap.iloc[:-1]['endTime'])), ('slotEnd', list(eventList_DF_noOverlap.iloc[1:]['startTime'])))))
                # Append these slots to slotList_DF_temp.
                slotList_DF_temp = slotList_DF_temp.append(multiSlot_temp, ignore_index=True)
            # Sort slotList_DF_temp by either slotStart or slotEnd. Because there is no overlap, both method are equivalent.
            slotList_DF_temp = slotList_DF_temp.sort_values(by='slotStart')
        # Reset index.
        slotList_DF_temp.reset_index(drop=True, inplace=True)
        
        if isPrintIntermediateDF:
            print('\nslotList_DF_temp:')
            print(slotList_DF_temp)
        
        duration_str_temp = duration_list[i_iteration]
        #print('duration_str_temp = {}'.format(duration_str_temp))
        print('Combining events with {} minutes duration range:'.format(duration_str_temp))
        # Load first duration range in duration_list. This list contains the events with longest duration.
        eventList_temp = search_result_raw['true'][duration_str_temp]
        # 1) Check point 1.
        # If event list not empty, put it into DataFrame.
        if not eventList_temp: # An empty list is itself considered false in true value testing.
            # Event list is empty. Skip the rest operations.
            print('\nEvent list eventList_temp is empty!')
            print('Go the the next iteration!')
            continue

        # Create headers.
        eventList_temp_Header = ['startTime', 'turnTime', 'endTime', 'duration', 'topTurn', 'residue_diff', 'residue_fit', 'theta_phi', 'VHT']
        # Convert 2-D list to DataFrame.
        eventList_DF_0_original_temp = pd.DataFrame(eventList_temp, columns=eventList_temp_Header)
        # Parse string to datetime.
        eventList_DF_0_original_temp['startTime'] = pd.to_datetime(eventList_DF_0_original_temp['startTime'], format="%Y%m%d%H%M")
        eventList_DF_0_original_temp['turnTime'] = pd.to_datetime(eventList_DF_0_original_temp['turnTime'], format="%Y%m%d%H%M")
        eventList_DF_0_original_temp['endTime'] = pd.to_datetime(eventList_DF_0_original_temp['endTime'], format="%Y%m%d%H%M")

        # Find all records from eventList_DF_0_original_temp that fit the slots of slotList_DF_temp.
        print('\nFitting the events into available slots...')
        if isVerbose:
            print('Before fitting, totoal records is {}'.format(len(eventList_DF_0_original_temp)))
        # Make a copy of eventList_DF_0_original_temp.
        eventList_DF_1_fitSlot_temp = eventList_DF_0_original_temp.copy()
        # Add keepFlag column to eventList_DF_1_fitSlot_temp.
        eventList_DF_1_fitSlot_temp = eventList_DF_1_fitSlot_temp.assign(keepFlag=[False]*len(eventList_DF_1_fitSlot_temp))
        for index, oneSlot_temp in slotList_DF_temp.iterrows():
            #print('index = {}'.format(index))
            #print(slot_record)
            keepMask = (eventList_DF_1_fitSlot_temp['startTime']>=oneSlot_temp['slotStart'])&(eventList_DF_1_fitSlot_temp['endTime']<=oneSlot_temp['slotEnd'])
            if(keepMask.sum()>0):
                # Set true flag.
                eventList_DF_1_fitSlot_temp.loc[eventList_DF_1_fitSlot_temp[keepMask].index, 'keepFlag'] = True
        # Keep the records with true flag.
        eventList_DF_1_fitSlot_temp = eventList_DF_1_fitSlot_temp[eventList_DF_1_fitSlot_temp['keepFlag']==True]
        # Reset index.
        eventList_DF_1_fitSlot_temp.reset_index(drop=True, inplace=True)
        if isVerbose:
            print('After fitting, totoal records is {}'.format(len(eventList_DF_1_fitSlot_temp)))
        print('Done.')
        
        # 2) Check point 2.
        # After fitting slots, check if eventList_DF_1_fitSlot_temp is empty.
        if eventList_DF_1_fitSlot_temp.empty:
            # DataFrame eventList_DF_1_fitSlot_temp is empty. Skip the rest operations.
            print('\nDataFrame eventList_DF_1_fitSlot_temp is empty!')
            print('Go the the next iteration!')
            continue

        # Remove the event with residue_diff > min_residue_diff and residue_fit > min_residue_fit.
        print('\nRemoving events with residue_diff > {} and residue_fit > {}...'.format(min_residue_diff, min_residue_fit))
        if isVerbose:
            print('Before Removing, total records is {}.'.format(len(eventList_DF_1_fitSlot_temp)))
        eventList_DF_2_fineResidue_temp = eventList_DF_1_fitSlot_temp[(eventList_DF_1_fitSlot_temp['residue_diff']<=min_residue_diff)&(eventList_DF_1_fitSlot_temp['residue_fit']<=min_residue_fit)]
        # Reset index
        eventList_DF_2_fineResidue_temp.reset_index(drop=True, inplace=True)
        if isVerbose:
            print('After Removing, total records is {}.'.format(len(eventList_DF_2_fineResidue_temp)))
        print('Done.')

        # 3) Check point 3.
        # After removing bad residue, check if eventList_DF_2_fineResidue_temp is empty.
        if eventList_DF_2_fineResidue_temp.empty:
            # DataFrame eventList_DF_2_fineResidue_temp is empty. Skip the rest operations.
            print('\nDataFrame eventList_DF_2_fineResidue_temp is empty!')
            print('Go the the next iteration!')
            continue
        
        # Remove records contains shock.
        if isRemoveShock:
            print('\nRemoving events containing shock...')
            eventList_DF_2_fineResidue_temp = eventList_DF_2_fineResidue_temp.copy()
            len_eventList_DF_2_before = len(eventList_DF_2_fineResidue_temp)
            # Get trimmed shockList.
            shockList_trimmed_DF = shockList_DF[(shockList_DF.index>=datetimeStart)&(shockList_DF.index<=datetimeEnd)]
            spacecraftID_dict = {'WIND':'Wind', 'ACE':'ACE'}
            shockList_trimmed_specifiedSpacecraft_DF = shockList_trimmed_DF[(shockList_trimmed_DF['Spacecraft'].str.contains(spacecraftID_dict[spacecraftID]))]
            # Reset index. The original index is shock time. Put shock time into a new column.
            shockList_trimmed_specifiedSpacecraft_DF = shockList_trimmed_specifiedSpacecraft_DF.reset_index().rename(columns={'index':'shockTime'}).copy()
            len_shockList = len(shockList_trimmed_specifiedSpacecraft_DF)
            for index_temp, shock_record_temp in shockList_trimmed_specifiedSpacecraft_DF.iterrows():
                if isVerbose:
                    print('Checking if shock in flux rope: checking duration {} minutes, {}/{}...'.format(duration_str_temp, index_temp+1, len_shockList))
                shockTime_temp = shock_record_temp['shockTime']
                mask_containShock = (eventList_DF_2_fineResidue_temp['startTime']<=shockTime_temp)&(eventList_DF_2_fineResidue_temp['endTime']>=shockTime_temp)
                if mask_containShock.sum()>0:
                    #print(shockTime_temp)
                    #print(eventList_DF_2_fineResidue_temp[mask_containShock])
                    eventList_DF_2_fineResidue_temp = eventList_DF_2_fineResidue_temp[~mask_containShock]
            len_eventList_DF_2_after = len(eventList_DF_2_fineResidue_temp)
            eventList_DF_2_fineResidue_temp = eventList_DF_2_fineResidue_temp.reset_index(drop=True)
            if isVerbose:
                print('Before removing shock, total records is {}.'.format(len_eventList_DF_2_before))
                print('Before removing shock, total records is {}.'.format(len_eventList_DF_2_after))
            print('Done.')
                
            # 4) Check point 4.
            # After removing discontinuity, check if eventList_DF_2_fineResidue_temp is empty.
            if eventList_DF_2_fineResidue_temp.empty:
                # eventList_DF_2_fineResidue_temp is empty. Skip the rest operations in this loop.
                print('\nAfter removing shock, DataFrame eventList_DF_2_fineResidue_temp is empty!')
                print('Go the the next iteration!')
                continue

        # Clean up the records with same turnTime.
        print('\nCombining events with same turnTime...')
        if isVerbose:
            print('Before combining, total records is {}.'.format(len(eventList_DF_2_fineResidue_temp)))
        # Sort by turnTime.
        eventList_DF_2_fineResidue_temp = eventList_DF_2_fineResidue_temp.sort_values(by='turnTime')
        # Group by turnTime.
        index_min_Residue_diff_inGrouped = eventList_DF_2_fineResidue_temp.groupby(['turnTime'], sort=False)['residue_diff'].transform(min) == eventList_DF_2_fineResidue_temp['residue_diff']
        # Pick the event with min residue_diff among the events sharing same turnPoint.
        eventList_DF_3_combinedByTurnTime_temp = eventList_DF_2_fineResidue_temp[index_min_Residue_diff_inGrouped]
        # Reset index
        eventList_DF_3_combinedByTurnTime_temp.reset_index(drop=True, inplace=True)
        if isVerbose:
            print('After combining, total records is {}.'.format(len(eventList_DF_3_combinedByTurnTime_temp)))
        print('Done.')

        # No need to check whether eventList_DF_3_combinedByTurnTime_temp is empty.
        # If eventList_DF_2_fineResidue_temp is not empty, eventList_DF_3_combinedByTurnTime_temp cannot be empty.
        
        
        print('\nRemoving events failed in walen test, |walenTest_k| > {}...'.format(walenTest_k_threshold))
        # Add an walen test result column.
        # .assign() always returns a copy of the data, leaving the original DataFrame untouched.
        eventList_DF_4_passWalenTest_temp = eventList_DF_3_combinedByTurnTime_temp.copy()
        eventList_DF_4_passWalenTest_temp = eventList_DF_4_passWalenTest_temp.assign(r = len(eventList_DF_4_passWalenTest_temp)*[np.nan])
        eventList_DF_4_passWalenTest_temp = eventList_DF_4_passWalenTest_temp.assign(k = len(eventList_DF_4_passWalenTest_temp)*[np.nan])
        # Cacluate walen test r value.
        eventList_DF_4_passWalenTest_temp.reset_index(drop=True, inplace=True)
        len_eventList_DF_4_before = len(eventList_DF_4_passWalenTest_temp)
        for index, FR_record in eventList_DF_4_passWalenTest_temp.iterrows():
            if isVerbose:
                print('Walen test: checking duration {} minutes, {}/{}...'.format(duration_str_temp, index+1, len_eventList_DF_4_before))
            theta_deg, phi_deg = FR_record['theta_phi']
            VHT_inGSE = np.array(FR_record['VHT'])
            # Grab the data for one FR_record.
            selectedRange_mask = (data_DF.index >= FR_record['startTime']) & (data_DF.index <= FR_record['endTime'])
            # The data of fluxrope candidate.
            FR_record_data = data_DF.iloc[selectedRange_mask]
            # Interpolate FR_record_data if there is any NaNs in any column.
            FR_record_data_interpolated = FR_record_data.copy(deep=True)
            FR_record_data_interpolated.interpolate(method='time', limit=None, inplace=True)
            # interpolate won't fill leading NaNs, so we use backward fill.
            FR_record_data_interpolated.bfill(inplace=True)
            FR_record_data_interpolated.ffill(inplace=True)

            # Apply walen test on the result(in Flux Rope frame).
            B_inGSE = FR_record_data_interpolated[['Bx', 'By', 'Bz']]
            Vsw_inGSE = FR_record_data_interpolated[['Vx', 'Vy', 'Vz']]
            # Project B_inGSE, VHT_inGSE, and Vsw_inFR into Flux Rope Frame.
            matrix_transToFluxRopeFrame = angle2matrix(theta_deg, phi_deg, np.array(VHT_inGSE))
            B_inFR = B_inGSE.dot(matrix_transToFluxRopeFrame)
            VHT_inFR = VHT_inGSE.dot(matrix_transToFluxRopeFrame)
            Vsw_inFR = Vsw_inGSE.dot(matrix_transToFluxRopeFrame)
            # Proton mass density. Original Np is in #/cc ( cc = cubic centimeter). Multiply by 1e6 to convert cc to m^3.
            P_massDensity = FR_record_data['Np'] * m_proton * 1e6 # In kg/m^3.
            len_P_massDensity = len(P_massDensity)
            P_massDensity_array = np.array(P_massDensity)
            P_massDensity_array = np.reshape(P_massDensity_array, (len_P_massDensity, 1))
            # Alfven speed. Multiply by 1e-9 to convert nT to T. Divided by 1000.0 to convert m/s to km/s.
            VA = np.array(B_inFR * 1e-9) / np.sqrt(mu0 * P_massDensity_array) / 1000.0
            VA_1D = np.reshape(VA, VA.size)
            V_remaining = np.array(Vsw_inFR - VHT_inFR)
            V_remaining_1D = np.reshape(V_remaining, V_remaining.size)
            # Call walen test function.
            # First row is x component, second row is y component, third row is z component.
            walenTest_slope, walenTest_intercept, walenTest_r_value = walenTest(VA_1D, V_remaining_1D)
            eventList_DF_4_passWalenTest_temp.loc[index, 'r'] = round(walenTest_r_value, 4) # r.
            eventList_DF_4_passWalenTest_temp.loc[index, 'k'] = round(walenTest_slope, 4) # k.
            #print('r={}'.format(walenTest_r_value))

        # Remove the records with |k|>walenTest_k_threshold.
        #eventList_DF_4_passWalenTest_temp = eventList_DF_4_passWalenTest_temp[(abs(eventList_DF_4_passWalenTest_temp['r'])<=walenTest_r_threshold)|(abs(eventList_DF_4_passWalenTest_temp['k'])<walenTest_k_threshold)]
        eventList_DF_4_passWalenTest_temp = eventList_DF_4_passWalenTest_temp[(abs(eventList_DF_4_passWalenTest_temp['k'])<=walenTest_k_threshold)]
        # Drop 'r'
        eventList_DF_4_passWalenTest_temp = eventList_DF_4_passWalenTest_temp.drop('r', axis=1)
        # Drop 'k'
        eventList_DF_4_passWalenTest_temp = eventList_DF_4_passWalenTest_temp.drop('k', axis=1)
        # Drop 'topTurn' column.
        eventList_DF_4_passWalenTest_temp = eventList_DF_4_passWalenTest_temp.drop('topTurn', axis=1)
        # Reset index
        eventList_DF_4_passWalenTest_temp.reset_index(drop=True, inplace=True)
        # Get length after.
        len_eventList_DF_4_after = len(eventList_DF_4_passWalenTest_temp)
        if isVerbose:
            print('Before Walen test, total records is {}.'.format(len_eventList_DF_4_before))
            print('After Walen test, total records is {}.'.format(len_eventList_DF_4_after))
        print('Done.')
        
        # 4) Check point 5.
        # After walen test, check if eventList_DF_4_passWalenTest_temp is empty.
        if eventList_DF_4_passWalenTest_temp.empty:
            # eventList_DF_4_passWalenTest_temp is empty. Skip the rest operations in this loop.
            print('\nDataFrame eventList_DF_4_passWalenTest_temp is empty!')
            print('Go the the next iteration!')
            continue
        
        # Remove records with large Vsw fluctuations.
        print('\nRemoving events with large Vsw fluctuations, Vsw_std > {} or Vsw_diff > {}...'.format(Vsw_std_threshold, Vsw_diff_threshold))
        eventList_DF_5_noDiscontinuity_temp = eventList_DF_4_passWalenTest_temp.copy()
        # Add 'Vsw_std' and 'Vsw_diff' column to eventList_DF_5_noDiscontinuity_temp.
        eventList_DF_5_noDiscontinuity_temp = eventList_DF_5_noDiscontinuity_temp.assign(Vsw_std=[-1.0]*len(eventList_DF_5_noDiscontinuity_temp))
        eventList_DF_5_noDiscontinuity_temp = eventList_DF_5_noDiscontinuity_temp.assign(Vsw_diff=[-1.0]*len(eventList_DF_5_noDiscontinuity_temp))
        
        # Loop each record to calculate Vsw_std and Vsw_diff and boolean value of 'containShock'.
        len_eventList_DF_5_before = len(eventList_DF_5_noDiscontinuity_temp)
        for index_temp, oneRecord_temp in eventList_DF_5_noDiscontinuity_temp.iterrows():
            if isVerbose:
                print('Remove discontinuity: checking duration {} minutes, {}/{}...'.format(duration_str_temp, index_temp+1, len_eventList_DF_5_before))
            startTime_temp = oneRecord_temp['startTime']
            endTime_temp = oneRecord_temp['endTime']
            # Grab the data for one FR_record.
            selectedRange_mask = (data_DF.index >= oneRecord_temp['startTime']) & (data_DF.index <= oneRecord_temp['endTime'])
            # The data of fluxrope candidate.
            oneRecord_data_temp = data_DF.iloc[selectedRange_mask]
            # Get solar wind slice copy.
            oneRecord_Vsw_temp = oneRecord_data_temp[['Vx', 'Vy', 'Vz']].copy()
            # Calculate Vsw magnitude.
            Vsw_temp = np.sqrt(np.square(oneRecord_Vsw_temp).sum(axis=1))
            # Get Vsw_diff
            Vsw_diff_temp = np.nanmax(np.array(Vsw_temp)) - np.nanmin(np.array(Vsw_temp))
            # Add std to fluxRopeList_with_std_diff_DF.
            eventList_DF_5_noDiscontinuity_temp.loc[index_temp, 'Vsw_std'] = Vsw_temp.std(skipna=True)
            eventList_DF_5_noDiscontinuity_temp.loc[index_temp, 'Vsw_diff'] = Vsw_diff_temp

        mask_toBeRemoved = (eventList_DF_5_noDiscontinuity_temp['Vsw_std']>Vsw_std_threshold)|(eventList_DF_5_noDiscontinuity_temp['Vsw_diff']>Vsw_diff_threshold)
        eventList_DF_5_noDiscontinuity_temp = eventList_DF_5_noDiscontinuity_temp[~mask_toBeRemoved]
        len_eventList_DF_5_after = len(eventList_DF_5_noDiscontinuity_temp)
        # Drop 'Vsw_std'
        eventList_DF_5_noDiscontinuity_temp = eventList_DF_5_noDiscontinuity_temp.drop('Vsw_std', axis=1)
        # Drop 'Vsw_diff'
        eventList_DF_5_noDiscontinuity_temp = eventList_DF_5_noDiscontinuity_temp.drop('Vsw_diff', axis=1)
        # Reset index.
        eventList_DF_5_noDiscontinuity_temp.reset_index(drop=True, inplace=True)
        if isVerbose:
            print('Before removing large Vsw fluctuation, total records is {}.'.format(len_eventList_DF_5_before))
            print('After removing large Vsw fluctuation, total records is {}.'.format(len_eventList_DF_5_after))
        print('Done.')
        
        # 4) Check point 6.
        # After removing discontinuity, check if eventList_DF_5_noDiscontinuity_temp is empty.
        if eventList_DF_5_noDiscontinuity_temp.empty:
            # eventList_DF_5_noDiscontinuity_temp is empty. Skip the rest operations in this loop.
            print('\nDataFrame eventList_DF_5_noDiscontinuity_temp is empty!')
            print('Go the the next iteration!')
            continue
        
        # Until now, we still may have duplicated events with same residue_diff but different residue_fit. Keep this in mind when perform furture operations.
        
        # Clean events with less than turnTime_tolerance minutes turnTime difference.
        # Make a deep copy.
        eventList_DF_6_cleanedTurnTime_temp = eventList_DF_5_noDiscontinuity_temp.copy() # Default is deep copy.
        # Add difference of turn time column. Combine close turnTime.
        eventList_DF_6_cleanedTurnTime_temp = eventList_DF_6_cleanedTurnTime_temp.assign(diff=eventList_DF_6_cleanedTurnTime_temp['turnTime'].diff())
        eventList_DF_6_cleanedTurnTime_temp = eventList_DF_6_cleanedTurnTime_temp.assign(keepFlag=[True]*len(eventList_DF_6_cleanedTurnTime_temp))
        # Retrieving column index from column name.
        index_column_keepFlag = eventList_DF_6_cleanedTurnTime_temp.columns.get_loc('keepFlag')
        print('\nCombinning events with less than {} minutes turnTime difference...'.format(turnTime_tolerance))
        if isVerbose:
            print('Before combinning, total records is {}.'.format(len(eventList_DF_6_cleanedTurnTime_temp)))
        i_index = 1
        while(i_index < len(eventList_DF_6_cleanedTurnTime_temp)):
            if(eventList_DF_6_cleanedTurnTime_temp['diff'].iloc[i_index] <= timedelta(minutes=turnTime_tolerance)):
                cluster_begin_temp = i_index - 1
                while((eventList_DF_6_cleanedTurnTime_temp['diff'].iloc[i_index] <= timedelta(minutes=turnTime_tolerance)) ):
                    i_index += 1
                    if (i_index > len(eventList_DF_6_cleanedTurnTime_temp)-1):
                        break
                cluster_end_temp = i_index - 1
                #print('\nCluster index range = ({}~{})'.format(cluster_begin_temp, cluster_end_temp))
                # Get minimum residue_diff. .iloc[a:b]=[a,b), .loc[a:b]=[a,b]
                min_residue_diff_index_temp = eventList_DF_6_cleanedTurnTime_temp['residue_diff'].iloc[cluster_begin_temp:cluster_end_temp+1].idxmin()
                # Set record with min_residue as true, others as false.
                eventList_DF_6_cleanedTurnTime_temp.iloc[cluster_begin_temp:cluster_end_temp+1, index_column_keepFlag] = False
                eventList_DF_6_cleanedTurnTime_temp.loc[min_residue_diff_index_temp, 'keepFlag'] = True
            else:
                # For .iloc, can only use integer to specify row and column. No column string allowed.
                eventList_DF_6_cleanedTurnTime_temp.iloc[i_index, index_column_keepFlag] = True
                i_index += 1
        # Remove the records labeled as false.
        eventList_DF_6_cleanedTurnTime_temp = eventList_DF_6_cleanedTurnTime_temp[eventList_DF_6_cleanedTurnTime_temp['keepFlag']==True]
        eventList_DF_6_cleanedTurnTime_temp.reset_index(drop=True, inplace=True)
        # Drop diff.
        eventList_DF_6_cleanedTurnTime_temp = eventList_DF_6_cleanedTurnTime_temp.drop('diff', axis=1)
        # Reset index
        eventList_DF_6_cleanedTurnTime_temp.reset_index(drop=True, inplace=True)
        if isVerbose:
            print('After combinning, total records is {}.'.format(len(eventList_DF_6_cleanedTurnTime_temp)))
        print('Done.')
        
        # If eventList_DF_5_passWalenTest_temp is not empty, eventList_DF_6_cleanedTurnTime_temp can not be empty.
        # So we don't need to check if eventList_DF_6_cleanedTurnTime_temp is empty.

        print('\nCleaning events with bad fitting curve...')
        if isVerbose:
            print('Before cleaning, total records is {}.'.format(len(eventList_DF_6_cleanedTurnTime_temp)))
        # Clean the records with bad fitting curve shape.
        eventList_DF_7_fineFittingCurve_temp = eventList_DF_6_cleanedTurnTime_temp.copy()
        # Calculate fitting coefficients, remove the events with high tail, high tail diff, high std.
        # Add a column 'lowTail'.
        eventList_DF_7_fineFittingCurve_temp['lowTail'] = [None]*len(eventList_DF_7_fineFittingCurve_temp)
        #eventList_DF_7_fineFittingCurve_temp['lowTail_value'] = [-999999]*len(eventList_DF_7_fineFittingCurve_temp)
        eventList_DF_7_fineFittingCurve_temp['lowTailDiff'] = [None]*len(eventList_DF_7_fineFittingCurve_temp)
        #eventList_DF_7_fineFittingCurve_temp['lowTailDiff_value'] = [-999999]*len(eventList_DF_7_fineFittingCurve_temp)
        eventList_DF_7_fineFittingCurve_temp['lowStd'] = [None]*len(eventList_DF_7_fineFittingCurve_temp)
        #eventList_DF_7_fineFittingCurve_temp['lowStd_value'] = [-999999]*len(eventList_DF_7_fineFittingCurve_temp)
        for index, record in eventList_DF_7_fineFittingCurve_temp.iterrows():
            #print('index = {}'.format(index))
            fluxRopeStartTime = record['startTime']
            fluxRopeTurnTime   = record['turnTime']
            fluxRopeEndTime   = record['endTime']
            theta_deg, phi_deg = record['theta_phi']
            Residue_diff = record['residue_diff']
            Residue_fit = record['residue_fit']
            VHT_inGSE = np.array(record['VHT'])
            # Grab data in specific range.
            selectedRange_mask = (data_DF.index >= fluxRopeStartTime) & (data_DF.index <= fluxRopeEndTime)
            # The data of fluxrope candidate.
            record_data_temp = data_DF.iloc[selectedRange_mask]

            # Keys: Index([u'Bx', u'By', u'Bz', u'Vx', u'Vy', u'Vz', u'Np', u'Tp', u'Te'], dtype='object')
            # Get Magnetic field slice.
            B_inGSE = record_data_temp.ix[:,['Bx', 'By', 'Bz']].copy(deep=True)
            # Get the solar wind slice.
            Vsw_inGSE = record_data_temp.ix[:,['Vx', 'Vy', 'Vz']].copy(deep=True)
            # Get the proton number density slice.
            Np_inGSE = record_data_temp.ix[:,['Np']].copy(deep=True)
            # Get the proton temperature slice. In Kelvin.
            Tp_inGSE = record_data_temp.ix[:,['Tp']].copy(deep=True)
            if 'Te' in record_data_temp.keys():
                # Get the electron temperature slice. In Kelvin.
                Te_inGSE = record_data_temp.ix[:,['Te']].copy(deep=True)
            
            # If there is any NaN in B_inGSE, try to interpolate.
            if B_inGSE.isnull().values.sum():
                if isVerbose:
                    print('Found NaNs, interpolate B.')
                B_inGSE_copy = B_inGSE.copy(deep=True)
                # limit=3 means only interpolate the gap shorter than 4.
                B_inGSE_copy.interpolate(method='time', limit=None, inplace=True)
                # interpolate won't fill leading NaNs, so we use backward fill.
                B_inGSE_copy.bfill(inplace=True)
                B_inGSE_copy.ffill(inplace=True)
                if B_inGSE_copy.isnull().values.sum():
                    print('Too many NaNs in B. Skip this record. If this situation happens, please check.')
                    continue
                else:
                    B_inGSE = B_inGSE_copy

            # If there is any NaN in Vsw_inGSE, try to interpolate.
            if Vsw_inGSE.isnull().values.sum():
                if isVerbose:
                    print('Found NaNs, interpolate Vsw.')
                Vsw_inGSE_copy = Vsw_inGSE.copy(deep=True)
                # limit=3 means only interpolate the gap shorter than 4.
                Vsw_inGSE_copy.interpolate(method='time', limit=None, inplace=True)
                # interpolate won't fill leading NaNs, so we use backward fill.
                Vsw_inGSE_copy.bfill(inplace=True)
                Vsw_inGSE_copy.ffill(inplace=True)
                if Vsw_inGSE_copy.isnull().values.sum():
                    print('Too many NaNs in Vsw. Skip this record. If this situation happens, please check.')
                    continue
                else:
                    Vsw_inGSE = Vsw_inGSE_copy
                    
            # If there is any NaN in Np_inGSE, try to interpolate.
            if Np_inGSE.isnull().values.sum():
                if isVerbose:
                    print('Found NaNs, interpolate Np.')
                Np_inGSE_copy = Np_inGSE.copy(deep=True)
                # limit=3 means only interpolate the gap shorter than 4.
                Np_inGSE_copy.interpolate(method='time', limit=None, inplace=True)
                # interpolate won't fill leading NaNs, so we use backward fill.
                Np_inGSE_copy.bfill(inplace=True)
                Np_inGSE_copy.ffill(inplace=True)
                if Np_inGSE_copy.isnull().values.sum():
                    print('Too many NaNs in Vsw. Skip this record. If this situation happens, please check.')
                    continue
                else:
                    Np_inGSE = Np_inGSE_copy

            # Direction cosines:
            # x = rcos(alpha) = rsin(theta)cos(phi) => cos(alpha) = sin(theta)cos(phi)
            # y = rcos(beta)  = rsin(theta)sin(phi) => cos(beta)  = sin(theta)sin(phi)
            # z = rcos(gamma) = rcos(theta)         => cos(gamma) = cos(theta)
            # Use direction cosines to construct a unit vector.
            theta_rad = factor_deg2rad * theta_deg
            phi_rad   = factor_deg2rad * phi_deg

            # Form new Z_unitVector according to direction cosines.
            Z_unitVector = np.array([np.sin(theta_rad)*np.cos(phi_rad), np.sin(theta_rad)*np.sin(phi_rad), np.cos(theta_rad)])
            # Find X axis from Z axis and -VHT.
            X_unitVector = findXaxis(Z_unitVector, -VHT_inGSE)
            # Find the Y axis to form a right-handed coordinater with X and Z.
            Y_unitVector = formRighHandFrame(X_unitVector, Z_unitVector)

            # Project B_inGSE into FluxRope Frame.
            matrix_transToFluxRopeFrame = np.array([X_unitVector, Y_unitVector, Z_unitVector]).T
            B_inFR = B_inGSE.dot(matrix_transToFluxRopeFrame)

            # Project VHT_inGSE into FluxRope Frame.
            VHT_inFR = VHT_inGSE.dot(matrix_transToFluxRopeFrame)

            # Calculate A(x,0) by integrating By. A(x,0) = Integrate[-By(s,0)ds, {s, 0, x}], where ds = -Vht dot unit(X) dt.
            ds = - VHT_inFR[0] * 1000.0 * dt # Space increment along X axis. Convert km/s to m/s.
            # Don't forget to convert km/s to m/s, and convert nT to T. By = B_inFR[1]
            A = integrate.cumtrapz(-B_inFR[1]*1e-9, dx=ds, initial=0)
            # Calculate Pt(x,0). Pt(x,0)=p(x,0)+Bz^2/(2mu0). By = B_inFR[2]
            Pt = np.array((B_inFR[2] * 1e-9)**2 / (2.0*mu0) * 1e9) # 1e9 convert unit form pa to npa.
            # Find the index of turnPoint.
            index_turnTime = B_inFR.index.get_loc(fluxRopeTurnTime)
            # Split A and Pt into two branches.
            A_sub1 = A[:index_turnTime+1]
            A_sub2 = A[index_turnTime:]
            Pt_sub1 = Pt[:index_turnTime+1]
            Pt_sub2 = Pt[index_turnTime:]
            
            A_tail1 = A[0]
            A_tail2 = A[-1]
            A_turn_point = A[index_turnTime]
            Pt_turn_point = Pt[index_turnTime]
            A_max = max(A)
            A_min = min(A)
            
            # Debug.
            if 0:
                print('Debug:')
                print('fluxRopeStartTime = {}'.format(fluxRopeStartTime))
                print('fluxRopeTurnTime = {}'.format(fluxRopeTurnTime))
                print('fluxRopeEndTime = {}'.format(fluxRopeEndTime))
                print('theta_deg, phi_deg = {}'.format(theta_deg, phi_deg))
                print('Residue_diff = {}'.format(Residue_diff))
                print('Residue_fit = {}'.format(Residue_fit))
                print('VHT_inGSE = {}'.format(VHT_inGSE))
                #print('A = {}'.format(A))
                print('A_turn_point = {}'.format(A_turn_point))
                print('A_tail1 = {}'.format(A_tail1))
                print('A_tail2 = {}'.format(A_tail2))
            
            
            # Find the tail of A, the head is turn point.
            if (A_turn_point > A_tail1)and(A_turn_point > A_tail2):
                A_tail = min(A_tail1, A_tail2)
            elif (A_turn_point < A_tail1)and(A_turn_point < A_tail2):
                A_tail = max(A_tail1, A_tail2)
            else:
                print('No double-folding, discard this record. When happen, please check!')
                continue
            
            z = np.polyfit(A, Pt, 3)
            Func_Pt_A = np.poly1d(z)
            A_turn_fit = A_turn_point
            Pt_turn_fit = Func_Pt_A(A_turn_point)

            # Calculate std of the residual.
            Func_Pt_A_value = Func_Pt_A(A)
            max_Func_Pt_A_value = max(Func_Pt_A_value)
            min_Func_Pt_A_value = min(Func_Pt_A_value)
            Pt_fit_std = np.std(Pt - Func_Pt_A(A))/(max_Func_Pt_A_value - min_Func_Pt_A_value)

            # Set flag.
            lowTail = ((Func_Pt_A(A_tail) - min_Func_Pt_A_value)/(max_Func_Pt_A_value - min_Func_Pt_A_value)) < max_tailPercentile
            lowTailDiff = abs(A_tail1 - A_tail2)/(A_max - A_min) < max_tailDiff
            lowStd = Pt_fit_std <= max_PtFitStd
            eventList_DF_7_fineFittingCurve_temp.loc[index, 'lowTail'] = lowTail
            eventList_DF_7_fineFittingCurve_temp.loc[index, 'lowTailDiff'] = lowTailDiff
            eventList_DF_7_fineFittingCurve_temp.loc[index, 'lowStd'] = lowStd
            
            #eventList_DF_7_fineFittingCurve_temp.loc[index, 'lowTail_value'] = ((Func_Pt_A(A_tail) - min_Func_Pt_A_value)/(max_Func_Pt_A_value - min_Func_Pt_A_value))
            #eventList_DF_7_fineFittingCurve_temp.loc[index, 'lowTailDiff_value'] = abs(A_tail1 - A_tail2)/(A_max - A_min)
            #eventList_DF_7_fineFittingCurve_temp.loc[index, 'lowStd_value'] = Pt_fit_std
            #print(eventList_DF_7_fineFittingCurve_temp)
            
            # Debug.
            if 0:
                plt.plot(A_turn_fit, Pt_turn_fit, 'g^-')
                plt.plot(A_sub1, Pt_sub1, 'ro-', A_sub2, Pt_sub2, 'bo-', np.sort(A), Func_Pt_A(np.sort(A)),'g--')
                plt.plot(A_turn_fit, Pt_turn_fit, 'g^-')
                plt.title('diff={},  fit={}, std={} \n{}~{} lowTail={},lowTailDiff={}'.format(Residue_diff, Residue_fit, Pt_fit_std,  fluxRopeStartTime, fluxRopeEndTime, lowTail, lowTailDiff))
                plt.show()
            
        
        keepMask = (eventList_DF_7_fineFittingCurve_temp['lowTail']&eventList_DF_7_fineFittingCurve_temp['lowTailDiff']&eventList_DF_7_fineFittingCurve_temp['lowStd'])
        eventList_DF_7_fineFittingCurve_temp = eventList_DF_7_fineFittingCurve_temp[keepMask]
        eventList_DF_7_fineFittingCurve_temp.reset_index(drop=True, inplace=True)

        # Drop diff.
        eventList_DF_7_fineFittingCurve_temp = eventList_DF_7_fineFittingCurve_temp.drop('lowTail', axis=1)
        eventList_DF_7_fineFittingCurve_temp = eventList_DF_7_fineFittingCurve_temp.drop('lowTailDiff', axis=1)
        eventList_DF_7_fineFittingCurve_temp = eventList_DF_7_fineFittingCurve_temp.drop('lowStd', axis=1)
        #eventList_DF_7_fineFittingCurve_temp = eventList_DF_7_fineFittingCurve_temp.drop('lowTail_value', axis=1)
        #eventList_DF_7_fineFittingCurve_temp = eventList_DF_7_fineFittingCurve_temp.drop('lowTailDiff_value', axis=1)
        #eventList_DF_7_fineFittingCurve_temp = eventList_DF_7_fineFittingCurve_temp.drop('lowStd_value', axis=1)
        
        # Reset index
        eventList_DF_7_fineFittingCurve_temp.reset_index(drop=True, inplace=True)
        if isVerbose:
            print('After cleaning, total records is {}.'.format(len(eventList_DF_7_fineFittingCurve_temp)))
        print('Done.')

        # 7) Check point 7.
        # After removing bad curve, check if eventList_DF_7_fineFittingCurve_temp is empty.
        if eventList_DF_7_fineFittingCurve_temp.empty:
            # eventList_DF_7_fineFittingCurve_temp is empty. Do nothing.
            print('\nDataFrame eventList_DF_7_fineFittingCurve_temp is empty!')
            print('Go the the next iteration!')
            continue
        
        # Clean other overlapped events, with turnTime difference longer than 5 minutes.
        # Sort eventList_DF_7_fineFittingCurve_temp by endTime.
        eventList_DF_7_fineFittingCurve_temp = eventList_DF_7_fineFittingCurve_temp.sort_values(by='endTime')
        # Make a copy.
        eventList_DF_7_fineFittingCurve_copy_temp = eventList_DF_7_fineFittingCurve_temp.copy()
        # Use the interval scheduling greedy algorithm to remove the overlapes.
        print('\nRemoving overlapped events...')
        if isVerbose:
            print('Before removing, total records is {}.'.format(len(eventList_DF_7_fineFittingCurve_temp)))
        # Drop keepFlag.
        eventList_DF_7_fineFittingCurve_copy_temp.drop('keepFlag', axis=1, inplace=True)
        # Sort by endTime.
        eventList_DF_7_fineFittingCurve_copy_temp.sort_values(by='endTime', inplace=True)
        # Reset index.
        eventList_DF_7_fineFittingCurve_copy_temp.reset_index(drop=True, inplace=True)
        # Add reduced_residue (not used).
        eventList_DF_7_fineFittingCurve_copy_temp['reduced_residue'] = eventList_DF_7_fineFittingCurve_copy_temp['residue_diff']*0.4 + eventList_DF_7_fineFittingCurve_copy_temp['residue_fit']*0.6
        
        # Debug.
        if 0:
            print('eventList_DF_7_fineFittingCurve_copy_temp:')
            print(eventList_DF_7_fineFittingCurve_copy_temp)
        
        # Remove overlap using greedy algorithm.
        count_scheduled_record = 0
        while len(eventList_DF_7_fineFittingCurve_copy_temp) != 0:
            # Find all records overlap with the first one, including itself.
            # Get end time of first one.
            endTime_temp = eventList_DF_7_fineFittingCurve_copy_temp['endTime'].iloc[0]
            # Find the index of all overlapped records, including itself.
            index_df_overlap_temp = eventList_DF_7_fineFittingCurve_copy_temp['startTime'] < endTime_temp
            # Save the first one into eventList_DF_noOverlap.
            eventList_DF_noOverlap = eventList_DF_noOverlap.append(eventList_DF_7_fineFittingCurve_copy_temp.iloc[0], ignore_index=True)
            # Counter + 1.
            count_scheduled_record += 1
            # Remove all the overlapped records from eventList_DF_7_fineFittingCurve_copy_temp, including itself.
            eventList_DF_7_fineFittingCurve_copy_temp = eventList_DF_7_fineFittingCurve_copy_temp[~index_df_overlap_temp]
        if isVerbose:
            print('After removing, total records is {}.'.format(count_scheduled_record))
        # Sort total records list by endTime.
        eventList_DF_noOverlap.sort_values(by='endTime', inplace=True)
        print('Done.')

        print('\nAppending records to eventList_DF_noOverlap...') # This step has been done in while loop.
        print('Done.')

        if isPrintIntermediateDF:
            print('\neventList_DF_0_original_temp:')
            print(eventList_DF_0_original_temp)
            print('\neventList_DF_1_fitSlot_temp:')
            print(eventList_DF_1_fitSlot_temp)
            print('\neventList_DF_2_fineResidue_temp:')
            print(eventList_DF_2_fineResidue_temp)
            print('\neventList_DF_3_combinedByTurnTime_temp:')
            print(eventList_DF_3_combinedByTurnTime_temp)
            print('\neventList_DF_4_passWalenTest_temp:')
            print(eventList_DF_4_passWalenTest_temp)
            print('\neventList_DF_5_noDiscontinuity_temp:')
            print(eventList_DF_5_noDiscontinuity_temp)
            print('\neventList_DF_6_cleanedTurnTime_temp:')
            print(eventList_DF_6_cleanedTurnTime_temp)
            print('\neventList_DF_7_fineFittingCurve_temp:')
            print(eventList_DF_7_fineFittingCurve_temp)
            print('\neventList_DF_noOverlap:')
            print(eventList_DF_noOverlap)

    # Reset index.
    eventList_DF_noOverlap.reset_index(drop=True, inplace=True)
    # Save DataFrame to pickle file.
    print('\nSaving eventList_DF_noOverlap to pickle file...')
    # If plotFolder does not exist, create it.
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    eventList_DF_noOverlap.to_pickle(output_dir + '/' + output_filename + '.p')
    print('Done.')

    return eventList_DF_noOverlap

#########################################################################

# Calculate more information of given flux rope.
def get_more_flux_rope_info_old_without_jz(data_DF, dataObject_or_dataPath, **kwargs):
    # Input content: dataObject_or_dataPath should be the data or path of no overlapped eventlist.
    # Check input datatype:
    # If dataObject_or_dataPath is an object(dict):
    if isinstance(dataObject_or_dataPath, pd.DataFrame):
        print('\nYour input is a DataFrame data.')
        search_result_no_overlap_DF = dataObject_or_dataPath
    elif isinstance(dataObject_or_dataPath, str):
        print('\nYour input is a path. Load the dictionary data via this path.')
        search_result_no_overlap_DF = pd.read_pickle(open(dataObject_or_dataPath, 'rb'))
    else:
        print('\nPlease input the correct datatype! The input data must be a DataFrame or the path of a DataFrame!')
        return None
    
    # Set default value.
    # output filename.
    output_filename = 'search_result_detailed_info'
    # output dir.
    output_dir = os.getcwd()
    # isVerbose.
    isVerbose = False
    
    print('\nDefault parameters:')
    print('output_dir            = {}.'.format(output_dir))
    print('output_filename       = {}.'.format(output_filename))
    
    # If keyword is specified, overwrite the default value.
    if 'output_dir' in kwargs:
        output_dir = kwargs['output_dir']
        print('output_dir is set to {}.'.format(output_dir))
    if 'output_filename' in kwargs:
        output_filename = kwargs['output_filename']
        print('output_filename is set to {}.'.format(output_filename))
    if 'isVerbose' in kwargs:
        isVerbose = kwargs['isVerbose']
        print('isVerbose is set to {}'.format())
    
    # Create an empty dataframe.
    eventList_DF_detailedInfo = pd.DataFrame(columns=['startTime', 'turnTime', 'endTime', 'duration', 'residue_diff', 'residue_fit', 'theta_deg', 'phi_deg', 'VHT_inGSE[0]', 'VHT_inGSE[1]', 'VHT_inGSE[2]', 'X_unitVector[0]', 'X_unitVector[1]', 'X_unitVector[2]', 'Y_unitVector[0]', 'Y_unitVector[1]', 'Y_unitVector[2]', 'Z_unitVector[0]', 'Z_unitVector[1]', 'Z_unitVector[2]', 'walenTest_slope', 'walenTest_intercept', 'walenTest_r_value',  'B_abs_mean', 'Bx_abs_mean', 'By_abs_mean', 'Bz_abs_mean', 'B_std', 'Bx_std', 'By_std', 'Bz_std', 'Bx_inFR_abs_mean', 'By_inFR_abs_mean', 'Bz_inFR_abs_mean', 'Bx_inFR_std', 'By_inFR_std', 'Bz_inFR_std', 'B_magnitude_max', 'Vsw_magnitude_mean', 'Tp_mean', 'Np_mean', 'Te_mean', 'Beta_mean', 'Beta_p_mean', 'lambda1', 'lambda2', 'lambda3', 'eigenVectorMaxVar_lambda1[0]', 'eigenVectorMaxVar_lambda1[1]', 'eigenVectorMaxVar_lambda1[2]', 'eigenVectorInterVar_lambda2[0]', 'eigenVectorInterVar_lambda2[1]', 'eigenVectorInterVar_lambda2[2]', 'eigenVectorMinVar_lambda3[0]', 'eigenVectorMinVar_lambda3[1]', 'eigenVectorMinVar_lambda3[2]'])

    for index_FR in xrange(len(search_result_no_overlap_DF)):
        print('\nCalculating detailed information of flux ropes: {}/{}...'.format(index_FR+1, len(search_result_no_overlap_DF)))
        oneEvent = search_result_no_overlap_DF.iloc[index_FR]
        startTime = oneEvent['startTime']
        turnTime  = oneEvent['turnTime']
        endTime  = oneEvent['endTime']
        duration  = oneEvent['duration']
        residue_diff = oneEvent['residue_diff']
        residue_fit = oneEvent['residue_fit']
        theta_deg, phi_deg = oneEvent['theta_phi']
        VHT_inGSE = np.array(oneEvent['VHT'])
        
        '''
        print('startTime = {}'.format(startTime))
        print('turnTime = {}'.format(turnTime))
        print('endTime = {}'.format(endTime))
        print('(theta_deg, phi_deg) = ({},{})'.format(theta_deg, phi_deg))
        print('residue_diff = {}'.format(residue_diff))
        print('residue_fit = {}'.format(residue_fit))
        '''
        
        # Grab data in specific range.
        selectedRange_mask = (data_DF.index >= startTime) & (data_DF.index <= endTime)
        # The data of fluxrope candidate.
        data_oneFR_DF = data_DF.iloc[selectedRange_mask]
        #print(data_oneFR_DF)


        # Physics constants.
        mu0 = 4.0 * np.pi * 1e-7 #(N/A^2) magnetic constant permeability of free space vacuum permeability
        m_proton = 1.6726219e-27 # Proton mass. In kg.
        factor_deg2rad = np.pi/180.0 # Convert degree to rad.
        k_Boltzmann = 1.3806488e-23 # Boltzmann constant, in J/K.
        # Parameters.
        dt = 60.0 # For 1 minute resolution data, the time increment is 60 seconds.
        
        # Keys: Index([u'Bx', u'By', u'Bz', u'Vx', u'Vy', u'Vz', u'Np', u'Tp', u'Te'], dtype='object')
        # Get Magnetic field slice.
        B_inGSE = data_oneFR_DF.ix[:,['Bx', 'By', 'Bz']] # Produce a reference.
        # Get the solar wind slice.
        Vsw_inGSE = data_oneFR_DF.ix[:,['Vx', 'Vy', 'Vz']] # Produce a reference.
        # Get the proton number density slice.
        Np = data_oneFR_DF.ix[:,['Np']] # Produce a reference.
        # Get the proton temperature slice. In Kelvin.
        Tp = data_oneFR_DF.ix[:,['Tp']] # Produce a reference.
        if 'Te' in data_oneFR_DF.keys():
            # Get the electron temperature slice. In Kelvin.
            Te = data_oneFR_DF.ix[:,['Te']] # Produce a reference.
        
        # If there is any NaN in B_inGSE, try to interpolate.
        if B_inGSE.isnull().values.sum():
            if isVerbose:
                print('Found NaNs, interpolate B.')
            B_inGSE_copy = B_inGSE.copy()
            # limit=3 means only interpolate the gap shorter than 4.
            B_inGSE_copy.interpolate(method='time', limit=None, inplace=True)
            # interpolate won't fill leading NaNs, so we use backward fill.
            B_inGSE_copy.bfill(inplace=True)
            B_inGSE_copy.ffill(inplace=True)
            if B_inGSE_copy.isnull().values.sum():
                print('Too many NaNs in B. Skip this record. If this situation happens, please check.')
                detailed_info_dict = None
                continue
            else:
                B_inGSE = B_inGSE_copy

        # If there is any NaN in Vsw_inGSE, try to interpolate.
        if Vsw_inGSE.isnull().values.sum():
            if isVerbose:
                print('Found NaNs, interpolate Vsw.')
            Vsw_inGSE_copy = Vsw_inGSE.copy()
            # limit=3 means only interpolate the gap shorter than 4.
            Vsw_inGSE_copy.interpolate(method='time', limit=None, inplace=True)
            # interpolate won't fill leading NaNs, so we use backward fill.
            Vsw_inGSE_copy.bfill(inplace=True)
            Vsw_inGSE_copy.ffill(inplace=True)
            if Vsw_inGSE_copy.isnull().values.sum():
                print('Too many NaNs in Vsw. Skip this record. If this situation happens, please check.')
                detailed_info_dict = None
                continue
            else:
                Vsw_inGSE = Vsw_inGSE_copy
                
        # If there is any NaN in Np, try to interpolate.
        if Np.isnull().values.sum():
            if isVerbose:
                print('Found NaNs, interpolate Np.')
            Np_copy = Np.copy()
            # limit=3 means only interpolate the gap shorter than 4.
            Np_copy.interpolate(method='time', limit=None, inplace=True)
            # interpolate won't fill leading NaNs, so we use backward fill.
            Np_copy.bfill(inplace=True)
            Np_copy.ffill(inplace=True)
            if Np_copy.isnull().values.sum():
                print('Too many NaNs in Vsw. Skip this record. If this situation happens, please check.')
                detailed_info_dict = None
                continue
            else:
                Np = Np_copy

        # Direction cosines:
        # x = rcos(alpha) = rsin(theta)cos(phi) => cos(alpha) = sin(theta)cos(phi)
        # y = rcos(beta)  = rsin(theta)sin(phi) => cos(beta)  = sin(theta)sin(phi)
        # z = rcos(gamma) = rcos(theta)         => cos(gamma) = cos(theta)
        # Use direction cosines to construct a unit vector.
        theta_rad = factor_deg2rad * theta_deg
        phi_rad   = factor_deg2rad * phi_deg

        # Form new Z_unitVector according to direction cosines.
        Z_unitVector = np.array([np.sin(theta_rad)*np.cos(phi_rad), np.sin(theta_rad)*np.sin(phi_rad), np.cos(theta_rad)])
        # Find X axis from Z axis and -VHT.
        X_unitVector = findXaxis(Z_unitVector, -VHT_inGSE)
        # Find the Y axis to form a right-handed coordinater with X and Z.
        Y_unitVector = formRighHandFrame(X_unitVector, Z_unitVector)

        # Project B_inGSE into FluxRope Frame.
        matrix_transToFluxRopeFrame = np.array([X_unitVector, Y_unitVector, Z_unitVector]).T
        B_inFR = B_inGSE.dot(matrix_transToFluxRopeFrame)

        # Project VHT_inGSE into FluxRope Frame.
        VHT_inFR = VHT_inGSE.dot(matrix_transToFluxRopeFrame)
        # Project Vsw_inFR into FluxRope Frame.
        Vsw_inFR = Vsw_inGSE.dot(matrix_transToFluxRopeFrame)
        
        # Calculate the covariance matrix of Magnetic field.
        covM_B_inGSE = B_inGSE.cov()
        # Calculate the eigenvalues and eigenvectors of convariance matrix of B field.
        lambda1, lambda2, lambda3, eigenVectorMaxVar_lambda1, eigenVectorInterVar_lambda2, eigenVectorMinVar_lambda3 = eigenMatrix(covM_B_inGSE, formXYZ=True)

        '''
        # Project B_DataFrame onto new Frame(MVB frame).The dot product of two dataframe requires the
        # columns and indices are same, so we convert to np.array.
        B_inMVB = B_inGSE.dot(np.array(eigenVectors_covM_B_inGSE))
        # Project VHt_inFR onto new Frame(MVB frame).The dot product of two dataframe requires the
        # columns and indices are same, so we convert to np.array.
        VHT_inMVB = VHT_inGSE.dot(np.array(eigenVectors_covM_B_inGSE))
        '''

        # Calculate A(x,0) by integrating By. A(x,0) = Integrate[-By(s,0)ds, {s, 0, x}], where ds = -Vht dot unit(X) dt.
        ds = - VHT_inFR[0] * 1000.0 * dt # Space increment along X axis. Convert km/s to m/s.
        # Don't forget to convert km/s to m/s, and convert nT to T. By = B_inFR[1]
        A = integrate.cumtrapz(-B_inFR[1]*1e-9, dx=ds, initial=0)
        # Calculate Pt(x,0). Pt(x,0)=p(x,0)+Bz^2/(2mu0). By = B_inFR[2]
        Pt = np.array((B_inFR[2] * 1e-9)**2 / (2.0*mu0) * 1e9) # 1e9 convert unit form pa to npa.
        # Find the index of turnPoint.
        index_turnTime = B_inFR.index.get_loc(turnTime)
        # Split A and Pt into two branches.
        A_sub1 = A[:index_turnTime+1]
        A_sub2 = A[index_turnTime:]
        Pt_sub1 = Pt[:index_turnTime+1]
        Pt_sub2 = Pt[index_turnTime:]
        
        z = np.polyfit(A, Pt, 3)
        Func_Pt_A = np.poly1d(z)

        '''
        plt.plot(A_sub1, Pt_sub1, 'ro-', A_sub2, Pt_sub2, 'bo-', np.sort(A), Func_Pt_A(np.sort(A)),'g--')
        plt.title('residue_diff = {},  residue_fit = {}'.format(residue_diff, residue_fit))
        plt.show()
        '''
        
        # Apply walen test on the result(in optimal frame).
        # Proton mass density. Original Np is in #/cc ( cc = cubic centimeter). Multiply by 1e6 to convert cc to m^3.
        P_massDensity = Np * m_proton * 1e6 # In kg/m^3.
        len_P_massDensity = len(P_massDensity)
        P_massDensity_array = np.array(P_massDensity)
        P_massDensity_array = np.reshape(P_massDensity_array, (len_P_massDensity, 1))
        # Alfven speed. Multiply by 1e-9 to convert nT to T. Divided by 1000.0 to convert m/s to km/s.
        VA_inFR = np.array(B_inFR * 1e-9) / np.sqrt(mu0 * P_massDensity_array) / 1000.0
        VA_inFR_1D = np.reshape(VA_inFR, VA_inFR.size)
        V_remaining = np.array(Vsw_inFR - VHT_inFR)
        V_remaining_1D = np.reshape(V_remaining, V_remaining.size)
        # Call walen test function.
        # First row is x component, second row is y component, third row is z component.
        walenTest_slope, walenTest_intercept, walenTest_r_value = walenTest(VA_inFR_1D, V_remaining_1D)
        
        # Get B statistical properties.
        B_norm_DF = pd.DataFrame(np.sqrt(np.square(B_inGSE).sum(axis=1)),columns=['|B|'])
        B_magnitude_max = B_norm_DF['|B|'].max(skipna=True)
        B_inGSE = pd.concat([B_inGSE, B_norm_DF], axis=1)
        B_std_Series = B_inGSE.std(axis=0,skipna=True,numeric_only=True)
        B_abs_mean_Series = B_inGSE.abs().mean(axis=0,skipna=True,numeric_only=True)
        
        B_abs_mean = round(B_abs_mean_Series['|B|'],4)
        Bx_abs_mean = round(B_abs_mean_Series[0],4)
        By_abs_mean = round(B_abs_mean_Series[1],4)
        Bz_abs_mean = round(B_abs_mean_Series[2],4)
        B_std = round(B_std_Series['|B|'],4)
        Bx_std = round(B_std_Series[0],4)
        By_std = round(B_std_Series[1],4)
        Bz_std = round(B_std_Series[2],4)
        
        # B_inFR.
        B_inFR_std_Series = B_inFR.std(axis=0,skipna=True,numeric_only=True)
        B_inFR_abs_mean_Series = B_inFR.abs().mean(axis=0,skipna=True,numeric_only=True)
        Bx_inFR_abs_mean = round(B_inFR_abs_mean_Series[0],4)
        By_inFR_abs_mean = round(B_inFR_abs_mean_Series[1],4)
        Bz_inFR_abs_mean = round(B_inFR_abs_mean_Series[2],4)
        Bx_inFR_std = round(B_inFR_std_Series[0],4)
        By_inFR_std = round(B_inFR_std_Series[1],4)
        Bz_inFR_std = round(B_inFR_std_Series[2],4)
        
        # Get Vsw statistical properties.
        Vsw_norm_DF = pd.DataFrame(np.sqrt(np.square(Vsw_inGSE).sum(axis=1)),columns=['|Vsw|'])
        Vsw_magnitude_mean = Vsw_norm_DF['|Vsw|'].mean(skipna=True)
        
        # Get Plasma Beta statistical properties.
        Tp_mean = np.mean(np.ma.masked_invalid(np.array(Tp['Tp']))) # Exclude nan and inf.
        # Divided by 1e6 to convert unit to 10^6K.
        Tp_mean = Tp_mean/1e6
        Tp_mean = round(Tp_mean, 6)
        # Calculate Np_mean.
        # Original Np is in #/cc ( cc = cubic centimeter).
        Np_mean = float(Np.mean(skipna=True, numeric_only=True))# In #/cc
        # Calculate Te_mean.
        if 'Te' in data_oneFR_DF.keys():
            # Divided by 1e6 to convert unit to 10^6K.
            Te_mean = float(Te.mean(skipna=True, numeric_only=True))/1e6
        else:
            Te_mean = None
        
        #print('Np_mean = {}'.format(Np_mean))
        #print('Te_mean = {}'.format(Te_mean))
        #print('Tp_mean = {}'.format(Tp_mean))
        
        # Calculate plasma Dynamic Pressure PD.
        # Original Np is in #/cc ( cc = cubic centimeter). Multiply by 1e6 to convert cc to m^3.
        Pp = np.array(Np['Np']) * 1e6 * k_Boltzmann * np.array(Tp['Tp']) # Proton pressure.
        if 'Te' in data_oneFR_DF.keys():
            Pe = np.array(Np['Np']) * 1e6 * k_Boltzmann * np.array(Te['Te']) # Electron pressure.
            PD = Pp + Pe # Total dynamic pressure.
        else:
            PD = Pp

        # Calculate plasma Magnetic pressure PB.
        PB = (np.array(B_norm_DF['|B|'])*1e-9)**2/(2*mu0)
        # Calculate plasma Beta = PD/PB
        Beta = PD/PB
        Beta_mean = np.mean(np.ma.masked_invalid(Beta)) # Exclude nan and inf.
        Beta_p = Pp/PB
        Beta_p_mean = np.mean(np.ma.masked_invalid(Beta_p))

        detailed_info_dict = {'startTime':startTime, 'turnTime':turnTime, 'endTime':endTime, 'duration':duration, 'residue_diff':residue_diff, 'residue_fit':residue_fit, 'theta_deg':theta_deg, 'phi_deg':phi_deg, 'VHT_inGSE[0]':VHT_inGSE[0], 'VHT_inGSE[1]':VHT_inGSE[1], 'VHT_inGSE[2]':VHT_inGSE[2], 'X_unitVector[0]':X_unitVector[0], 'X_unitVector[1]':X_unitVector[1], 'X_unitVector[2]':X_unitVector[2], 'Y_unitVector[0]':Y_unitVector[0], 'Y_unitVector[1]':Y_unitVector[1], 'Y_unitVector[2]':Y_unitVector[2], 'Z_unitVector[0]':Z_unitVector[0], 'Z_unitVector[1]':Z_unitVector[1], 'Z_unitVector[2]':Z_unitVector[2], 'walenTest_slope':walenTest_slope, 'walenTest_intercept':walenTest_intercept, 'walenTest_r_value':walenTest_r_value,  'B_abs_mean':B_abs_mean, 'Bx_abs_mean':Bx_abs_mean, 'By_abs_mean':By_abs_mean, 'Bz_abs_mean':Bz_abs_mean, 'B_std':B_std, 'Bx_std':Bx_std, 'By_std':By_std, 'Bz_std':Bz_std, 'Bx_inFR_abs_mean':Bx_inFR_abs_mean, 'By_inFR_abs_mean':By_inFR_abs_mean, 'Bz_inFR_abs_mean':Bz_inFR_abs_mean, 'Bx_inFR_std':Bx_inFR_std, 'By_inFR_std':By_inFR_std, 'Bz_inFR_std':Bz_inFR_std, 'B_magnitude_max':B_magnitude_max, 'Vsw_magnitude_mean':Vsw_magnitude_mean, 'Tp_mean':Tp_mean, 'Np_mean':Np_mean, 'Te_mean':Te_mean, 'Beta_mean':Beta_mean, 'Beta_p_mean':Beta_p_mean, 'lambda1':lambda1, 'lambda2':lambda2, 'lambda3':lambda3, 'eigenVectorMaxVar_lambda1[0]':eigenVectorMaxVar_lambda1[0], 'eigenVectorMaxVar_lambda1[1]':eigenVectorMaxVar_lambda1[1], 'eigenVectorMaxVar_lambda1[2]':eigenVectorMaxVar_lambda1[2], 'eigenVectorInterVar_lambda2[0]':eigenVectorInterVar_lambda2[0], 'eigenVectorInterVar_lambda2[1]':eigenVectorInterVar_lambda2[1], 'eigenVectorInterVar_lambda2[2]':eigenVectorInterVar_lambda2[2], 'eigenVectorMinVar_lambda3[0]':eigenVectorMinVar_lambda3[0], 'eigenVectorMinVar_lambda3[1]':eigenVectorMinVar_lambda3[1], 'eigenVectorMinVar_lambda3[2]':eigenVectorMinVar_lambda3[2]}

        # Append detailed_info_dict to FR_detailed_info_DF.
        if not (detailed_info_dict is None):
            eventList_DF_detailedInfo = eventList_DF_detailedInfo.append(detailed_info_dict, ignore_index=True)

    # Save DataFrame to pickle file.
    print('\nSaving eventList_DF_detailedInfo to pickle file...')
    # If plotFolder does not exist, create it.
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    eventList_DF_detailedInfo.to_pickle(output_dir + '/' + output_filename + '.p')
    print('Done.')
    
    return eventList_DF_detailedInfo

###############################################################################

# Calculate more information of given flux rope.
def get_more_flux_rope_info(data_DF, dataObject_or_dataPath, **kwargs):
    # Input content: dataObject_or_dataPath should be the data or path of no overlapped eventlist.
    # Check input datatype:
    # If dataObject_or_dataPath is an object(dict):
    if isinstance(dataObject_or_dataPath, pd.DataFrame):
        print('\nYour input is a DataFrame data.')
        search_result_no_overlap_DF = dataObject_or_dataPath
    elif isinstance(dataObject_or_dataPath, str):
        print('\nYour input is a path. Load the dictionary data via this path.')
        search_result_no_overlap_DF = pd.read_pickle(open(dataObject_or_dataPath, 'rb'))
    else:
        print('\nPlease input the correct datatype! The input data must be a DataFrame or the path of a DataFrame!')
        return None
    
    # Set default value.
    # output filename.
    output_filename = 'search_result_detailed_info'
    # output dir.
    output_dir = os.getcwd()
    # isVerbose.
    isVerbose = False
    
    print('\nDefault parameters:')
    print('output_dir            = {}.'.format(output_dir))
    print('output_filename       = {}.'.format(output_filename))
    
    # If keyword is specified, overwrite the default value.
    if 'output_dir' in kwargs:
        output_dir = kwargs['output_dir']
        print('output_dir is set to {}.'.format(output_dir))
    if 'output_filename' in kwargs:
        output_filename = kwargs['output_filename']
        print('output_filename is set to {}.'.format(output_filename))
    if 'isVerbose' in kwargs:
        isVerbose = kwargs['isVerbose']
        print('isVerbose is set to {}'.format())
    
    # Create an empty dataframe.
    eventList_DF_detailedInfo = pd.DataFrame(columns=['startTime', 'turnTime', 'endTime', 'duration', 'residue_diff', 'residue_fit', 'theta_deg', 'phi_deg', 'A_range', 'Pt_coeff', 'Path_length', 'VHT_inGSE[0]', 'VHT_inGSE[1]', 'VHT_inGSE[2]', 'X_unitVector[0]', 'X_unitVector[1]', 'X_unitVector[2]', 'Y_unitVector[0]', 'Y_unitVector[1]', 'Y_unitVector[2]', 'Z_unitVector[0]', 'Z_unitVector[1]', 'Z_unitVector[2]', 'walenTest_slope', 'walenTest_intercept', 'walenTest_r_value',  'B_abs_mean', 'Bx_abs_mean', 'By_abs_mean', 'Bz_abs_mean', 'B_std', 'Bx_std', 'By_std', 'Bz_std', 'Bx_inFR_abs_mean', 'By_inFR_abs_mean', 'Bz_inFR_abs_mean', 'Bx_inFR_std', 'By_inFR_std', 'Bz_inFR_std', 'B_magnitude_max', 'Vsw_magnitude_mean', 'Tp_mean', 'Np_mean', 'Te_mean', 'Beta_mean', 'Beta_p_mean', 'lambda1', 'lambda2', 'lambda3', 'eigenVectorMaxVar_lambda1[0]', 'eigenVectorMaxVar_lambda1[1]', 'eigenVectorMaxVar_lambda1[2]', 'eigenVectorInterVar_lambda2[0]', 'eigenVectorInterVar_lambda2[1]', 'eigenVectorInterVar_lambda2[2]', 'eigenVectorMinVar_lambda3[0]', 'eigenVectorMinVar_lambda3[1]', 'eigenVectorMinVar_lambda3[2]'])

    for index_FR in xrange(len(search_result_no_overlap_DF)):
        print('\nCalculating detailed information of flux ropes: {}/{}...'.format(index_FR+1, len(search_result_no_overlap_DF)))
        oneEvent = search_result_no_overlap_DF.iloc[index_FR]
        startTime = oneEvent['startTime']
        turnTime  = oneEvent['turnTime']
        endTime  = oneEvent['endTime']
        duration  = oneEvent['duration']
        residue_diff = oneEvent['residue_diff']
        residue_fit = oneEvent['residue_fit']
        theta_deg, phi_deg = oneEvent['theta_phi']
        VHT_inGSE = np.array(oneEvent['VHT'])
        
        '''
        print('startTime = {}'.format(startTime))
        print('turnTime = {}'.format(turnTime))
        print('endTime = {}'.format(endTime))
        print('(theta_deg, phi_deg) = ({},{})'.format(theta_deg, phi_deg))
        print('residue_diff = {}'.format(residue_diff))
        print('residue_fit = {}'.format(residue_fit))
        '''
        
        # Grab data in specific range.
        selectedRange_mask = (data_DF.index >= startTime) & (data_DF.index <= endTime)
        # The data of fluxrope candidate.
        data_oneFR_DF = data_DF.iloc[selectedRange_mask]
        #print(data_oneFR_DF)


        # Physics constants.
        mu0 = 4.0 * np.pi * 1e-7 #(N/A^2) magnetic constant permeability of free space vacuum permeability
        m_proton = 1.6726219e-27 # Proton mass. In kg.
        factor_deg2rad = np.pi/180.0 # Convert degree to rad.
        k_Boltzmann = 1.3806488e-23 # Boltzmann constant, in J/K.
        # Parameters.
        dt = 60.0 # For 1 minute resolution data, the time increment is 60 seconds.
        
        # Keys: Index([u'Bx', u'By', u'Bz', u'Vx', u'Vy', u'Vz', u'Np', u'Tp', u'Te'], dtype='object')
        # Get Magnetic field slice.
        B_inGSE = data_oneFR_DF.ix[:,['Bx', 'By', 'Bz']] # Produce a reference.
        # Get the solar wind slice.
        Vsw_inGSE = data_oneFR_DF.ix[:,['Vx', 'Vy', 'Vz']] # Produce a reference.
        # Get the proton number density slice.
        Np = data_oneFR_DF.ix[:,['Np']] # Produce a reference.
        # Get the proton temperature slice. In Kelvin.
        Tp = data_oneFR_DF.ix[:,['Tp']] # Produce a reference.
        if 'Te' in data_oneFR_DF.keys():
            # Get the electron temperature slice. In Kelvin.
            Te = data_oneFR_DF.ix[:,['Te']] # Produce a reference.
        
        # If there is any NaN in B_inGSE, try to interpolate.
        if B_inGSE.isnull().values.sum():
            if isVerbose:
                print('Found NaNs, interpolate B.')
            B_inGSE_copy = B_inGSE.copy()
            # limit=3 means only interpolate the gap shorter than 4.
            B_inGSE_copy.interpolate(method='time', limit=None, inplace=True)
            # interpolate won't fill leading NaNs, so we use backward fill.
            B_inGSE_copy.bfill(inplace=True)
            B_inGSE_copy.ffill(inplace=True)
            if B_inGSE_copy.isnull().values.sum():
                print('Too many NaNs in B. Skip this record. If this situation happens, please check.')
                detailed_info_dict = None
                continue
            else:
                B_inGSE = B_inGSE_copy

        # If there is any NaN in Vsw_inGSE, try to interpolate.
        if Vsw_inGSE.isnull().values.sum():
            if isVerbose:
                print('Found NaNs, interpolate Vsw.')
            Vsw_inGSE_copy = Vsw_inGSE.copy()
            # limit=3 means only interpolate the gap shorter than 4.
            Vsw_inGSE_copy.interpolate(method='time', limit=None, inplace=True)
            # interpolate won't fill leading NaNs, so we use backward fill.
            Vsw_inGSE_copy.bfill(inplace=True)
            Vsw_inGSE_copy.ffill(inplace=True)
            if Vsw_inGSE_copy.isnull().values.sum():
                print('Too many NaNs in Vsw. Skip this record. If this situation happens, please check.')
                detailed_info_dict = None
                continue
            else:
                Vsw_inGSE = Vsw_inGSE_copy
                
        # If there is any NaN in Np, try to interpolate.
        if Np.isnull().values.sum():
            if isVerbose:
                print('Found NaNs, interpolate Np.')
            Np_copy = Np.copy()
            # limit=3 means only interpolate the gap shorter than 4.
            Np_copy.interpolate(method='time', limit=None, inplace=True)
            # interpolate won't fill leading NaNs, so we use backward fill.
            Np_copy.bfill(inplace=True)
            Np_copy.ffill(inplace=True)
            if Np_copy.isnull().values.sum():
                print('Too many NaNs in Vsw. Skip this record. If this situation happens, please check.')
                detailed_info_dict = None
                continue
            else:
                Np = Np_copy

        # Direction cosines:
        # x = rcos(alpha) = rsin(theta)cos(phi) => cos(alpha) = sin(theta)cos(phi)
        # y = rcos(beta)  = rsin(theta)sin(phi) => cos(beta)  = sin(theta)sin(phi)
        # z = rcos(gamma) = rcos(theta)         => cos(gamma) = cos(theta)
        # Use direction cosines to construct a unit vector.
        theta_rad = factor_deg2rad * theta_deg
        phi_rad   = factor_deg2rad * phi_deg

        # Form new Z_unitVector according to direction cosines.
        Z_unitVector = np.array([np.sin(theta_rad)*np.cos(phi_rad), np.sin(theta_rad)*np.sin(phi_rad), np.cos(theta_rad)])
        # Find X axis from Z axis and -VHT.
        X_unitVector = findXaxis(Z_unitVector, -VHT_inGSE)
        # Find the Y axis to form a right-handed coordinater with X and Z.
        Y_unitVector = formRighHandFrame(X_unitVector, Z_unitVector)

        # Project B_inGSE into FluxRope Frame.
        matrix_transToFluxRopeFrame = np.array([X_unitVector, Y_unitVector, Z_unitVector]).T
        B_inFR = B_inGSE.dot(matrix_transToFluxRopeFrame)
        
        # Check if Bz has negative values, if does, flip Z-axis direction.
        num_Bz_lt0 = (B_inFR[2]<0).sum()
        num_Bz_gt0 = (B_inFR[2]>0).sum()
        # If the negative Bz is more than positive Bz, filp.
        if (num_Bz_lt0 > num_Bz_gt0):
            # Reverse the direction of Z-axis.
            print('Reverse the direction of Z-axis!')
            Z_unitVector = -Z_unitVector
            # Recalculat theta and phi with new Z_unitVector.
            theta_deg, phi_deg = directionVector2angle(Z_unitVector)
            # Refind X axis frome Z axis and -Vsw.
            X_unitVector = findXaxis(Z_unitVector, -VHT_inGSE)
            # Refind the Y axis to form a right-handed coordinater with X and Z.
            Y_unitVector = formRighHandFrame(X_unitVector, Z_unitVector)
            # Reproject B_inGSE_DataFrame into flux rope (FR) frame.
            matrix_transToFluxRopeFrame = np.array([X_unitVector, Y_unitVector, Z_unitVector]).T
            B_inFR = B_inGSE.dot(matrix_transToFluxRopeFrame)
            
        # Project VHT_inGSE into FluxRope Frame.
        VHT_inFR = VHT_inGSE.dot(matrix_transToFluxRopeFrame)
        # Project Vsw_inFR into FluxRope Frame.
        Vsw_inFR = Vsw_inGSE.dot(matrix_transToFluxRopeFrame)
        
        # Calculate the covariance matrix of Magnetic field.
        covM_B_inGSE = B_inGSE.cov()
        # Calculate the eigenvalues and eigenvectors of convariance matrix of B field.
        lambda1, lambda2, lambda3, eigenVectorMaxVar_lambda1, eigenVectorInterVar_lambda2, eigenVectorMinVar_lambda3 = eigenMatrix(covM_B_inGSE, formXYZ=True)

        '''
        # Project B_DataFrame onto new Frame(MVB frame).The dot product of two dataframe requires the
        # columns and indices are same, so we convert to np.array.
        B_inMVB = B_inGSE.dot(np.array(eigenVectors_covM_B_inGSE))
        # Project VHt_inFR onto new Frame(MVB frame).The dot product of two dataframe requires the
        # columns and indices are same, so we convert to np.array.
        VHT_inMVB = VHT_inGSE.dot(np.array(eigenVectors_covM_B_inGSE))
        '''

        # Calculate A(x,0) by integrating By. A(x,0) = Integrate[-By(s,0)ds, {s, 0, x}], where ds = -Vht dot unit(X) dt.
        ds = - VHT_inFR[0] * 1000.0 * dt # Space increment along X axis. Convert km/s to m/s. m/minutes.
        # Don't forget to convert km/s to m/s, and convert nT to T. By = B_inFR[1]
        A = integrate.cumtrapz(-B_inFR[1]*1e-9, dx=ds, initial=0)
        # Calculate Pt(x,0). Pt(x,0)=p(x,0)+Bz^2/(2mu0). By = B_inFR[2]
        Pt = np.array((B_inFR[2] * 1e-9)**2 / (2.0*mu0) * 1e9) # 1e9 convert unit form pa to npa.
        # Find the index of turnPoint.
        index_turnTime = B_inFR.index.get_loc(turnTime)
        # Split A and Pt into two branches.
        A_sub1 = A[:index_turnTime+1]
        A_sub2 = A[index_turnTime:]
        Pt_sub1 = Pt[:index_turnTime+1]
        Pt_sub2 = Pt[index_turnTime:]
        
        z = np.polyfit(A, Pt, 3)
        Func_Pt_A = np.poly1d(z)
        
        Pt_coeff = list(z)
        A_range = [min(A), max(A)]
        Path_length = ds * duration # The lenght of spacecraft trajectory across the flux rope.

        '''
        plt.plot(A_sub1, Pt_sub1, 'ro-', A_sub2, Pt_sub2, 'bo-', np.sort(A), Func_Pt_A(np.sort(A)),'g--')
        plt.title('residue_diff = {},  residue_fit = {}'.format(residue_diff, residue_fit))
        plt.show()
        '''
        
        # Apply walen test on the result(in optimal frame).
        # Proton mass density. Original Np is in #/cc ( cc = cubic centimeter). Multiply by 1e6 to convert cc to m^3.
        P_massDensity = Np * m_proton * 1e6 # In kg/m^3.
        len_P_massDensity = len(P_massDensity)
        P_massDensity_array = np.array(P_massDensity)
        P_massDensity_array = np.reshape(P_massDensity_array, (len_P_massDensity, 1))
        # Alfven speed. Multiply by 1e-9 to convert nT to T. Divided by 1000.0 to convert m/s to km/s.
        VA_inFR = np.array(B_inFR * 1e-9) / np.sqrt(mu0 * P_massDensity_array) / 1000.0
        VA_inFR_1D = np.reshape(VA_inFR, VA_inFR.size)
        V_remaining = np.array(Vsw_inFR - VHT_inFR)
        V_remaining_1D = np.reshape(V_remaining, V_remaining.size)
        # Call walen test function.
        # First row is x component, second row is y component, third row is z component.
        walenTest_slope, walenTest_intercept, walenTest_r_value = walenTest(VA_inFR_1D, V_remaining_1D)
        
        # Get B statistical properties.
        B_norm_DF = pd.DataFrame(np.sqrt(np.square(B_inGSE).sum(axis=1)),columns=['|B|'])
        B_magnitude_max = B_norm_DF['|B|'].max(skipna=True)
        B_inGSE = pd.concat([B_inGSE, B_norm_DF], axis=1)
        B_std_Series = B_inGSE.std(axis=0,skipna=True,numeric_only=True)
        B_abs_mean_Series = B_inGSE.abs().mean(axis=0,skipna=True,numeric_only=True)
        
        B_abs_mean = round(B_abs_mean_Series['|B|'],4)
        Bx_abs_mean = round(B_abs_mean_Series[0],4)
        By_abs_mean = round(B_abs_mean_Series[1],4)
        Bz_abs_mean = round(B_abs_mean_Series[2],4)
        B_std = round(B_std_Series['|B|'],4)
        Bx_std = round(B_std_Series[0],4)
        By_std = round(B_std_Series[1],4)
        Bz_std = round(B_std_Series[2],4)
        
        # B_inFR.
        B_inFR_std_Series = B_inFR.std(axis=0,skipna=True,numeric_only=True)
        B_inFR_abs_mean_Series = B_inFR.abs().mean(axis=0,skipna=True,numeric_only=True)
        Bx_inFR_abs_mean = round(B_inFR_abs_mean_Series[0],4)
        By_inFR_abs_mean = round(B_inFR_abs_mean_Series[1],4)
        Bz_inFR_abs_mean = round(B_inFR_abs_mean_Series[2],4)
        Bx_inFR_std = round(B_inFR_std_Series[0],4)
        By_inFR_std = round(B_inFR_std_Series[1],4)
        Bz_inFR_std = round(B_inFR_std_Series[2],4)
        
        # Get Vsw statistical properties.
        Vsw_norm_DF = pd.DataFrame(np.sqrt(np.square(Vsw_inGSE).sum(axis=1)),columns=['|Vsw|'])
        Vsw_magnitude_mean = Vsw_norm_DF['|Vsw|'].mean(skipna=True)
        
        # Get Plasma Beta statistical properties.
        Tp_mean = np.mean(np.ma.masked_invalid(np.array(Tp['Tp']))) # Exclude nan and inf.
        # Divided by 1e6 to convert unit to 10^6K.
        Tp_mean = Tp_mean/1e6
        Tp_mean = round(Tp_mean, 6)
        # Calculate Np_mean.
        # Original Np is in #/cc ( cc = cubic centimeter).
        Np_mean = float(Np.mean(skipna=True, numeric_only=True))# In #/cc
        # Calculate Te_mean.
        if 'Te' in data_oneFR_DF.keys():
            # Divided by 1e6 to convert unit to 10^6K.
            Te_mean = float(Te.mean(skipna=True, numeric_only=True))/1e6
        else:
            Te_mean = None
        
        #print('Np_mean = {}'.format(Np_mean))
        #print('Te_mean = {}'.format(Te_mean))
        #print('Tp_mean = {}'.format(Tp_mean))
        
        # Calculate plasma Dynamic Pressure PD.
        # Original Np is in #/cc ( cc = cubic centimeter). Multiply by 1e6 to convert cc to m^3.
        Pp = np.array(Np['Np']) * 1e6 * k_Boltzmann * np.array(Tp['Tp']) # Proton pressure.
        if 'Te' in data_oneFR_DF.keys():
            Pe = np.array(Np['Np']) * 1e6 * k_Boltzmann * np.array(Te['Te']) # Electron pressure.
            PD = Pp + Pe # Total dynamic pressure.
        else:
            PD = Pp

        # Calculate plasma Magnetic pressure PB.
        PB = (np.array(B_norm_DF['|B|'])*1e-9)**2/(2*mu0)
        # Calculate plasma Beta = PD/PB
        Beta = PD/PB
        Beta_mean = np.mean(np.ma.masked_invalid(Beta)) # Exclude nan and inf.
        Beta_p = Pp/PB
        Beta_p_mean = np.mean(np.ma.masked_invalid(Beta_p))

        detailed_info_dict = {'startTime':startTime, 'turnTime':turnTime, 'endTime':endTime, 'duration':duration, 'residue_diff':residue_diff, 'residue_fit':residue_fit, 'theta_deg':theta_deg, 'phi_deg':phi_deg, 'A_range':A_range, 'Pt_coeff':Pt_coeff, 'Path_length':Path_length, 'VHT_inGSE[0]':VHT_inGSE[0], 'VHT_inGSE[1]':VHT_inGSE[1], 'VHT_inGSE[2]':VHT_inGSE[2], 'X_unitVector[0]':X_unitVector[0], 'X_unitVector[1]':X_unitVector[1], 'X_unitVector[2]':X_unitVector[2], 'Y_unitVector[0]':Y_unitVector[0], 'Y_unitVector[1]':Y_unitVector[1], 'Y_unitVector[2]':Y_unitVector[2], 'Z_unitVector[0]':Z_unitVector[0], 'Z_unitVector[1]':Z_unitVector[1], 'Z_unitVector[2]':Z_unitVector[2], 'walenTest_slope':walenTest_slope, 'walenTest_intercept':walenTest_intercept, 'walenTest_r_value':walenTest_r_value,  'B_abs_mean':B_abs_mean, 'Bx_abs_mean':Bx_abs_mean, 'By_abs_mean':By_abs_mean, 'Bz_abs_mean':Bz_abs_mean, 'B_std':B_std, 'Bx_std':Bx_std, 'By_std':By_std, 'Bz_std':Bz_std, 'Bx_inFR_abs_mean':Bx_inFR_abs_mean, 'By_inFR_abs_mean':By_inFR_abs_mean, 'Bz_inFR_abs_mean':Bz_inFR_abs_mean, 'Bx_inFR_std':Bx_inFR_std, 'By_inFR_std':By_inFR_std, 'Bz_inFR_std':Bz_inFR_std, 'B_magnitude_max':B_magnitude_max, 'Vsw_magnitude_mean':Vsw_magnitude_mean, 'Tp_mean':Tp_mean, 'Np_mean':Np_mean, 'Te_mean':Te_mean, 'Beta_mean':Beta_mean, 'Beta_p_mean':Beta_p_mean, 'lambda1':lambda1, 'lambda2':lambda2, 'lambda3':lambda3, 'eigenVectorMaxVar_lambda1[0]':eigenVectorMaxVar_lambda1[0], 'eigenVectorMaxVar_lambda1[1]':eigenVectorMaxVar_lambda1[1], 'eigenVectorMaxVar_lambda1[2]':eigenVectorMaxVar_lambda1[2], 'eigenVectorInterVar_lambda2[0]':eigenVectorInterVar_lambda2[0], 'eigenVectorInterVar_lambda2[1]':eigenVectorInterVar_lambda2[1], 'eigenVectorInterVar_lambda2[2]':eigenVectorInterVar_lambda2[2], 'eigenVectorMinVar_lambda3[0]':eigenVectorMinVar_lambda3[0], 'eigenVectorMinVar_lambda3[1]':eigenVectorMinVar_lambda3[1], 'eigenVectorMinVar_lambda3[2]':eigenVectorMinVar_lambda3[2]}

        # Append detailed_info_dict to FR_detailed_info_DF.
        if not (detailed_info_dict is None):
            eventList_DF_detailedInfo = eventList_DF_detailedInfo.append(detailed_info_dict, ignore_index=True)

    # Save DataFrame to pickle file.
    print('\nSaving eventList_DF_detailedInfo to pickle file...')
    # If plotFolder does not exist, create it.
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    eventList_DF_detailedInfo.to_pickle(output_dir + '/' + output_filename + '.p')
    print('Done.')
    
    return eventList_DF_detailedInfo

###############################################################################

def plot_time_series_data(data_DF, spacecraftID, output_dir, **kwargs):
    # Check keywords.
    # Get flux rope label info.
    if 'fluxRopeList_DF' in kwargs:
        isLabelFluxRope = True
        fluxRopeList_DF = kwargs['fluxRopeList_DF']
    else:
        isLabelFluxRope = False
    # Get shock label info.
    if 'shockTimeList' in kwargs:
        shockTimeList = kwargs['shockTimeList']
        if shockTimeList: # If not empty.
            isLabelShock = True
        else:
            isLabelShock = False
    else: 
        isLabelShock = False
    # Check if pitch angle data exist.
    if 'pithcAngle_DF' in kwargs:
        data_pitchAngle_DF = kwargs['pithcAngle_DF']
        if data_pitchAngle_DF is not None:
            isPlotPitchAngle = True
        else: 
            isPlotPitchAngle = False
    else:
        isPlotPitchAngle = False
    # Check x interval setting
    fig_x_interval = 1
    if 'fig_x_interval' in kwargs:
        fig_x_interval = kwargs['fig_x_interval']
    
    # Plot function cannot handle the all NaN array.
    for column in data_DF:
        if data_DF[column].isnull().all():
            data_DF[column].fillna(value=0, inplace=True)
        
    # Make plots.
    # Physics constants.
    mu0 = 4.0 * np.pi * 1e-7 #(N/A^2) magnetic constant permeability of free space vacuum permeability.
    m_proton = 1.6726219e-27 # Proton mass. In kg.
    factor_deg2rad = np.pi/180.0 # Convert degree to rad.
    k_Boltzmann = 1.3806488e-23 # Boltzmann constant, in J/K.
    # Set plot format defination.
    #fig_formatter = mdates.DateFormatter('%m/%d %H:%M') # Full format is ('%Y-%m-%d %H:%M:%S').
    fig_formatter = mdates.DateFormatter('%H:%M') # Full format is ('%Y-%m-%d %H:%M:%S').
    fig_hour_locator = dates.HourLocator(interval=fig_x_interval)
    fig_title_fontsize = 11
    fig_ylabel_fontsize = 8
    fig_ytick_fontsize = 7
    fig_xtick_fontsize = 7
    fig_linewidth=0.5

    if spacecraftID=='WIND':
        # Get range start and end time from data_DF.
        rangeStart = data_DF.index[0]
        rangeEnd = data_DF.index[-1]
        
        # Create Figure Title from date.
        rangeStart_str = rangeStart.strftime('%Y/%m/%d %H:%M')
        rangeEnd_str = rangeEnd.strftime('%Y/%m/%d %H:%M')
        figureTitle = 'Flux Rope Case '+ rangeStart_str + ' ~ ' + rangeEnd_str + ' (WIND)'
        plotFileName = 'WIND_'+ rangeStart.strftime('%Y%m%d%H%M') + '_' + rangeEnd.strftime('%Y%m%d%H%M')

        # Create multi-plot fig.
        if isPlotPitchAngle:
            fig, ax = plt.subplots(5, 1, sharex=True,figsize=(9, 5.5))
            Bfield = ax[0]
            Vsw = ax[1]
            Np_ratio = ax[2]
            Tp_beta = ax[3]
            e_pitch = ax[4]
        else:
            fig, ax = plt.subplots(4, 1, sharex=True,figsize=(9, 4.4))
            Bfield = ax[0]
            Vsw = ax[1]
            Np_ratio = ax[2]
            Tp_beta = ax[3]

        # 1) Plot WIN magnetic field.
        # Get B data.
        Bfield.xaxis.set_major_formatter(fig_formatter) # Set xlabel format.
        Bfield.set_title(figureTitle, fontsize = fig_title_fontsize) # All subplots will share this title.
        Bfield.set_ylabel(r'$B$ (nT)',fontsize=fig_ylabel_fontsize) # Label font size.
        Bfield.tick_params(axis='y', labelsize=fig_ytick_fontsize) # Tick font size.
        Bfield.yaxis.set_major_locator(MaxNLocator(4))
        Bfield.plot(data_DF.index,data_DF['Bx'],'-r',linewidth=fig_linewidth,label='Bx')
        Bfield.plot(data_DF.index,data_DF['By'],'-g',linewidth=fig_linewidth,label='By')
        Bfield.plot(data_DF.index,data_DF['Bz'],'-b',linewidth=fig_linewidth,label='Bz')
        WIN_Btotal = (data_DF['Bx']**2 + data_DF['By']**2 + data_DF['Bz']**2)**0.5
        Bfield.plot(data_DF.index,WIN_Btotal,color='black',linewidth=fig_linewidth)
        Bfield.axhline(0, color='black',linewidth=0.5,linestyle='dashed') # Zero line, must placed after data plot
        # Only plot color span do not label flux rope in this panel.
        if isLabelFluxRope:
            for index, oneFluxRopeRecord in fluxRopeList_DF.iterrows():
                startTime_temp = oneFluxRopeRecord['startTime']
                endTime_temp = oneFluxRopeRecord['endTime']
                # Plot color span.
                Bfield.axvspan(startTime_temp, endTime_temp, color='y', alpha=0.2, lw=0)
                # Plot boundary line.
                Bfield.axvline(startTime_temp, color='black', linewidth=0.2)
                Bfield.axvline(endTime_temp, color='black', linewidth=0.2)
        # Indicate shock Time. No label.
        if isLabelShock:
            for shockTime in shockTimeList:
                Bfield.axvline(shockTime, color='black', linewidth=1, linestyle='dashed')# Shock Time.
        Bfield.legend(loc='center left',prop={'size':5}, bbox_to_anchor=(1, 0.5))

        # 2) Plot WIND solar wind bulk speed
        Vsw.set_ylabel(r'$V_{sw}$ (km/s)', fontsize=fig_ylabel_fontsize)
        Vsw.yaxis.set_major_locator(MaxNLocator(4))
        Vsw.tick_params(axis='y', labelsize=fig_ytick_fontsize) # Tick font size.
        WIN_Vtotal = (data_DF['Vx']**2 + data_DF['Vz']**2 + data_DF['Vz']**2)**0.5
        Vsw.plot(data_DF.index,WIN_Vtotal,linewidth=fig_linewidth,color='b')
        # Plot color span and label flux rope in this panel.
        if isLabelFluxRope:
            # Get y range, used for setting label position.
            y_min, y_max = Vsw.axes.get_ylim()
            for index, oneFluxRopeRecord in fluxRopeList_DF.iterrows():
                startTime_temp = oneFluxRopeRecord['startTime']
                endTime_temp = oneFluxRopeRecord['endTime']
                # Set label horizontal(x) position.
                label_position_x = startTime_temp+(endTime_temp-startTime_temp)/2
                # Set label verticle(y) position.
                if not (index+1)%2: # Odd number.
                    label_position_y = y_min + (y_max - y_min)/4.0 # One quater of the y range above y_min.
                else: # Even number.
                    label_position_y = y_min + 3*(y_max - y_min)/4.0 # One quater of the y range above y_min.
                # Set label text.
                label_text = str(index+1)
                # Plot color span.
                Vsw.axvspan(startTime_temp, endTime_temp, color='y', alpha=0.2, lw=0)
                # Plot boundary line.
                Vsw.axvline(startTime_temp, color='black', linewidth=0.2)
                Vsw.axvline(endTime_temp, color='black', linewidth=0.2)
                # Place label.
                Vsw.text(label_position_x, label_position_y, label_text, fontsize = 8, horizontalalignment='center', verticalalignment='center',bbox={'boxstyle':'round','pad':0.2, 'edgecolor':'None', 'facecolor':'white', 'alpha':0.6})
                Vsw.text(label_position_x, label_position_y, label_text, fontsize = 8, horizontalalignment='center', verticalalignment='center',bbox={'boxstyle':'round','pad':0.2, 'edgecolor':'black', 'facecolor':'None'})
        # Indicate shock Time. And label it.
        if isLabelShock:
            # Get y range, used for setting label position.
            y_min, y_max = Vsw.axes.get_ylim()
            for shockTime in shockTimeList:
                # Set label horizontal(x) position.
                label_position_x = shockTime
                # Set label verticle(y) position.
                label_position_y = y_min + (y_max - y_min)/2.0 # Half of the y range above y_min.
                # Set label text.
                shockTime_str = shockTime.strftime('%H:%M')
                label_text = 'SHOCK\n'+shockTime_str
                Vsw.axvline(shockTime, color='black', linewidth=1, linestyle='dashed')# Shock Time.
                Vsw.text(label_position_x, label_position_y, label_text, fontsize = 7,horizontalalignment='center', verticalalignment='center',bbox={'boxstyle':'round','pad':0.2, 'edgecolor':'None', 'facecolor':'white', 'alpha':0.6})
                Vsw.text(label_position_x, label_position_y, label_text, fontsize = 7,horizontalalignment='center', verticalalignment='center',bbox={'boxstyle':'round','pad':0.2, 'edgecolor':'black', 'facecolor':'None'})
        # 3) Plot WIN Proton number density, and alpha/proton ratio.
        ratio = np.divide(data_DF['N_alpha'], data_DF['Np'])
        Np_ratio.set_ylabel(r'$N_{p}$ (#/cc)', fontsize=fig_ylabel_fontsize)
        Np_ratio.yaxis.set_major_locator(MaxNLocator(4))
        Np_ratio.tick_params(axis='y', labelsize=fig_ytick_fontsize) # Tick font size.
        Np_ratio.set_ylim([0,30])
        Np_ratio.plot(data_DF.index,data_DF['Np'],linewidth=fig_linewidth,color='b')
        # Only plot color span do not label flux rope in this panel.
        if isLabelFluxRope:
            for index, oneFluxRopeRecord in fluxRopeList_DF.iterrows():
                startTime_temp = oneFluxRopeRecord['startTime']
                endTime_temp = oneFluxRopeRecord['endTime']
                # Plot color span.
                Np_ratio.axvspan(startTime_temp, endTime_temp, color='y', alpha=0.2, lw=0)
                Np_ratio.axvline(startTime_temp, color='black', linewidth=0.2)
                Np_ratio.axvline(endTime_temp, color='black', linewidth=0.2)
        # Indicate shock Time. No label.
        if isLabelShock:
            for shockTime in shockTimeList:
                Np_ratio.axvline(shockTime, color='black', linewidth=1, linestyle='dashed')# Shock Time.
        Np_ratio_twin = Np_ratio.twinx()
        Np_ratio_twin.set_ylim([0,0.1])
        Np_ratio_twin.set_ylabel(r'$N_{\alpha}/N_p$', color='g', fontsize=fig_ylabel_fontsize) # Label font size.
        Np_ratio_twin.yaxis.set_major_locator(MaxNLocator(4))
        Np_ratio_twin.tick_params(axis='y', labelsize=fig_ytick_fontsize) # Tick font size.
        #print(data_DF.index)
        #print(ratio)
        Np_ratio_twin.plot(data_DF.index, ratio, color='g', linewidth=fig_linewidth)
        for ticklabel in Np_ratio_twin.get_yticklabels(): # Set label color to green
            ticklabel.set_color('g')

        # 4) Plot WIN Proton temperature, and plasma beta.
        Beta_p = data_DF['Np']*1e6*k_Boltzmann*data_DF['Tp']/(np.square(WIN_Btotal*1e-9)/(2.0*mu0))
        Tp_beta.set_ylabel(r'$T_{p}$ ($10^6$K)', fontsize=fig_ylabel_fontsize)
        Tp_beta.yaxis.set_major_locator(MaxNLocator(4))
        Tp_beta.tick_params(axis='y', labelsize=fig_ytick_fontsize) # Tick font size.
        Tp_inMK = data_DF['Tp']/1e6
        Tp_beta.set_ylim([0,0.28])
        Tp_beta.plot(Tp_inMK.index, Tp_inMK, linewidth=fig_linewidth,color='b')
        # Only plot color span do not label flux rope in this panel.
        if isLabelFluxRope:
            for index, oneFluxRopeRecord in fluxRopeList_DF.iterrows():
                startTime_temp = oneFluxRopeRecord['startTime']
                endTime_temp = oneFluxRopeRecord['endTime']
                # Plot color span.
                Tp_beta.axvspan(startTime_temp, endTime_temp, color='y', alpha=0.2, lw=0)
                Tp_beta.axvline(startTime_temp, color='black', linewidth=0.2)
                Tp_beta.axvline(endTime_temp, color='black', linewidth=0.2)
        # Indicate shock Time. No label.
        if isLabelShock:
            for shockTime in shockTimeList:
                Tp_beta.axvline(shockTime, color='black', linewidth=1, linestyle='dashed')# Shock Time.
        # Set double x axis for beta.
        Tp_beta_twin = Tp_beta.twinx()
        Tp_beta_twin.set_ylabel(r'$\beta_p$',color='g',fontsize=fig_ylabel_fontsize) # Label font size.
        Tp_beta_twin.yaxis.set_major_locator(MaxNLocator(5))
        Tp_beta_twin.tick_params(axis='y', labelsize=fig_ytick_fontsize) # Tick font size.
        Tp_beta_twin.plot(Beta_p.index, Beta_p, color='g', linewidth=fig_linewidth)
        Tp_beta_twin.set_ylim([0,2.2]) #Bata.
        for ticklabel in Tp_beta_twin.get_yticklabels(): # Set label color to red
            ticklabel.set_color('g')
        
        if not isPlotPitchAngle:
            # This is a shared axis for all subplot.
            Tp_beta.xaxis.set_major_locator(fig_hour_locator)            
            Tp_beta.tick_params(axis='x', which='major', labelsize=fig_xtick_fontsize)
        else:
            # Plot PitchAngle:
            print('Plotting suprathermal electron pitch angle distribution...')
            # Check year.
            currentYear = data_pitchAngle_DF.index[0].year
            if currentYear in range(1996, 2002):
                energy_label = '94.02eV'
            elif currentYear in range(2002, 2017):
                energy_label = '96.71eV'
            else:
                energy_label = ' '
            # Slice data.
            startTime_temp = data_DF.index[0]
            endTime_temp = data_DF.index[-1]
            selectedRange_mask_temp = (data_pitchAngle_DF.index >= startTime_temp) & (data_pitchAngle_DF.index <= endTime_temp)
            data_pitchAngle_DF = data_pitchAngle_DF.iloc[selectedRange_mask_temp]
            # Set label format and locator.
            e_pitch.xaxis.set_major_formatter(fig_formatter) # Set xlabel format.
            e_pitch.yaxis.set_major_locator(MaxNLocator(3))
            #e_pitch.legend(loc='upper right',prop={'size':7})
            #e_pitch.set_xlabel('Time',fontsize=fig_xlabel_fontsize)
            e_pitch.set_ylabel(r'$E$ ' + r'$($' + energy_label + r'$)$' + '\n' + r'$pitch$ ' + r'$angle$ (deg)',fontsize=fig_ylabel_fontsize) # Label font size.
            e_pitch.set_ylim([0,180])
            e_pitch.tick_params(axis='y', which='major', labelsize=fig_ytick_fontsize) # Tick font size.
            # Plot 2D data.
            ax_e_pitch = e_pitch.pcolormesh(data_pitchAngle_DF.index, data_pitchAngle_DF.columns.values, data_pitchAngle_DF.values.transpose(), cmap='jet', norm=matplotlib.colors.LogNorm(vmin=1e-31, vmax=1e-26))
            #plt.subplots_adjust(left=left_margin_coord, right=right_margin_coord, top=top_margin_coord, bottom=bottom_margin_coord)
            
            '''
            left_margin = 0.065    # the left side of the subplots of the figure
            right_margin = 0.935   # the right side of the subplots of the figure
            top_margin = 0.95     # the top of the subplots of the figure
            bottom_margin = 0.05  # the bottom of the subplots of the figure
            width_gap = None       # the amount of width reserved for blank space between subplots
            height_gap = 0.5       # the amount of height reserved for white space between subplots
            plt.subplots_adjust(left=left_margin, bottom=bottom_margin, right=right_margin, top=top_margin, wspace=width_gap, hspace=height_gap)
            '''

            # Plot color bar.
            box = e_pitch.get_position() # Get pannel position.
            pad, width = 0.005, 0.005 # pad = distance to panel, width = colorbar width.
            cax = fig.add_axes([box.xmax + pad, box.ymin, width, box.height]) # Set colorbar position.
            ax_e_pitch_cbar = fig.colorbar(ax_e_pitch, ax=e_pitch, cax=cax)
            #ax_e_pitch_cbar.set_label('#/[cc*(cm/s)^3]', rotation=270, fontsize=fig_ytick_fontsize)
            ax_e_pitch_cbar.ax.minorticks_on()
            ax_e_pitch_cbar.ax.tick_params(labelsize=fig_ytick_fontsize)
            ax_e_pitch_cbar.outline.set_linewidth(fig_linewidth)
            
            # This is a shared axis for all subplot
            e_pitch.xaxis.set_major_locator(fig_hour_locator)
            e_pitch.tick_params(axis='x', which='major', labelsize=fig_xtick_fontsize)
            
            # Only plot color span do not label flux rope in this panel.
            if isLabelFluxRope:
                for index, oneFluxRopeRecord in fluxRopeList_DF.iterrows():
                    startTime_temp = oneFluxRopeRecord['startTime']
                    endTime_temp = oneFluxRopeRecord['endTime']
                    # Plot color span.
                    #e_pitch.axvspan(startTime_temp, endTime_temp, color='y', alpha=0.2, lw=0)
                    e_pitch.axvline(startTime_temp, color='black', linestyle=':', linewidth=fig_linewidth) # Shading region.
                    e_pitch.axvline(endTime_temp, color='black', linestyle=':', linewidth=fig_linewidth) # Shading region.
            # Indicate shock Time. No label.
            if isLabelShock:
                for shockTime in shockTimeList:
                    e_pitch.axvline(shockTime, color='black', linewidth=1, linestyle='dashed')# Shock Time.
        
         
        # Save to two places.
        print('\nSaving plot: {}...'.format(plotFileName))
        fig.savefig(output_dir + '/' + plotFileName + '.png', format='png', dpi=300, bbox_inches='tight')
        #fig.savefig(single_plot_dir + '/' + plotFileName + '.png', format='png', dpi=300, bbox_inches='tight')
        saved_filename = output_dir + '/' + plotFileName + '.png'
        print('Done.')

    elif spacecraftID=='ACE':
        # Get range start and end time from data_DF.
        rangeStart = data_DF.index[0]
        rangeEnd = data_DF.index[-1]
        
        # Create Figure Title from date.
        rangeStart_str = rangeStart.strftime('%Y/%m/%d %H:%M')
        rangeEnd_str = rangeEnd.strftime('%Y/%m/%d %H:%M')
        figureTitle = 'Flux Rope Case '+ rangeStart_str + ' ~ ' + rangeEnd_str + ' (ACE)'
        plotFileName = 'ACE_'+ rangeStart.strftime('%Y%m%d%H%M') + '_' + rangeEnd.strftime('%Y%m%d%H%M')

        # Create multi-plot fig.
        if isLabelShock:
            fig, ax = plt.subplots(6, 1, sharex=True,figsize=(9, 6.6))
            Bfield = ax[0]
            Vsw = ax[1]
            Np_ratio = ax[2]
            Tp_beta = ax[3]
            LEMS120 = ax[4]
            LEMS120_norm = ax[5]
        else:
            fig, ax = plt.subplots(5, 1, sharex=True,figsize=(9, 5.5))
            Bfield = ax[0]
            Vsw = ax[1]
            Np_ratio = ax[2]
            Tp_beta = ax[3]
            LEMS120 = ax[4]

        # 1) Plot ACE magnetic field.
        # Get B data.
        Bfield.xaxis.set_major_formatter(fig_formatter) # Set xlabel format.
        Bfield.set_title(figureTitle, fontsize = 12) # All subplots will share this title.
        Bfield.set_ylabel(r'$B$ (nT)',fontsize=fig_ylabel_fontsize) # Label font size.
        Bfield.tick_params(axis='y', labelsize=fig_ytick_fontsize) # Tick font size.
        Bfield.yaxis.set_major_locator(MaxNLocator(4))
        Bfield.plot(data_DF.index,data_DF['Bx'],'-r',linewidth=fig_linewidth,label='Bx')
        Bfield.plot(data_DF.index,data_DF['By'],'-g',linewidth=fig_linewidth,label='By')
        Bfield.plot(data_DF.index,data_DF['Bz'],'-b',linewidth=fig_linewidth,label='Bz')
        WIN_Btotal = (data_DF['Bx']**2 + data_DF['By']**2 + data_DF['Bz']**2)**0.5
        Bfield.plot(data_DF.index,WIN_Btotal,color='black',linewidth=fig_linewidth)
        Bfield.axhline(0, color='black',linewidth=0.5,linestyle='dashed') # Zero line, must placed after data plot
        # Only plot color span do not label flux rope in this panel.
        if isLabelFluxRope:
            for index, oneFluxRopeRecord in fluxRopeList_DF.iterrows():
                startTime_temp = oneFluxRopeRecord['startTime']
                endTime_temp = oneFluxRopeRecord['endTime']
                # Plot color span.
                Bfield.axvspan(startTime_temp, endTime_temp, color='y', alpha=0.2, lw=0)
                Bfield.axvline(startTime_temp, color='black', linewidth=0.2)
                Bfield.axvline(endTime_temp, color='black', linewidth=0.2)
        # Indicate shock Time. No label.
        if isLabelShock:
            for shockTime in shockTimeList:
                Bfield.axvline(shockTime, color='black', linewidth=1, linestyle='dashed')# Shock Time.
        Bfield.legend(loc='center left',prop={'size':5}, bbox_to_anchor=(1, 0.5))

        # 2) Plot ACE solar wind bulk speed
        Vsw.set_ylabel(r'$V_{sw}$ (km/s)', fontsize=fig_ylabel_fontsize)
        Vsw.yaxis.set_major_locator(MaxNLocator(4))
        Vsw.tick_params(axis='y', labelsize=fig_ytick_fontsize) # Tick font size.
        WIN_Vtotal = (data_DF['Vx']**2 + data_DF['Vz']**2 + data_DF['Vz']**2)**0.5
        Vsw.plot(data_DF.index,WIN_Vtotal,linewidth=fig_linewidth,color='b')
        # Plot color span and label flux rope in this panel.
        if isLabelFluxRope:
            # Get y range, used for setting label position.
            y_min, y_max = Vsw.axes.get_ylim()
            for index, oneFluxRopeRecord in fluxRopeList_DF.iterrows():
                startTime_temp = oneFluxRopeRecord['startTime']
                endTime_temp = oneFluxRopeRecord['endTime']
                # Set label horizontal(x) position.
                label_position_x = startTime_temp+(endTime_temp-startTime_temp)/2
                # Set label verticle(y) position.
                if not (index+1)%2: # Odd number.
                    label_position_y = y_min + (y_max - y_min)/4.0 # One quater of the y range above y_min.
                else: # Even number.
                    label_position_y = y_min + 3*(y_max - y_min)/4.0 # One quater of the y range above y_min.
                # Set label text.
                label_text = str(index+1)
                # Plot color span.
                Vsw.axvspan(startTime_temp, endTime_temp, color='y', alpha=0.2, lw=0)
                Vsw.axvline(startTime_temp, color='black', linewidth=0.2)
                Vsw.axvline(endTime_temp, color='black', linewidth=0.2)
                # Place label.
                Vsw.text(label_position_x, label_position_y, label_text, fontsize = 8, horizontalalignment='center', verticalalignment='center',bbox={'boxstyle':'round','pad':0.2, 'edgecolor':'None', 'facecolor':'white', 'alpha':0.6})
                Vsw.text(label_position_x, label_position_y, label_text, fontsize = 8, horizontalalignment='center', verticalalignment='center',bbox={'boxstyle':'round','pad':0.2, 'edgecolor':'black', 'facecolor':'None'})
        # Indicate shock Time. And label it.
        if isLabelShock:
            # Get y range, used for setting label position.
            y_min, y_max = Vsw.axes.get_ylim()
            for shockTime in shockTimeList:
                # Set label horizontal(x) position.
                label_position_x = shockTime
                # Set label verticle(y) position.
                label_position_y = y_min + (y_max - y_min)/2.0 # Half of the y range above y_min.
                # Set label text.
                shockTime_str = shockTime.strftime('%H:%M')
                label_text = 'SHOCK\n'+shockTime_str
                Vsw.axvline(shockTime, color='black', linewidth=1, linestyle='dashed')# Shock Time.
                Vsw.text(label_position_x, label_position_y, label_text, fontsize = 7,horizontalalignment='center', verticalalignment='center',bbox={'boxstyle':'round','pad':0.2, 'edgecolor':'None', 'facecolor':'white', 'alpha':0.6})
                Vsw.text(label_position_x, label_position_y, label_text, fontsize = 7,horizontalalignment='center', verticalalignment='center',bbox={'boxstyle':'round','pad':0.2, 'edgecolor':'black', 'facecolor':'None'})
        # 3) Plot ACE Proton number density, and alpha/proton ratio.
        Np_ratio.set_ylabel(r'$N_{p}$ (#/cc)', fontsize=fig_ylabel_fontsize)
        Np_ratio.yaxis.set_major_locator(MaxNLocator(4))
        Np_ratio.tick_params(axis='y', labelsize=fig_ytick_fontsize) # Tick font size.
        Np_ratio.set_ylim([0,30])
        Np_ratio.plot(data_DF.index,data_DF['Np'],linewidth=fig_linewidth,color='b')
        # Only plot color span do not label flux rope in this panel.
        if isLabelFluxRope:
            for index, oneFluxRopeRecord in fluxRopeList_DF.iterrows():
                startTime_temp = oneFluxRopeRecord['startTime']
                endTime_temp = oneFluxRopeRecord['endTime']
                # Plot color span.
                Np_ratio.axvspan(startTime_temp, endTime_temp, color='y', alpha=0.2, lw=0)
                Np_ratio.axvline(startTime_temp, color='black', linewidth=0.2)
                Np_ratio.axvline(endTime_temp, color='black', linewidth=0.2)
        # Indicate shock Time. No label.
        if isLabelShock:
            for shockTime in shockTimeList:
                Np_ratio.axvline(shockTime, color='black', linewidth=1, linestyle='dashed')# Shock Time.
        Np_ratio_twin = Np_ratio.twinx()
        Np_ratio_twin.set_ylim([0,0.1])
        Np_ratio_twin.set_ylabel(r'$N_{\alpha}/N_p$', color='g', fontsize=fig_ylabel_fontsize) # Label font size.
        Np_ratio_twin.yaxis.set_major_locator(MaxNLocator(4))
        Np_ratio_twin.tick_params(axis='y', labelsize=fig_ytick_fontsize) # Tick font size.
        Np_ratio_twin.plot(data_DF.index,data_DF['Alpha2Proton_ratio'], color='g', linewidth=fig_linewidth)
        for ticklabel in Np_ratio_twin.get_yticklabels(): # Set label color to green
            ticklabel.set_color('g')

        # 4) Plot ACE Proton temperature, and plasma beta.
        Beta_p = data_DF['Np']*1e6*k_Boltzmann*data_DF['Tp']/(np.square(WIN_Btotal*1e-9)/(2.0*mu0))
        Tp_beta.set_ylabel(r'$T_{p}$ ($10^6$K)', fontsize=fig_ylabel_fontsize)
        Tp_beta.yaxis.set_major_locator(MaxNLocator(4))
        Tp_beta.tick_params(axis='y', labelsize=fig_ytick_fontsize) # Tick font size.
        Tp_inMK = data_DF['Tp']/1e6
        Tp_beta.set_ylim([0,0.28])
        Tp_beta.plot(Tp_inMK.index, Tp_inMK, linewidth=fig_linewidth,color='b')
        # Only plot color span do not label flux rope in this panel.
        if isLabelFluxRope:
            for index, oneFluxRopeRecord in fluxRopeList_DF.iterrows():
                startTime_temp = oneFluxRopeRecord['startTime']
                endTime_temp = oneFluxRopeRecord['endTime']
                # Plot color span.
                Tp_beta.axvspan(startTime_temp, endTime_temp, color='y', alpha=0.2, lw=0)
                Tp_beta.axvline(startTime_temp, color='black', linewidth=0.2)
                Tp_beta.axvline(endTime_temp, color='black', linewidth=0.2)
        # Indicate shock Time. No label.
        if isLabelShock:
            for shockTime in shockTimeList:
                Tp_beta.axvline(shockTime, color='black', linewidth=1, linestyle='dashed')# Shock Time.
        # Set double x axis for beta.
        Tp_beta_twin = Tp_beta.twinx()
        Tp_beta_twin.set_ylabel(r'$\beta_p$',color='g',fontsize=fig_ylabel_fontsize) # Label font size.
        Tp_beta_twin.yaxis.set_major_locator(MaxNLocator(5))
        Tp_beta_twin.tick_params(axis='y', labelsize=fig_ytick_fontsize) # Tick font size.
        Tp_beta_twin.plot(Beta_p.index, Beta_p, color='g', linewidth=fig_linewidth)
        Tp_beta_twin.set_ylim([0,2.2]) #Bata.
        for ticklabel in Tp_beta_twin.get_yticklabels(): # Set label color to red
            ticklabel.set_color('g')
        
        # 5) Plot LEMS120 data.
        LEMS120.xaxis.set_major_formatter(fig_formatter) # Set xlabel format.
        LEMS120.set_ylabel(r'ions ($10^4$#/sec)',fontsize=fig_ylabel_fontsize) # Label font size.
        LEMS120.tick_params(axis='y', labelsize=fig_ytick_fontsize) # Tick font size.
        LEMS120.yaxis.set_major_locator(MaxNLocator(4))
        colors = {'1':'midnightblue', '2':'steelblue', '3':'darkorange', '4':'gold', '5':'darkorchid', '6':'mediumseagreen', '7':'c', '8':'maroon'}
        LEMS120.plot(data_DF.index,data_DF['LEMS120_P1']/10e4,color=colors['1'],linewidth=fig_linewidth,label='0.047-0.068')
        LEMS120.plot(data_DF.index,data_DF['LEMS120_P2']/10e4,color=colors['2'],linewidth=fig_linewidth,label='0.068-0.115')
        LEMS120.plot(data_DF.index,data_DF['LEMS120_P3']/10e4,color=colors['3'],linewidth=fig_linewidth,label='0.115-0.195')
        LEMS120.plot(data_DF.index,data_DF['LEMS120_P4']/10e4,color=colors['4'],linewidth=fig_linewidth,label='0.195-0.321')
        LEMS120.plot(data_DF.index,data_DF['LEMS120_P5']/10e4,color=colors['5'],linewidth=fig_linewidth,label='0.321-0.580')
        LEMS120.plot(data_DF.index,data_DF['LEMS120_P6']/10e4,color=colors['6'],linewidth=fig_linewidth,label='0.580-1.06')
        LEMS120.plot(data_DF.index,data_DF['LEMS120_P7']/10e4,color=colors['7'],linewidth=fig_linewidth,label='1.06-1.90')
        LEMS120.plot(data_DF.index,data_DF['LEMS120_P8']/10e4,color=colors['8'],linewidth=fig_linewidth,label='1.90-4.80 (MeV)')
        # Only plot color span do not label flux rope in this panel.
        if isLabelFluxRope:
            for index, oneFluxRopeRecord in fluxRopeList_DF.iterrows():
                startTime_temp = oneFluxRopeRecord['startTime']
                endTime_temp = oneFluxRopeRecord['endTime']
                # Plot color span.
                LEMS120.axvspan(startTime_temp, endTime_temp, color='y', alpha=0.2, lw=0)
                LEMS120.axvline(startTime_temp, color='black', linewidth=0.2)
                LEMS120.axvline(endTime_temp, color='black', linewidth=0.2)
        # Indicate shock Time. No label.
        if isLabelShock:
            for shockTime in shockTimeList:
                LEMS120.axvline(shockTime, color='black', linewidth=1, linestyle='dashed')# Shock Time.
        LEMS120.legend(loc='center left',prop={'size':4}, bbox_to_anchor=(1, 0.5))
        
        # 5) Plot LEMS120(normalized) data.
        if isLabelShock:
            firstShockTime = shockTimeList[0]
            norm_factor_P1 = data_DF.loc[firstShockTime, 'LEMS120_P1']
            norm_factor_P2 = data_DF.loc[firstShockTime, 'LEMS120_P2']
            norm_factor_P3 = data_DF.loc[firstShockTime, 'LEMS120_P3']
            norm_factor_P4 = data_DF.loc[firstShockTime, 'LEMS120_P4']
            norm_factor_P5 = data_DF.loc[firstShockTime, 'LEMS120_P5']
            norm_factor_P6 = data_DF.loc[firstShockTime, 'LEMS120_P6']
            norm_factor_P7 = data_DF.loc[firstShockTime, 'LEMS120_P7']
            norm_factor_P8 = data_DF.loc[firstShockTime, 'LEMS120_P8']
            
            LEMS120_norm.xaxis.set_major_formatter(fig_formatter) # Set xlabel format.
            LEMS120_norm.set_ylabel(r'Amplification',fontsize=fig_ylabel_fontsize) # Label font size.
            LEMS120_norm.tick_params(axis='y', labelsize=fig_ytick_fontsize) # Tick font size.
            LEMS120_norm.yaxis.set_major_locator(MaxNLocator(4))
            colors = {'1':'midnightblue', '2':'steelblue', '3':'darkorange', '4':'gold', '5':'darkorchid', '6':'mediumseagreen', '7':'c', '8':'maroon'}
            LEMS120_norm.plot(data_DF.index,data_DF['LEMS120_P1']/norm_factor_P1,color=colors['1'],linewidth=fig_linewidth,label='0.047-0.068')
            LEMS120_norm.plot(data_DF.index,data_DF['LEMS120_P2']/norm_factor_P2,color=colors['2'],linewidth=fig_linewidth,label='0.068-0.115')
            LEMS120_norm.plot(data_DF.index,data_DF['LEMS120_P3']/norm_factor_P3,color=colors['3'],linewidth=fig_linewidth,label='0.115-0.195')
            LEMS120_norm.plot(data_DF.index,data_DF['LEMS120_P4']/norm_factor_P4,color=colors['4'],linewidth=fig_linewidth,label='0.195-0.321')
            LEMS120_norm.plot(data_DF.index,data_DF['LEMS120_P5']/norm_factor_P5,color=colors['5'],linewidth=fig_linewidth,label='0.321-0.580')
            LEMS120_norm.plot(data_DF.index,data_DF['LEMS120_P6']/norm_factor_P6,color=colors['6'],linewidth=fig_linewidth,label='0.580-1.06')
            LEMS120_norm.plot(data_DF.index,data_DF['LEMS120_P7']/norm_factor_P7,color=colors['7'],linewidth=fig_linewidth,label='1.06-1.90')
            LEMS120_norm.plot(data_DF.index,data_DF['LEMS120_P8']/norm_factor_P8,color=colors['8'],linewidth=fig_linewidth,label='1.90-4.80 (MeV)')
            # Only plot color span do not label flux rope in this panel.
            if isLabelFluxRope:
                for index, oneFluxRopeRecord in fluxRopeList_DF.iterrows():
                    startTime_temp = oneFluxRopeRecord['startTime']
                    endTime_temp = oneFluxRopeRecord['endTime']
                    # Plot color span.
                    LEMS120_norm.axvspan(startTime_temp, endTime_temp, color='y', alpha=0.2, lw=0)
                    LEMS120_norm.axvline(startTime_temp, color='black', linewidth=0.2)
                    LEMS120_norm.axvline(endTime_temp, color='black', linewidth=0.2)
            # Indicate shock Time. No label.
            for shockTime in shockTimeList:
                LEMS120_norm.axvline(shockTime, color='black', linewidth=1, linestyle='dashed')# Shock Time.
            LEMS120_norm.legend(loc='center left',prop={'size':4}, bbox_to_anchor=(1, 0.5))
            
            # Shared x-axis for all panels.
            LEMS120_norm.xaxis.set_major_locator(fig_hour_locator)
            LEMS120_norm.tick_params(axis='x', which='major', labelsize=fig_xtick_fontsize) # This is a shared axis for all subplot
        else:
            # Shared x-axis for all panels.
            LEMS120.xaxis.set_major_locator(fig_hour_locator)
            LEMS120.tick_params(axis='x', which='major', labelsize=fig_xtick_fontsize) # This is a shared axis for all subplot
        
        # Save to two place.
        print('\nSaving plot: {}...'.format(plotFileName))
        fig.savefig(output_dir + '/' + plotFileName + '.png', format='png', dpi=300, bbox_inches='tight')
        #fig.savefig(single_plot_dir + '/' + plotFileName + '.png', format='png', dpi=300, bbox_inches='tight')
        saved_filename = output_dir + '/' + plotFileName + '.png'
        print('Done.')
    
    else:
        print('\nPlease specify the correct spacecraft ID!')
        return False
        
    return saved_filename

###############################################################################

# Stack two images verticlly.
def vstack_images(file1, file2):
    """Merge two images into one, displayed side by side
    :param file1: path to first image file
    :param file2: path to second image file
    :return: the merged Image object
    """
    image1 = Image.open(file1)
    image2 = Image.open(file2)

    (width1, height1) = image1.size
    (width2, height2) = image2.size

    result_width = max(width1, width2)
    result_height = height1 + height2

    result = Image.new('RGB', (result_width, result_height), color=(255,255,255))
    result.paste(im=image1, box=(0, 0))
    result.paste(im=image2, box=(0, height1))
    return result











