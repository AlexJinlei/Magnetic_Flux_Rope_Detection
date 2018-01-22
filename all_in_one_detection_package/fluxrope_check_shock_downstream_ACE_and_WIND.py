import os
import sys
import numpy as np # Scientific calculation package.
from datetime import datetime # Import datetime class from datetime package.
from datetime import timedelta
import pandas as pd
sys.path.append(os.getcwd())
import MyPythonPackage.fluxrope as FR

import matplotlib
import matplotlib.pyplot as plt # Plot package, make matlab style plots.
from matplotlib.ticker import AutoMinorLocator
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import FuncFormatter
from matplotlib.lines import Line2D
import matplotlib.dates as mdates
import matplotlib.cbook as cbook

import warnings
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")


# Terminal output format.
pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 180)

#====================================================================================

isSearch = 0
isCombineRawReslut = 1
isGetMoreInfo = 1
isLabelFluxRope = 1
isPlot = 1
isPlotCombined = 1
isSaveCsv = 1
isSpeak = 1
isVerbose = 0

# xtick interval.
fig_x_interval = 1 # In hours.

duration_range_tuple=((10,20),(20,30),(30,40),(40,50),(50,60),(60,80),(80,100),(100,120),(120,140),(140,160),(160,180))

#====================================================================================

# Load shock list.
inputDir = '/Users/jz0006/GoogleDrive/GS/GS_FluxRopeDetectionPackage/shockList/'
shockListFileName = 'IPShock_ACE_and_WIND_1998_2016_DF.p'
IPShock_ACE_and_WIND_1998_2016_DF = pd.read_pickle(inputDir + shockListFileName)

# Select range.
selectedStartTime = datetime(1999,3,9,23,40,0)
selectedEndTime = datetime(1999,3,10,9,40,59)
selected_IPShock = IPShock_ACE_and_WIND_1998_2016_DF[(IPShock_ACE_and_WIND_1998_2016_DF['ACE']>=selectedStartTime)&(IPShock_ACE_and_WIND_1998_2016_DF['ACE']<=selectedEndTime)]

# Truncate datetime in selected_IPShock, remove seconds. There is no seconds in time series data.
selected_IPShock = selected_IPShock.copy()
for index, oneShockRecord in selected_IPShock.iterrows():
    selected_IPShock.loc[index, 'ACE'] = selected_IPShock.loc[index, 'ACE'].replace(second=0, microsecond=0)
    selected_IPShock.loc[index, 'WIND'] = selected_IPShock.loc[index, 'WIND'].replace(second=0, microsecond=0)


# Plot flux rope with shock for both WIND and ACE.
for index, oneShockRecord in selected_IPShock.iterrows():
    
    # 1) Plot ACE record.
    # Specify spacecraft.
    spacecraftID = 'ACE' # 'ACE' or 'WIND'.
    
    # In DataFrame, time is in type of <class 'pandas.tslib.Timestamp'>, must convert to <type 'datetime.datetime'>.
    oneShockTime = oneShockRecord[spacecraftID].to_datetime()
    oneShockTime_str = oneShockTime.strftime('%Y%m%d%H%M')
    
    # Put in shockTimeList.
    shockTimeList = [oneShockTime]

    # Plot time range.
    datetimeStart = shockTimeList[0] - timedelta(hours=1)
    datetimeEnd = shockTimeList[0] + timedelta(hours=9)
    print('Start Time: {}'.format(datetimeStart))
    print('End   Time: {}\n'.format(datetimeEnd))
    

    # Get parameters from input datetime.
    year_start = datetimeStart.year
    month_start = datetimeStart.month
    day_start = datetimeStart.day
    hour_start = datetimeStart.hour
    minute_start = datetimeStart.minute
    year_end = datetimeEnd.year
    month_end = datetimeEnd.month
    day_end = datetimeEnd.day
    hour_end = datetimeEnd.hour
    minute_end = datetimeEnd.minute
    datetimeStart_str = datetimeStart.strftime('%Y%m%d%H%M')
    datetimeEnd_str = datetimeEnd.strftime('%Y%m%d%H%M')

    # Get current working directory.
    cwd = os.getcwd()
    # Pitch angle data path
    pitch_dir = cwd + '/pitch_angle_WIND'
    # Create case folder.
    case_dir = cwd + '/case_folder/' + spacecraftID + '_' + datetimeStart_str + '_' + datetimeEnd_str
    # csv filename.
    csv_filename = spacecraftID + '_' + datetimeStart_str + '_' + datetimeEnd_str + '.csv'
    # If case_dir folder does not exist, create it.
    if not os.path.exists(case_dir):
        os.makedirs(case_dir)
    # Create case_plot folder. # Save all plots.
    case_plot_dir = cwd + '/case_plot/'
    # If case_dir folder does not exist, create it.
    if not os.path.exists(case_dir):
        os.makedirs(case_dir)
    # Create data_cache folder.
    data_cache_dir = case_dir + '/' + spacecraftID + '_' + 'data_cache'
    # If data_cache folder does not exist, create it.
    if not os.path.exists(data_cache_dir):
        os.makedirs(data_cache_dir)
    # Create data_pickle_dir.
    data_pickle_dir = case_dir + '/' + spacecraftID + '_' + 'data_pickle'
    # If data_pickle folder does not exist, create it.
    if not os.path.exists(data_pickle_dir):
        os.makedirs(data_pickle_dir)
    # Create search_result_dir.
    search_result_dir = case_dir + '/' + spacecraftID + '_' + 'search_result'
    # If search_result folder does not exist, create it.
    if not os.path.exists(search_result_dir):
        os.makedirs(search_result_dir)
    # Create single_plot_dir.
    single_plot_dir = case_dir + '/' + spacecraftID + '_' + 'plot'
    # If plot folder does not exist, create it.
    if not os.path.exists(single_plot_dir):
        os.makedirs(single_plot_dir)

    # 1) Download data from specified spacecraft with specified start and end time.
    data_dict = FR.download_data(spacecraftID, data_cache_dir, datetimeStart, datetimeEnd)

    # 2) Preprocess data. Put all variables into DataFrame(DF).
    data_DF = FR.preprocess_data(data_dict, data_pickle_dir, isPlotFilterProcess=False)
    #print(data_DF)

    if isSearch:
        # 3) Detect flux ropes.
        # if n_theta_grid=12, d_theta_deg = 90/12 = 7.5, d_phi_deg = 360/24 = 15
        search_result_raw = FR.detect_flux_rope(data_DF, duration_range_tuple, search_result_dir, n_theta_grid=9)
        if isSpeak:
            os.system('say "ace flux rope searching process has finished"')
    if isCombineRawReslut:
        # 4) Clean up overlapped records.
        search_result_raw_filename = search_result_dir + '/search_result_raw.p'
        search_result_no_overlap_DF = FR.clean_up_raw_result(data_DF, search_result_raw_filename, walenTest_k_threshold=0.3, min_residue_diff=0.12, min_residue_fit=0.14, output_dir=search_result_dir, isPrintIntermediateDF=False, isVerbose=False)
        
        if isVerbose:
            print(search_result_no_overlap_DF)
    if isGetMoreInfo:
        # 5) Get more flux rope information.
        search_result_detail_info_DF = FR.get_more_flux_rope_info(data_DF, search_result_no_overlap_DF, output_dir=search_result_dir)
        
        if isVerbose:
            print(search_result_detail_info_DF)
            
        if isSaveCsv:
            print('Saving {}...'.format(csv_filename))
            search_result_detail_info_DF.to_csv(path_or_buf=case_plot_dir + csv_filename)
            print('Done.')

    if isPlot:
        # 6) Plot time series data for given time range. Label flux ropes.
        if isLabelFluxRope:
            fig = FR.plot_time_series_data(data_DF, spacecraftID, case_plot_dir, fluxRopeList_DF=search_result_no_overlap_DF, shockTimeList=shockTimeList, fig_x_interval=fig_x_interval)
        else:
            fig = FR.plot_time_series_data(data_DF, spacecraftID, case_plot_dir, shockTimeList=shockTimeList, fig_x_interval=fig_x_interval)

        fig1_filename = fig 
    
    
    # 2) Plot WIND record.
    # Specify spacecraft.
    spacecraftID = 'WIND' # 'ACE' or 'WIND'.
    
    # In DataFrame, time is in type of <class 'pandas.tslib.Timestamp'>, must convert to <type 'datetime.datetime'>.
    shockTime = oneShockRecord[spacecraftID].to_datetime()
    # Put in shockTimeList.
    shockTimeList = [shockTime]
    
    # Plot time range.
    datetimeStart = shockTimeList[0] - timedelta(hours=1)
    datetimeEnd = shockTimeList[0] + timedelta(hours=9)
    print('Start Time: {}'.format(datetimeStart))
    print('End   Time: {}'.format(datetimeEnd))

    # Get parameters from input datetime.
    year_start = datetimeStart.year
    month_start = datetimeStart.month
    day_start = datetimeStart.day
    hour_start = datetimeStart.hour
    minute_start = datetimeStart.minute
    year_end = datetimeEnd.year
    month_end = datetimeEnd.month
    day_end = datetimeEnd.day
    hour_end = datetimeEnd.hour
    minute_end = datetimeEnd.minute
    datetimeStart_str = datetimeStart.strftime('%Y%m%d%H%M')
    datetimeEnd_str = datetimeEnd.strftime('%Y%m%d%H%M')

    # Get current working directory.
    cwd = os.getcwd()
    # Pitch angle data path
    pitch_dir = cwd + '/pitch_angle_WIND'
    # Create case folder.
    case_dir = cwd + '/case_folder/' + spacecraftID + '_' + datetimeStart_str + '_' + datetimeEnd_str
    # csv filename.
    csv_filename = spacecraftID + '_' + datetimeStart_str + '_' + datetimeEnd_str + '.csv'
    # If case_dir folder does not exist, create it.
    if not os.path.exists(case_dir):
        os.makedirs(case_dir)
    # Create case_plot folder. # Save all plots.
    case_plot_dir = cwd + '/case_plot/'
    # If case_dir folder does not exist, create it.
    if not os.path.exists(case_dir):
        os.makedirs(case_dir)
    # Create data_cache folder.
    data_cache_dir = case_dir + '/' + spacecraftID + '_' + 'data_cache'
    # If data_cache folder does not exist, create it.
    if not os.path.exists(data_cache_dir):
        os.makedirs(data_cache_dir)
    # Create data_pickle_dir.
    data_pickle_dir = case_dir + '/' + spacecraftID + '_' + 'data_pickle'
    # If data_pickle folder does not exist, create it.
    if not os.path.exists(data_pickle_dir):
        os.makedirs(data_pickle_dir)
    # Create search_result_dir.
    search_result_dir = case_dir + '/' + spacecraftID + '_' + 'search_result'
    # If search_result folder does not exist, create it.
    if not os.path.exists(search_result_dir):
        os.makedirs(search_result_dir)
    # Create single_plot_dir.
    single_plot_dir = case_dir + '/' + spacecraftID + '_' + 'plot'
    # If plot folder does not exist, create it.
    if not os.path.exists(single_plot_dir):
        os.makedirs(single_plot_dir)

    # 1) Download data from specified spacecraft with specified start and end time.
    data_dict = FR.download_data(spacecraftID, data_cache_dir, datetimeStart, datetimeEnd)

    # 2) Preprocess data. Put all variables into DataFrame(DF).
    data_DF = FR.preprocess_data(data_dict, data_pickle_dir, isPlotFilterProcess=False)
    
    # 3) Detect flux ropes.
    if isSearch:
        # if n_theta_grid=12, d_theta_deg = 90/12 = 7.5, d_phi_deg = 360/24 = 15
        search_result_raw = FR.detect_flux_rope(data_DF, duration_range_tuple, search_result_dir, n_theta_grid=9)
        os.system('say "wind flux rope searching process has finished"')
    
    # 4) Clean up overlapped records.
    if isCombineRawReslut:        
        search_result_raw_filename = search_result_dir + '/search_result_raw.p'
        search_result_no_overlap_DF = FR.clean_up_raw_result(data_DF, search_result_raw_filename, walenTest_k_threshold=0.3, min_residue_diff=0.12, min_residue_fit=0.14, output_dir=search_result_dir, isPrintIntermediateDF=False, isVerbose=False)
        
        if isVerbose:
            print(search_result_no_overlap_DF)
    
    # 5) Get more flux rope information.
    if isGetMoreInfo:
        # 5) Get more flux rope information.
        search_result_detail_info_DF = FR.get_more_flux_rope_info(data_DF, search_result_no_overlap_DF, output_dir=search_result_dir)
        
        if isVerbose:
            print(search_result_detail_info_DF)
            
        if isSaveCsv:
            print('Saving {}...'.format(csv_filename))
            search_result_detail_info_DF.to_csv(path_or_buf=case_plot_dir + csv_filename)
            print('Done.')
    
    # 6) Plot time series data for given time range. Label flux ropes.
    if isPlot:
        # Get WIND Pitch angle data.
        year_str = str(year_start)
        pithcAngle_DF = pd.read_pickle(pitch_dir + '/GS_' + year_str + '_PitchAngle_DataFrame_WIND.p')
    
    # Make combined plot.
    if isPlotCombined:
        if isLabelFluxRope:
            fig = FR.plot_time_series_data(data_DF, spacecraftID, case_plot_dir, fluxRopeList_DF=search_result_no_overlap_DF, shockTimeList=shockTimeList, pithcAngle_DF=pithcAngle_DF, fig_x_interval=fig_x_interval)
        else:
            fig = FR.plot_time_series_data(data_DF, spacecraftID, case_plot_dir, shockTimeList=shockTimeList, pithcAngle_DF=pithcAngle_DF, fig_x_interval=fig_x_interval)
        
        fig2_filename = fig
        
        combined = FR.vstack_images(fig1_filename, fig2_filename)
        combined_filename = os.getcwd()+'/combined/'+oneShockTime_str+'.png'
        combined.save(combined_filename)
        print('{} is saved!'.format(combined_filename))

# Done.
if isSpeak:
    os.system('say "program has finished"')





























