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
pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 300)

#====================================================================================

ID_dict = {'a':'ACE', 'w':'WIND'}

# Specify spacecraft.
spacecraftID = ID_dict['a'] # 'ACE' or 'WIND'.

datetimeStart = datetime(1999,1,1,00,00)
datetimeEnd = datetime(1999,1,1,11,59)

isSearch = 1
isCombineRawReslut = 1
isGetMoreInfo = 1
isLabelFluxRope = 1
isPlot = 1
isSaveCsv = 1
isSpeak = 1

# xtick interval.
fig_x_interval = 1 # In hours.

#duration_range_tuple=((10,20),(20,30),(30,40),(40,50),(50,60),(60,80),(80,100),(100,120),(120,140),(140,160),(160,180))

duration_range_tuple=((10,20),(20,30),)

#====================================================================================

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


'''
search_result_raw_filename = search_result_dir + '/search_result_raw.p'
search_result_raw = pickle.load(open(search_result_raw_filename, 'rb'))
for item in search_result_raw['true']['10~20']:
    print(item)
exit()
'''

# 3) Detect flux ropes.
if isSearch:
    # if n_theta_grid=12, d_theta_deg = 90/12 = 7.5, d_phi_deg = 360/24 = 15
    search_result_raw = FR.detect_flux_rope(data_DF, duration_range_tuple, search_result_dir, n_theta_grid=12)
    os.system('say "Flux rope searching process has finished"')

# 4) Clean up overlapped records.
if isCombineRawReslut:
    search_result_raw_filename = search_result_dir + '/search_result_raw.p'
    search_result_no_overlap_DF = FR.clean_up_raw_result(data_DF, search_result_raw_filename, walenTest_k_threshold=0.3, min_residue_diff=0.12, min_residue_fit=0.15, output_dir=search_result_dir, isPrintIntermediateDF=False, isVerbose=False)
    #print('\nsearch_result_no_overlap_DF:')
    #print(search_result_no_overlap_DF)

# 5) Get more flux rope information.
if isGetMoreInfo:
    search_result_detail_info_DF = FR.get_more_flux_rope_info(data_DF, search_result_no_overlap_DF, output_dir=search_result_dir)
    #print(search_result_detail_info_DF)
    if isSaveCsv:
        print('Saving {}...'.format(csv_filename))
        search_result_detail_info_DF.to_csv(path_or_buf=case_plot_dir + csv_filename)
        print('Done.')

# 6) Plot time series data for given time range. Label flux ropes.
if isPlot:
    if spacecraftID=='WIND': # If WIND, load pitch angle data.
        # Get Pitch angle data.
        year_str = str(year_start)
        pithcAngle_DF = pd.read_pickle(pitch_dir + '/GS_' + year_str + '_PitchAngle_DataFrame_WIND.p')
        # Make plot.
        if isLabelFluxRope:
            fig = FR.plot_time_series_data(data_DF, spacecraftID, case_plot_dir, fluxRopeList_DF=search_result_no_overlap_DF, pithcAngle_DF=pithcAngle_DF, fig_x_interval=fig_x_interval)
        else:
            fig = FR.plot_time_series_data(data_DF, spacecraftID, case_plot_dir, pithcAngle_DF=pithcAngle_DF, fig_x_interval=fig_x_interval)

    elif spacecraftID=='ACE':
        if isLabelFluxRope:
            fig = FR.plot_time_series_data(data_DF, spacecraftID, case_plot_dir, fluxRopeList_DF=search_result_no_overlap_DF,  fig_x_interval=fig_x_interval)
        else:
            fig = FR.plot_time_series_data(data_DF, spacecraftID, case_plot_dir, fig_x_interval=fig_x_interval)

if isSpeak:
    os.system('say "Program has finished"')





























