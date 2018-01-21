'''
Calculate covariance of three componets of magnetic field.
'''
import os
import time
import calendar
import numpy as np # Scientific calculation package.
from aenum import Enum # Enum data type
import pandas as pd
from ai import cdas # Import CDAweb API package.
from scipy.signal import savgol_filter # Savitzky-Golay filter
import astropy
from spacepy import pycdf
from datetime import datetime # Import datetime class from datetime package.
from datetime import timedelta
import matplotlib
import matplotlib.pyplot as plt # Plot package, make matlab style plots.
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import FuncFormatter
from matplotlib.lines import Line2D
import matplotlib.dates as mdates
import matplotlib.cbook as cbook

'''
# Electron pitch angle.
# [Available Time Range: 1994/11/30 00:00:20 - 2001/07/10 00:00:35]
# The value reported for any bin (including the spin-averaged "energy bins") is given as a phase-space density, f [#/{cc*(cm/s)^3}], averaged over contributing detectors.
#print('Downloading data from WI_H4_SWE...')
#WI_H4_SWE = cdas.get_data('istp_public', 'WI_H4_SWE', datetimeStart, datetimeEnd, ['eV', 'f_pitch_E00', 'f_pitch_E01', 'f_pitch_E02', 'f_pitch_E03', 'f_pitch_E04', 'f_pitch_E05', 'f_pitch_E06', 'f_pitch_E07', 'f_pitch_E08', 'f_pitch_E09', 'f_pitch_E10', 'f_pitch_E11', 'f_pitch_E12', 'f_pitch_E13', 'f_pitch_E14', 'f_pitch_E15'], cdf=True)
#print('Done.')
# Check keys: dict_keys(['Pitch_Angle', 'Echan_vals', 'f_pitch_E02', 'f_pitch_E03', 'f_pitch_E00', 'f_pitch_E01', 'f_pitch_E06', 'f_pitch_E07', 'f_pitch_E04', 'f_pitch_E05', 'f_pitch_E08', 'f_pitch_E09', 'eV', 'Epoch', 'Ve', 'f_pitch_E15', 'f_pitch_E14', 'f_pitch_E11', 'f_pitch_E10', 'f_pitch_E13', 'f_pitch_E12', 'metavar0', 'metavar1'])
# WI_H4_SWE['eV'][0] >>> array([   8.29799938,   10.75926971,   13.92375851,   17.93211555,  22.92497635,   29.74621391,   38.3255043 ,   49.78798676,   63.57110214,   94.02053833,  138.67500305,  203.30096436,   297.8137207 ,  441.55197144,  658.07348633,  971.28771973], dtype=float32)
'''

############################################## Download Data ###########################################
# Home directory.
homedir = os.environ['HOME']
# If turn cache on, do not download from one dataset more than one time. There is a bug to casue error.
# Make sure download every variables you need from one dataset at once.
# cdas.set_cache(True, '/Users/jz0006/GoogleDrive/MyResearchFolder/FluxRope/PythonPlotCode/data_cache')
cdas.set_cache(True, homedir + '/GoogleDrive/GS/data_cache')

for year in [1996]:
    # Set up time range.
    #year = 2008
    datetimeStart = datetime(year,1,1,0,0,0)
    #datetimeStart = datetime(year,8,16,0,0,5)
    datetimeEnd   = datetime(year,12,31,23,59,59)
    #datetimeEnd   = datetime(year,7,10,0,0,35)

    # Create time series timeStamp, time step is 1 minute.
    #timeRange = datetimeEnd - datetimeStart
    #timeRangeInMinutes = int(timeRange.total_seconds()//60)
    timeRangeInMinutes = (365+calendar.isleap(year))*24*60
    timeStampSeries = np.asarray([datetime(year,1,1,0,0,0) + timedelta(minutes=x) for x in range(0, timeRangeInMinutes)])

    # WI_H4_SWE Available Time Range: 1994/11/30 00:00:20 - 2001/07/10 00:00:35
    # If time range is within the range covered by WI_H4_SWE.
    if (datetimeStart>=datetime(1994,11,30,0,0,20))and(datetimeEnd<=datetime(2001,7,10,0,0,35)):
        print('Downloading data from WI_H4_SWE...')
        print(time.ctime())
        #WI_H4_SWE = cdas.get_data('istp_public', 'WI_H4_SWE', datetimeStart, datetimeEnd, ['f_pitch_E09', 'f_pitch_E10'], cdf=True)
        WI_H4_SWE = cdas.get_data('istp_public', 'WI_H4_SWE', datetimeStart, datetimeEnd, ['f_pitch_E09'], cdf=True)
        print(time.ctime())
        print('Done.')
        #print(WI_H4_SWE.keys())
        # Extract data from cdf file.
        print('Extracting data from cdf file...')
        print('Extracting ElectronPitch_Epoch...')
        ElectronPitch_Epoch = WI_H4_SWE['Epoch'][...]
        print('Extracting ElectronPitch_Flux...')
        ElectronPitch_Flux_094eV = WI_H4_SWE['f_pitch_E09'][...]
        #ElectronPitch_Flux_138eV = WI_H4_SWE['f_pitch_E10'][...]
        print(ElectronPitch_Flux_094eV.shape)
        print('Extracting ElectronPitch_PitchAngle...')
        ElectronPitch_PitchAngle = WI_H4_SWE['Pitch_Angle'][...]
        # Put ElectronPitch into DataFrame.
        print('Putting ElectronPitch_Flux_094eV into DataFrame...')
        ElectronPitch_Flux_094eV_DF = pd.DataFrame(ElectronPitch_Flux_094eV, index=ElectronPitch_Epoch, columns=list(ElectronPitch_PitchAngle[:]))
        #print('Putting ElectronPitch_Flux_138eV into DataFrame...')
        #ElectronPitch_Flux_138eV_DF = pd.DataFrame(ElectronPitch_Flux_138eV, index=ElectronPitch_Epoch, columns=list(ElectronPitch_PitchAngle[:]))
        # Resample to 1 minutes resolution.
        print(time.ctime())
        print('Resampling ElectronPitch_Flux_094eV data into 1 minute resolution...')
        ElectronPitch_Flux_094eV_DF.drop_duplicates(keep='first', inplace=True) # Drop duplicated records, keep first one.
        # Resample to 1 minute resolution. New added records will be filled with NaN.
        ElectronPitch_Flux_094eV_DF = ElectronPitch_Flux_094eV_DF.resample('1T').mean()
        # Reindex, fill index gaps.
        ElectronPitch_Flux_094eV_DF = ElectronPitch_Flux_094eV_DF.reindex(index=timeStampSeries)
        # Fill NAN with 0.0
        ElectronPitch_Flux_094eV_DF.fillna(value=0.0, inplace=True, limit=None)
        print(time.ctime())
        '''
        print('Resampling ElectronPitch_Flux_138eV data into 1 minute resolution...')
        ElectronPitch_Flux_138eV_DF.drop_duplicates(keep='first', inplace=True) # Drop duplicated records, keep first one.
        # Resample to 1 minute resolution. New added records will be filled with NaN.
        ElectronPitch_Flux_138eV_DF = ElectronPitch_Flux_138eV_DF.resample('1T').mean()
        # Reindex, fill index gaps.
        ElectronPitch_Flux_138eV_DF = ElectronPitch_Flux_138eV_DF.reindex(index=timeStampSeries)
        # Fill NAN with 0.0
        ElectronPitch_Flux_138eV_DF.fillna(value=0.0, inplace=True, limit=None)
        '''
        print(ElectronPitch_Flux_094eV_DF)

        # Create multi-plot fig.
        fig, ax = plt.subplots(1, 1, sharex=True,figsize=(10, 1))
        e_pitch = ax

        # Plots format defination.
        #fig_formatter = mdates.DateFormatter('%m/%d %H:%M') # Full format is ('%Y-%m-%d %H:%M:%S').
        fig_formatter = mdates.DateFormatter('%m/%d') # Full format is ('%Y-%m-%d %H:%M:%S').
        fig_ylabel_fontsize = 9
        fig_ytick_fontsize = 9
        fig_xtick_fontsize = 8
        fig_linewidth=0.5
        # Plot WIN Pitch Angle f_pitch_E09.
        print('Plotting pitch_E09...')
        e_pitch.xaxis.set_major_formatter(fig_formatter) # Set xlabel format.
        e_pitch.set_ylabel('E09(94.02eV)\n pitch angle',fontsize=fig_ylabel_fontsize) # Label font size.
        e_pitch.tick_params(axis='y', labelsize=fig_ytick_fontsize) # Tick font size.
        e_pitch.yaxis.set_major_locator(MaxNLocator(3))
        e_pitch.set_ylim(0,180)

        ax_e_pitch = e_pitch.pcolormesh(ElectronPitch_Flux_094eV_DF.index, ElectronPitch_PitchAngle, ElectronPitch_Flux_094eV_DF.values.transpose(), cmap='jet', norm=matplotlib.colors.LogNorm())

        box = e_pitch.get_position() # Get pannel position.
        pad, width = 0.01, 0.01 # pad = distance to panel, width = colorbar width.
        cax = fig.add_axes([box.xmax + pad, box.ymin, width, box.height]) # Set colorbar position.
        ax_e_pitch_cbar = fig.colorbar(ax_e_pitch, ax=e_pitch, cax=cax)
        #ax_e_pitch_cbar.set_label('#/[cc*(cm/s)^3]', rotation=270, fontsize=fig_ytick_fontsize)
        ax_e_pitch_cbar.ax.minorticks_on()
        ax_e_pitch_cbar.ax.tick_params(labelsize=fig_ytick_fontsize)
        ax_e_pitch_cbar.outline.set_linewidth(fig_linewidth)
        e_pitch.tick_params(axis='x', which='major', labelsize=fig_xtick_fontsize) # This is a shared axis for all subplot
        
        datetimeStart_str = str(datetimeStart.strftime('%Y%m%d%H%M'))
        datetimeEnd_str = str(datetimeEnd.strftime('%Y%m%d%H%M'))
        timeRange_str = datetimeStart_str + '_' + datetimeEnd_str
        
        fig.savefig(homedir + '/Desktop/' + timeRange_str + '_WI_H4_SWE.png', format='png', dpi=500, bbox_inches='tight')
        #fig.savefig(homedir + '/Desktop/' + fileTitle + '.eps', format='eps', dpi=300, bbox_inches='tight')
    elif (datetimeStart>=datetime(2002,8,16,0,0,5))and(datetimeEnd<=datetime(2017,2,18,23,59,44)):
        # Available Time Range: 2002/08/16 00:00:05 - 2017/02/18 23:59:44
        print('Downloading data from WI_H3_SWE...')
        print(time.ctime())
        # cdas.get_variables('istp_public', 'WI_H3_SWE')
        WI_H3_SWE = cdas.get_data('istp_public', 'WI_H3_SWE', datetimeStart, datetimeEnd, ['f_pitch_E04'], cdf=True)
        print(time.ctime())
        print('Done.')
        #print(WI_H3_SWE.keys())
        # Extract data from cdf file.
        print('Extracting data from cdf file...')
        print('Extracting ElectronPitch_Epoch...')
        ElectronPitch_Epoch = WI_H3_SWE['Epoch'][...]
        print('Extracting ElectronPitch_Flux...')
        ElectronPitch_Flux_096eV = WI_H3_SWE['f_pitch_E04'][...]
        print(ElectronPitch_Flux_096eV.shape)
        print('Extracting ElectronPitch_PitchAngle...')
        ElectronPitch_PitchAngle = WI_H3_SWE['Pitch_Angle'][...]
        # Put ElectronPitch into DataFrame.
        print('Putting ElectronPitch_Flux_096eV into DataFrame...')
        ElectronPitch_Flux_096eV_DF = pd.DataFrame(ElectronPitch_Flux_096eV, index=ElectronPitch_Epoch, columns=list(ElectronPitch_PitchAngle[:]))
        # Resample to 1 minutes resolution.
        print(time.ctime())
        print('Resampling ElectronPitch_Flux_096eV data into 1 minute resolution...')
        ElectronPitch_Flux_096eV_DF.drop_duplicates(keep='first', inplace=True) # Drop duplicated records, keep first one.
        # Resample to 1 minute resolution. New added records will be filled with NaN.
        ElectronPitch_Flux_096eV_DF = ElectronPitch_Flux_096eV_DF.resample('1T').mean()
        # Reindex, fill index gaps.
        ElectronPitch_Flux_096eV_DF = ElectronPitch_Flux_096eV_DF.reindex(index=timeStampSeries)
        # Fill NAN with 0.0
        ElectronPitch_Flux_096eV_DF.fillna(value=0.0, inplace=True, limit=None)
        print(time.ctime())
        print(ElectronPitch_Flux_096eV_DF)

        # Create multi-plot fig.
        fig, ax = plt.subplots(1, 1, sharex=True,figsize=(10, 1))
        e_pitch = ax

        # Plots format defination.
        #fig_formatter = mdates.DateFormatter('%m/%d %H:%M') # Full format is ('%Y-%m-%d %H:%M:%S').
        fig_formatter = mdates.DateFormatter('%m/%d') # Full format is ('%Y-%m-%d %H:%M:%S').
        fig_ylabel_fontsize = 9
        fig_ytick_fontsize = 9
        fig_xtick_fontsize = 8
        fig_linewidth=0.5
        # Plot WIN Pitch Angle f_pitch_E09.
        print('Plotting pitch_E05...')
        e_pitch.xaxis.set_major_formatter(fig_formatter) # Set xlabel format.
        e_pitch.set_ylabel('E04(96.71eV)\n pitch angle',fontsize=fig_ylabel_fontsize) # Label font size.
        e_pitch.tick_params(axis='y', labelsize=fig_ytick_fontsize) # Tick font size.
        e_pitch.yaxis.set_major_locator(MaxNLocator(3))
        e_pitch.set_ylim(0,180)

        ax_e_pitch = e_pitch.pcolormesh(ElectronPitch_Flux_096eV_DF.index, ElectronPitch_PitchAngle, ElectronPitch_Flux_096eV_DF.values.transpose(), cmap='jet', norm=matplotlib.colors.LogNorm())

        box = e_pitch.get_position() # Get pannel position.
        pad, width = 0.01, 0.01 # pad = distance to panel, width = colorbar width.
        cax = fig.add_axes([box.xmax + pad, box.ymin, width, box.height]) # Set colorbar position.
        ax_e_pitch_cbar = fig.colorbar(ax_e_pitch, ax=e_pitch, cax=cax)
        #ax_e_pitch_cbar.set_label('#/[cc*(cm/s)^3]', rotation=270, fontsize=fig_ytick_fontsize)
        ax_e_pitch_cbar.ax.minorticks_on()
        ax_e_pitch_cbar.ax.tick_params(labelsize=fig_ytick_fontsize)
        ax_e_pitch_cbar.outline.set_linewidth(fig_linewidth)
        e_pitch.tick_params(axis='x', which='major', labelsize=fig_xtick_fontsize) # This is a shared axis for all subplot
        
        datetimeStart_str = str(datetimeStart.strftime('%Y%m%d%H%M'))
        datetimeEnd_str = str(datetimeEnd.strftime('%Y%m%d%H%M'))
        timeRange_str = datetimeStart_str + '_' + datetimeEnd_str
        
        fig.savefig(homedir + '/Desktop/' + timeRange_str + '_WI_H3_SWE.png', format='png', dpi=500, bbox_inches='tight')
        #fig.savefig(homedir + '/Desktop/' + fileTitle + '.eps', format='eps', dpi=300, bbox_inches='tight')


        
    else:
        print('No data.')





exit()








#ElectronPitch_filename = 'wi_h4s_swe_19980324000058_19980326235837.cdf'

print('Reading cdf files:')
#print('Reading '+ ElectronPitch_filename + '...')

ElectronPitch_cdffile = WI_H4_SWE

#ElectronPitch_cdffile = pycdf.CDF('~/GoogleDrive/GS/data_cache/' + ElectronPitch_filename)



'''
# Put ElectronPitch into DataFrame.
ElectronPitch_DF = pd.DataFrame(ElectronPitch_Flux, index=ElectronPitch_Epoch, columns=list(ElectronPitch_PitchAngle[:]))
# Resample to 1 minutes resolution.
print('Resampling ElectronPitch_DF data into 1 minute resolution...')
ElectronPitch_DF.drop_duplicates(keep='first', inplace=True) # Drop duplicated records, keep first one.
# Resample to 1 minute resolution. New added records will be filled with NaN.
ElectronPitch_DF = ElectronPitch_DF.resample('1T').mean()
ElectronPitch_DF.fillna(value=0.0, inplace=True, limit=None)
#print(ElectronPitch_DF)

ElectronPitch_array = ElectronPitch_DF.values
print((ElectronPitch_array.shape))
'''



'''
# Electron pitch angle.
print('Downloading data from WI_H4_SWE...')
WI_H4_SWE = cdas.get_data('istp_public', 'WI_H4_SWE', datetimeStart, datetimeEnd, ['eV', 'f_pitch_E09'], cdf=True)
print('Done.')
# Truncate datetime to minutes.
for i in range(len(WI_H4_SWE['Epoch'])):
    WI_H4_SWE['Epoch'][i] = WI_H4_SWE['Epoch'][i].replace(second=0, microsecond=0)
# Calculate time range in minutes.
timeRangeInMinutes = int((WI_H4_SWE['Epoch'][-1] - WI_H4_SWE['Epoch'][1]).total_seconds())//60
# Get array shape.
WI_H4_SWE_arrayShape = WI_H4_SWE['f_pitch_E09'].shape
# Generate resampled time array.
resampleEpoch = np.asarray([WI_H4_SWE['Epoch'][0] + timedelta(minutes=x) for x in range(0, timeRangeInMinutes)])
print(resampleEpoch)
print('Resampling f_pitch_E09...')
# Generate resampled pitch angle array.
resample_pitch_E09 = np.zeros((len(resampleEpoch), WI_H4_SWE_arrayShape[1]))
# Resample pitch angle data by averaging.
for i in range(len(resampleEpoch)):
    selectedArrayByTimeStamp = WI_H4_SWE['f_pitch_E09'][WI_H4_SWE['Epoch']==resampleEpoch[i]]
    if len(selectedArrayByTimeStamp) != 0:
        resample_pitch_E09[i] = np.mean(selectedArrayByTimeStamp, axis=0 )
        #print(resample_pitch_E09[i]) # One line, with all pitch angles
print('Done.')
'''


'''
# Create multi-plot fig.
fig, ax = plt.subplots(1, 1, sharex=True,figsize=(8, 1))
e_pitch = ax

# Plots format defination.
fig_formatter = mdates.DateFormatter('%m/%d %H:%M') # Full format is ('%Y-%m-%d %H:%M:%S').
fig_ylabel_fontsize = 9
fig_ytick_fontsize = 9
fig_xtick_fontsize = 8
fig_linewidth=0.5

# 5) Plot WIN Pitch Angle f_pitch_E09.
print('Plotting pitch_E09...')
e_pitch.xaxis.set_major_formatter(fig_formatter) # Set xlabel format.
e_pitch.set_ylabel('E09(94.0eV)\n pitch angle',fontsize=fig_ylabel_fontsize) # Label font size.
e_pitch.tick_params(axis='y', labelsize=fig_ytick_fontsize) # Tick font size.
e_pitch.yaxis.set_major_locator(MaxNLocator(3))
e_pitch.set_ylim(0,180)

#print(ElectronPitch_array)
#print(ElectronPitch_DF)
ax_e_pitch = e_pitch.pcolormesh(ElectronPitch_DF.index, ElectronPitch_PitchAngle, ElectronPitch_array.transpose(), cmap='jet', norm=matplotlib.colors.LogNorm())

box = e_pitch.get_position() # Get pannel position.
pad, width = 0.01, 0.01 # pad = distance to panel, width = colorbar width.
cax = fig.add_axes([box.xmax + pad, box.ymin, width, box.height]) # Set colorbar position.
ax_e_pitch_cbar = fig.colorbar(ax_e_pitch, ax=e_pitch, cax=cax)
#ax_e_pitch_cbar.set_label('#/[cc*(cm/s)^3]', rotation=270, fontsize=fig_ytick_fontsize)
ax_e_pitch_cbar.ax.minorticks_on()
ax_e_pitch_cbar.ax.tick_params(labelsize=fig_ytick_fontsize)
ax_e_pitch_cbar.outline.set_linewidth(fig_linewidth)
e_pitch.tick_params(axis='x', which='major', labelsize=fig_xtick_fontsize) # This is a shared axis for all subplot

fig.savefig(homedir + '/Desktop/' + 'hahah' + '.png', format='png', dpi=500, bbox_inches='tight')
#fig.savefig(homedir + '/Desktop/' + fileTitle + '.eps', format='eps', dpi=300, bbox_inches='tight')
exit()

## Electron pitch angle.
#print('Downloading data from WI_H4_SWE...')
#WI_H4_SWE = cdas.get_data('istp_public', 'WI_H4_SWE', datetimeStart, datetimeEnd, ['eV', 'f_pitch_E00', 'f_pitch_E01', 'f_pitch_E02', 'f_pitch_E03', 'f_pitch_E04', 'f_pitch_E05', 'f_pitch_E06', 'f_pitch_E07', 'f_pitch_E08', 'f_pitch_E09', 'f_pitch_E10', 'f_pitch_E11', 'f_pitch_E12', 'f_pitch_E13', 'f_pitch_E14', 'f_pitch_E15'], cdf=True)
#print('Done.')
# Check keys: dict_keys(['Pitch_Angle', 'Echan_vals', 'f_pitch_E02', 'f_pitch_E03', 'f_pitch_E00', 'f_pitch_E01', 'f_pitch_E06', 'f_pitch_E07', 'f_pitch_E04', 'f_pitch_E05', 'f_pitch_E08', 'f_pitch_E09', 'eV', 'Epoch', 'Ve', 'f_pitch_E15', 'f_pitch_E14', 'f_pitch_E11', 'f_pitch_E10', 'f_pitch_E13', 'f_pitch_E12', 'metavar0', 'metavar1'])
# WI_H4_SWE['eV'][0] >>> array([   8.29799938,   10.75926971,   13.92375851,   17.93211555,  22.92497635,   29.74621391,   38.3255043 ,   49.78798676,   63.57110214,   94.02053833,  138.67500305,  203.30096436,   297.8137207 ,  441.55197144,  658.07348633,  971.28771973], dtype=float32)

# Truncate datetime to minutes.
for i in range(len(WI_H4_SWE['Epoch'])):
    WI_H4_SWE['Epoch'][i] = WI_H4_SWE['Epoch'][i].replace(second=0, microsecond=0)
# Calculate time range in minutes.
timeRangeInMinutes = int((WI_H4_SWE['Epoch'][-1] - WI_H4_SWE['Epoch'][1]).total_seconds())//60
# Get array shape.
WI_H4_SWE_arrayShape = WI_H4_SWE['f_pitch_E09'].shape
# Generate resampled time array.
resampleEpoch = np.asarray([WI_H4_SWE['Epoch'][0] + timedelta(minutes=x) for x in range(0, timeRangeInMinutes)])
print('Resampling f_pitch_E09...')
# Generate resampled pitch angle array.
resample_pitch_E09 = np.zeros((len(resampleEpoch), WI_H4_SWE_arrayShape[1]))
# Resample pitch angle data by averaging.
for i in range(len(resampleEpoch)):
    selectedArrayByTimeStamp = WI_H4_SWE['f_pitch_E09'][WI_H4_SWE['Epoch']==resampleEpoch[i]]
    if len(selectedArrayByTimeStamp) != 0:
        resample_pitch_E09[i] = np.mean(selectedArrayByTimeStamp, axis=0 )
print('Done.')

'''









