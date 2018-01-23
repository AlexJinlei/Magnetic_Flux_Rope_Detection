'''
Process electron pitch angel data.
'''
import os
import time
import calendar
import glob
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

homedir = os.environ['HOME']
rootDir = '/Users/jz0006/GoogleDrive/GS/'
inputDir = rootDir + 'data_cache/'
outputDir = rootDir + 'GS_DataPickleFormat/'

for year in range(1996, 2017, 1):
    print('\nyear = {}'.format(year))
    # WI_H4_SWE Available Time Range: 1994/11/30 00:00:20 - 2001/07/10 00:00:35
    # WI_H3_SWE Available Time Range: 2002/08/16 00:00:05 - 2017/02/18 23:59:44
    year_str = str(year)
    if year in range(1996, 2002):
        filename = glob.glob(inputDir+'wi_h4s_swe_'+year_str+'*.cdf')[0]
        channel_key = 'f_pitch_E09' # 94.02eV
        energy_label = '94.02eV'
    elif year in range(2002, 2017):
        filename = glob.glob(inputDir+'wi_h3s_swe_'+year_str+'*.cdf')[0]
        channel_key = 'f_pitch_E04' # 96.71eV
        energy_label = '96.71eV'
    # Read CDF file, return an dictionary object.
    # This object does not directly store the data from the CDF;
    # rather, it provides access to the data in a format that much
    # like a Python list or numpy :class:`~numpy.ndarray`.
    # Note that, in CDAweb API, it will be converted to dictionary object automatically by API.
    print('Reading cdf files:')
    print('Reading '+ filename + '...')
    pitchAngle_cdffile = pycdf.CDF(filename)

    # Create time series timeStamp, time step is 1 minute.
    timeRangeInMinutes = (365+calendar.isleap(year))*24*60
    timeStampSeries = np.asarray([datetime(year,1,1,0,0,0) + timedelta(minutes=x) for x in range(0, timeRangeInMinutes)])

    # Extract data from cdf file.
    print('Extracting data from cdf file...')
    print('Extracting ElectronPitch_Epoch...')
    ElectronPitch_Epoch = pitchAngle_cdffile['Epoch'][...]
    print('Extracting ElectronPitch_Flux...')
    ElectronPitch_Flux = pitchAngle_cdffile[channel_key][...]
    print('Extracting ElectronPitch_PitchAngle...')
    ElectronPitch_PitchAngle = pitchAngle_cdffile['Pitch_Angle'][...]
    # Put ElectronPitch into DataFrame.
    print('Putting ElectronPitch_Flux into DataFrame...')
    ElectronPitch_Flux_DF = pd.DataFrame(ElectronPitch_Flux, index=ElectronPitch_Epoch, columns=list(ElectronPitch_PitchAngle[:]))
    # Resample to 1 minutes resolution.
    print('Resampling ElectronPitch_Flux data into 1 minute resolution...')
    ElectronPitch_Flux_DF.drop_duplicates(keep='first', inplace=True) # Drop duplicated records, keep first one.
    # Resample to 1 minute resolution. New added records will be filled with NaN.
    ElectronPitch_Flux_DF = ElectronPitch_Flux_DF.resample('1T').mean()
    # Reindex, fill index gaps.
    ElectronPitch_Flux_DF = ElectronPitch_Flux_DF.reindex(index=timeStampSeries)
    # Fill NAN with 0.0
    ElectronPitch_Flux_DF.fillna(value=0.0, inplace=True, limit=None)

    # Save DataFrame into pickle file.
    print('Saving DataFrame to pickle file...')
    ElectronPitch_Flux_DF.to_pickle(outputDir + 'GS_'+str(year)+'_PitchAngle_DataFrame_WIND.p')
    print('Done')

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
    print('Plotting pitch angle...')
    e_pitch.xaxis.set_major_formatter(fig_formatter) # Set xlabel format.
    e_pitch.set_ylabel('pitch angle('+energy_label+')',fontsize=fig_ylabel_fontsize) # Label font size.
    e_pitch.tick_params(axis='y', labelsize=fig_ytick_fontsize) # Tick font size.
    e_pitch.yaxis.set_major_locator(MaxNLocator(3))
    e_pitch.set_ylim(0,180)

    ax_e_pitch = e_pitch.pcolormesh(ElectronPitch_Flux_DF.index, ElectronPitch_PitchAngle, ElectronPitch_Flux_DF.values.transpose(), cmap='jet', norm=matplotlib.colors.LogNorm())

    box = e_pitch.get_position() # Get pannel position.
    pad, width = 0.01, 0.01 # pad = distance to panel, width = colorbar width.
    cax = fig.add_axes([box.xmax + pad, box.ymin, width, box.height]) # Set colorbar position.
    ax_e_pitch_cbar = fig.colorbar(ax_e_pitch, ax=e_pitch, cax=cax)
    #ax_e_pitch_cbar.set_label('#/[cc*(cm/s)^3]', rotation=270, fontsize=fig_ytick_fontsize)
    ax_e_pitch_cbar.ax.minorticks_on()
    ax_e_pitch_cbar.ax.tick_params(labelsize=fig_ytick_fontsize)
    ax_e_pitch_cbar.outline.set_linewidth(fig_linewidth)
    e_pitch.tick_params(axis='x', which='major', labelsize=fig_xtick_fontsize) # This is a shared axis for all subplot
    
    fig.savefig(homedir + '/Desktop/' + year_str + '_PitchAngle_' + energy_label + '.png', format='png', dpi=500, bbox_inches='tight')
    plt.close(fig)





