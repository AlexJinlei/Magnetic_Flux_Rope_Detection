#!/usr/local/bin/python

import pickle
import numpy as np # Scientific calculation package.
import pandas as pd
import scipy as sp
from datetime import datetime # Import datetime class from datetime package.
import matplotlib
import matplotlib.pyplot as plt # Plot package, make matlab style plots.

# =================================== Read and Check data =======================================

GS_AllData_DataFrame = pd.read_pickle('../GS_DataPickleFormat/GS_1997_AllData_DataFrame_preprocessed.p')
GS_AllData_DataFrame0 = pd.read_pickle('../GS_DataPickleFormat/GS_1997_AllData_DataFrame_original.p')

print('Checking keys... {}'.format(GS_AllData_DataFrame.keys()))
print('Checking shape... {}'.format(GS_AllData_DataFrame.shape))
print('Checking the number of NaN... {}'.format(GS_AllData_DataFrame.isnull().values.sum()))

print('Checking keys... {}'.format(GS_AllData_DataFrame0.keys()))
print('Checking shape... {}'.format(GS_AllData_DataFrame0.shape))
print('Checking the number of NaN... {}'.format(GS_AllData_DataFrame0.isnull().values.sum()))

# Plots parameters
fig_line_width = 0.4
fig_ylabel_fontsize = 9
fig_xtick_fontsize = 8
fig_ytick_fontsize = 6
fig_legend_size = 5
n_plot_rows = len(GS_AllData_DataFrame.keys())

# ============================================ Plot =============================================

plotDateTimeStart = datetime(1997,1,1,0,0,0)
plotDateTimeEnd = datetime(1997,1,31,23,59,59)

#plotDateTimeStart = GS_AllData_DataFrame.index[1] # The beginning of all data.
#plotDateTimeEnd = GS_AllData_DataFrame.index[-1] # The end of all data.

# Plot processed data.
print('Plotting processed data..')
fig,ax = plt.subplots(n_plot_rows, 1, sharex=True,figsize=(16, 9))
i = 0 # Subplot index.
for key in GS_AllData_DataFrame.keys():
    print('Plotting {}...'.format(key))
    ax[i].plot(GS_AllData_DataFrame[plotDateTimeStart:plotDateTimeEnd].index, GS_AllData_DataFrame[key][plotDateTimeStart:plotDateTimeEnd], linewidth=fig_line_width, color='blue', label=str(key)+'_resampled')
    ax[i].set_ylabel(str(key), fontsize=fig_ylabel_fontsize)
    ax[i].tick_params(axis='y', which='major', labelsize=fig_ytick_fontsize)
    ax[i].legend(loc='upper left',prop={'size':fig_legend_size})
    i += 1
# This is a shared x axis.
ax[n_plot_rows-1].tick_params(axis='x', which='major', labelsize=fig_xtick_fontsize)
fig.savefig('GS_AllData_DataFrame_processed('+str(plotDateTimeStart)+'~'+str(plotDateTimeEnd)+').png',\
            format='png', dpi=500, bbox_inches='tight')

# Plot original data vs processed data.
print('Plotting processed vs original data..')
fig,ax = plt.subplots(n_plot_rows, 1, sharex=True,figsize=(16, 9))
i = 0 # Subplot index.
for key in GS_AllData_DataFrame.keys():
    print('Plotting {}...'.format(key))
    # Plot originla data.
    ax[i].plot(GS_AllData_DataFrame0[plotDateTimeStart:plotDateTimeEnd].index, \
               GS_AllData_DataFrame0[plotDateTimeStart:plotDateTimeEnd][key], \
               linewidth=fig_line_width, color='red', label=str(key)+'_original')
    # Plot processed data.
    ax[i].plot(GS_AllData_DataFrame[plotDateTimeStart:plotDateTimeEnd].index, \
               GS_AllData_DataFrame[plotDateTimeStart:plotDateTimeEnd][key], \
               linewidth=fig_line_width, color='blue', label=str(key)+'_resampled')
    ax[i].set_ylabel(str(key), fontsize=fig_ylabel_fontsize)
    ax[i].tick_params(axis='y', which='major', labelsize=fig_ytick_fontsize)
    ax[i].legend(loc='upper left',prop={'size':fig_legend_size})
    
    i += 1
# This is a shared x axis.
ax[n_plot_rows-1].tick_params(axis='x', which='major', labelsize=fig_xtick_fontsize)
fig.savefig('GS_AllData_DataFrame_original_vs_processed('+str(plotDateTimeStart)+'~'+str(plotDateTimeEnd)+').png',format='png', dpi=500, bbox_inches='tight')


























