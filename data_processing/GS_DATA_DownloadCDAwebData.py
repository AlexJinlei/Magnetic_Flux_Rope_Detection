'''
Calculate covariance of three componets of magnetic field.
'''
import os
import numpy as np # Scientific calculation package.
from numpy import linalg as la
#from aenum import Enum # Enum data type
from ai import cdas # Import CDAweb API package.
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


############################################## Download Data ###########################################
# Home directory.
homedir = os.environ['HOME']

# Set up time range.

year = 2015
datetimeStart = datetime(year,1,1,0,0,0)
datetimeEnd   = datetime(year,12,31,23,59,59)

# If turn cache on, do not download from one dataset more than one time. There is a bug to casue error.
# Make sure download every variables you need from one dataset at once.
# cdas.set_cache(True, '/Users/jz0006/GoogleDrive/MyResearchFolder/FluxRope/PythonPlotCode/data_cache')
cdas.set_cache(True, homedir + '/GoogleDrive/GS/data_cache')

# Download magnetic field data.
# The dimension of [WI_H0_MFI['BGSE'] is N X 3(N row, 3 column)
print('Downloading data from WI_H0_MFI...')
WI_H0_MFI = cdas.get_data('istp_public', 'WI_H0_MFI', datetimeStart, datetimeEnd, ['BGSE'], cdf=True)
print('Done.')

# Download solar wind data.
print('Downloading data from WI_K0_SWE...')
WI_K0_SWE = cdas.get_data('istp_public', 'WI_K0_SWE', datetimeStart, datetimeEnd, ['Np','V_GSE','THERMAL_SPD'], cdf=True)
print('Done.')

# Download electron temprature data. Unit is Kelvin.
# WI_H0_SWE time span = [1994/12/29 00:00:02, 2001/05/31 23:59:57]
# WI_H5_SWE time span = [2002/08/16 00:00:05, 2015/05/01 00:00:12]
if (datetimeStart>=datetime(1994,12,29,0,0,2) and datetimeEnd<=datetime(2001,5,31,23,59,57)):
    print('Downloading data from WI_H0_SWE...')
    WI_H0_SWE = cdas.get_data('istp_public', 'WI_H0_SWE', datetimeStart, datetimeEnd, ['Te'], cdf=True)
    print('Done.')
elif (datetimeStart>=datetime(2002,8,16,0,0,5) and datetimeEnd<=datetime(2016,1,19,23,59,27)):
    print('Downloading data from WI_H5_SWE...')
    WI_H5_SWE = cdas.get_data('istp_public', 'WI_H5_SWE', datetimeStart, datetimeEnd, ['T_elec'], cdf=True)
    print('Done.')
else:
    print('No electron data available!')










