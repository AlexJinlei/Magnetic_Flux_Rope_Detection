'''
Calculate covariance of three componets of magnetic field.
'''
import os
import sys
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

year_str = sys.argv[1]
year = int(year_str)
datetimeStart = datetime(year,1,1,0,0,0)
datetimeEnd   = datetime(year,12,31,23,59,59)

# If turn cache on, do not download from one dataset more than one time. There is a bug to casue error.
# Make sure download every variables you need from one dataset at once.
# cdas.set_cache(True, '/Users/jz0006/GoogleDrive/MyResearchFolder/FluxRope/PythonPlotCode/data_cache')
cdas.set_cache(True, homedir + '/GoogleDrive/GS/data_cache')

# Download magnetic field data from ACE.
# The dimension of [WI_H0_MFI['BGSE'] is N X 3(N row, 3 column)
print('Downloading data from AC_H0_MFI...')
# AC_H0_MFI [Available Time Range: 1997/09/02 00:00:12 - 2016/12/24 23:59:56]
AC_H0_MFI = cdas.get_data('istp_public', 'AC_H0_MFI', datetimeStart, datetimeEnd, ['BGSEc'], cdf=True)
print('Done.')

# Download solar wind data Np, V_GSE, and Thermal speed.
print('Downloading data from AC_H0_SWE...')
# Np: Solar Wind Proton Number Density, scalar.
# V_GSE: Solar Wind Velocity in GSE coord., 3 components.
# Tpr: radial component of the proton temperature. The radial component of the proton temperature is the (1,1) component of the temperature tensor, along the radial direction. It is obtained by integration of the ion (proton) distribution function.
# Alpha to proton density ratio.
# AC_H0_SWE [Available Time Range: 1998/02/04 00:00:31 - 2016/11/27 23:59:51]
AC_H0_SWE = cdas.get_data('istp_public', 'AC_H0_SWE', datetimeStart, datetimeEnd, ['Np','V_GSE','Tpr','alpha_ratio'], cdf=True)
print('Done.')

# Download EPAM data.
print('Downloading data from AC_H1_EPM...')
# {u'ShortDescription': u'particle_flux>species', u'Name': u'P1', u'LongDescription': u'P1 LEMS30 (Low Energy Magnetic Spectrometer; 0.047-0.065 MeV Ions), Sector Avg'}
# {u'ShortDescription': u'particle_flux>species', u'Name': u'P2', u'LongDescription': u'P2 LEMS30 (Low Energy Magnetic Spectrometer; 0.065-0.112 MeV Ions), Sector Avg'}
# {u'ShortDescription': u'particle_flux>species', u'Name': u'P3', u'LongDescription': u'P3 LEMS30 (Low Energy Magnetic Spectrometer; 0.112-0.187 MeV Ions), Sector Avg'}
# {u'ShortDescription': u'particle_flux>species', u'Name': u'P4', u'LongDescription': u'P4 LEMS30 (Low Energy Magnetic Spectrometer; 0.187-0.310 MeV Ions), Sector Avg'}
# {u'ShortDescription': u'particle_flux>species', u'Name': u'P5', u'LongDescription': u'P5 LEMS30 (Low Energy Magnetic Spectrometer; 0.310-0.580 MeV Ions), Sector Avg'}
# {u'ShortDescription': u'particle_flux>species', u'Name': u'P6', u'LongDescription': u'P6 LEMS30 (Low Energy Magnetic Spectrometer; 0.580-1.06 MeV Ions), Sector Avg'}
# {u'ShortDescription': u'particle_flux>species', u'Name': u'P7', u'LongDescription': u'P7 LEMS30 (Low Energy Magnetic Spectrometer; 1.06-1.91 MeV Ions), Sector Avg'}
# {u'ShortDescription': u'particle_flux>species', u'Name': u'P8', u'LongDescription': u'P8 LEMS30 (Low Energy Magnetic Spectrometer; 1.91-4.75 MeV Ions), Sector Avg'}
AC_H1_EPM = cdas.get_data('istp_public', 'AC_H1_EPM', datetimeStart, datetimeEnd, ['P1','P2','P3','P4','P5','P6','P7','P8'], cdf=True)
print('Done.')


'''
# Download electron temprature data. Unit is Kelvin.
# WI_H0_SWE time span = [1994/12/29 00:00:02, 2001/05/31 23:59:57]
# WI_H5_SWE time span = [2002/08/16 00:00:05, 2015/05/01 00:00:12]
if (datetimeStart>=datetime(1994,12,29,0,0,2) and datetimeEnd<=datetime(2001,5,31,23,59,57)):
    print('Downloading data from WI_H0_SWE...')
    WI_H0_MFI = cdas.get_data('istp_public', 'WI_H0_SWE', datetimeStart, datetimeEnd, ['Te'], cdf=True)
    print('Done.')
elif (datetimeStart>=datetime(2002,8,16,0,0,5) and datetimeEnd<=datetime(2016,1,19,23,59,27)):
    print('Downloading data from WI_H5_SWE...')
    WI_H0_MFI = cdas.get_data('istp_public', 'WI_H5_SWE', datetimeStart, datetimeEnd, ['T_elec'], cdf=True)
    print('Done.')
else:
    print('No electron data available!')
'''









