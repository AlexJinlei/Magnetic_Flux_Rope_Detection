# Preprocess Data Before Apply Flux Rope Detection Algorithm
## 1. DESCRIPTION
This folder contains all needed code to preprocess raw data. The raw data is firstly downloaded from [CDAweb](https://cdaweb.sci.gsfc.nasa.gov/index.html/) via the [API](https://pypi.python.org/pypi/ai.cdas/1.1.1) provided by the website. The original data format is CDF. Then the raw data will be cleared, refined, and resampled. In this procedure, a combination of multiple data processing technique will be applied, including digital filter, smoothing, statistical methode and so on. Then these data will be combined in to data sets in years. The final data format is pickle format. The pickle file is only a container, which contains variable in pandas.DataFrame datatype.
## 2. CODE LIST
### 1) Download data
- GS_DATA_DownloadCDAwebData_WIND.py
- GS_DATA_DownloadCDAwebData_ACE.py
- GS_DATA_DownloadCDAwebData_ElectronPitchAngle.py

These three files are used to download data observed by Wind spacecraft or ACE spacecrafts.

GS_DATA_DownloadCDAwebData_WIND.py downloads magnetic field data, solar wind data, and electron temprature data from Wind spacecraft.

GS_DATA_DownloadCDAwebData_ACE.py downloads magnetic field data, solar wind data (Np, V_GSE, Thermal speed, and Alpha/Proton ratio), and EPAM data (ions, channel P1 to P8) from ACE spacecraft.

GS_DATA_DownloadCDAwebData_ElectronPitchAngle.py downloads electron pitch angle data from Wind spacecraft.

Please specify year, start date and end date in the first a few lines in each code. Besides that, you may want to change the directories to save downloaded data.

### 2) Preprocess data
- GS_DATA_preprocess_one_year_data_Wind.py
- GS_DATA_preprocess_one_year_data_ACE.py
- GS_DATA_preprocess_ElectronPitchAngle.py

These three files are used to preprocess downloaded data. Please specify input data file name and the path to save output data.

### 3) Process data from other data source
- GS_DATA_convertShockList_to_DF.py

This file convert shock list data to pandas.DataFrame format.

### 4) Check processed data
- GS_DATA_check_processed_DataFrame.py

This file plot the original and the processed data to check the quality.
