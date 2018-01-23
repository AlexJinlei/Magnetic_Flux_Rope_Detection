# Preprocess Data Before Apply Flux Rope Detection Algorithm
## 1. DESCRIPTION
This folder contains all needed code to preprocess raw data. The raw data is firstly downloaded from [CDAweb](https://cdaweb.sci.gsfc.nasa.gov/index.html/) via the [API](https://pypi.python.org/pypi/ai.cdas/1.1.1) provided by the website. The original data format is CDF. Then the raw data will be cleared, refined, and resampled. In this procedure, a combination of multiple data processing technique will be applied, including digital filter, smoothing, statistical methode and so on. Then these data will be combined in to data sets in years. The final data format is pickle format. The pickle file is only a container, which contains variable in pandas.DataFrame datatype.
## 2. CODE LIST
### 1) Download data
- GS_DATA_DownloadCDAwebData_WIND.py
- GS_DATA_DownloadCDAwebData_ACE.py
- GS_DATA_DownloadCDAwebData_ElectronPitchAngle.py
