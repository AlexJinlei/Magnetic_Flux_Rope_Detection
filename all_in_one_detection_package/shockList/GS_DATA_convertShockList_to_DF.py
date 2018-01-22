import os
import sys
import glob
import pickle
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 120)

# Specify input and output path
rootDir = '/Users/jz0006/GoogleDrive/GS/'
inputDir = rootDir + 'GS_FluxRopeDetectionPackage/shockList/'
inputShockFile = 'shocks_20170303_035000.csv'
outputDir = rootDir + 'GS_FluxRopeDetectionPackage/shockList/'

# Read Shock Data.
IPShock_DF = pd.read_csv(inputDir+inputShockFile, header=0)
shockTime = IPShock_DF.apply(lambda row: datetime(row['Year'], row['Month'], row['Day'], row['Hour'], row['Minute'], row['Second']), axis=1)
IPShock_DF.insert(0, 'shockTime', shockTime) # This command is able to specify column position.
IPShock_DF.drop(['Year', 'Month', 'Day', 'Hour', 'Minute', 'Second'],inplace=True,axis=1)
IPShock_DF.sort_values(by='shockTime', inplace=True)
IPShock_DF.set_index('shockTime', drop=True, inplace=True) # Use shock time as index.
IPShock_DF.to_pickle(outputDir + '/IPShock_DF.p')

# IPShock_ACE_or_WIND_DF.
IPShock_ACE_or_WIND_DF = IPShock_DF[(IPShock_DF['Spacecraft'].str.contains('Wind')) | (IPShock_DF['Spacecraft'].str.contains('ACE'))]
IPShock_ACE_or_WIND_DF.to_pickle(outputDir + '/IPShock_ACE_or_WIND_DF.p')

# IPShock_ACE_or_WIND_1996_2016_DF.
IPShock_ACE_or_WIND_1996_2016_DF = IPShock_ACE_or_WIND_DF[(IPShock_ACE_or_WIND_DF.index>=datetime(1996,1,1,0,0,0))&(IPShock_ACE_or_WIND_DF.index<=datetime(2016,12,31,23,59,59))]
IPShock_ACE_or_WIND_1996_2016_DF.to_pickle(outputDir + '/IPShock_ACE_or_WIND_1996_2016_DF.p')

# IPShock_ACE_1998_2016_DF.
IPShock_ACE_1998_2016_DF = IPShock_ACE_or_WIND_1996_2016_DF[(IPShock_ACE_or_WIND_1996_2016_DF['Spacecraft'].str.contains('ACE'))]
IPShock_ACE_1998_2016_DF.to_pickle(outputDir + '/IPShock_ACE_1998_2016_DF.p')

# IPShock_WIND_1996_2016_DF.
IPShock_WIND_1996_2016_DF = IPShock_ACE_or_WIND_1996_2016_DF[(IPShock_ACE_or_WIND_1996_2016_DF['Spacecraft'].str.contains('Wind'))]
IPShock_WIND_1996_2016_DF.to_pickle(outputDir + '/IPShock_WIND_1996_2016_DF.p')

# IPShock_ACE_and_WIND_DF (only shock time) from 1996 to 2016.
IPShock_ACE_and_WIND_1998_2016_DF = pd.DataFrame(columns=['ACE', 'WIND'])
for index_time, oneShockRecord_ACE in IPShock_ACE_1998_2016_DF.iterrows():
    # Pick one record in ACE.
    timeRangeStart = index_time - timedelta(hours=3)
    timeRangeEnd   = index_time + timedelta(hours=3)
    # Find correspondence in WIND.
    correspondenceShockRecord_WIND = IPShock_WIND_1996_2016_DF[(IPShock_WIND_1996_2016_DF.index>timeRangeStart)&(IPShock_WIND_1996_2016_DF.index<timeRangeEnd)]
    if not correspondenceShockRecord_WIND.empty:
        if len(correspondenceShockRecord_WIND)==1:
            record_temp = pd.DataFrame({'ACE':[index_time], 'WIND':[correspondenceShockRecord_WIND.index[0]]})
            IPShock_ACE_and_WIND_1998_2016_DF = IPShock_ACE_and_WIND_1998_2016_DF.append(record_temp, ignore_index=True)
        else:
            # Choose the closest one.
            deltaTimeList = abs(correspondenceShockRecord_WIND.index - index_time)
            deltaTimeArray = np.array(deltaTimeList)
            selectedRecord_index = np.argmin(deltaTimeArray)
            record_temp = pd.DataFrame({'ACE':[index_time], 'WIND':[correspondenceShockRecord_WIND.index[selectedRecord_index]]})
            IPShock_ACE_and_WIND_1998_2016_DF = IPShock_ACE_and_WIND_1998_2016_DF.append(record_temp, ignore_index=True)

IPShock_ACE_and_WIND_1998_2016_DF.to_pickle(outputDir + '/IPShock_ACE_and_WIND_1998_2016_DF.p')
print(IPShock_ACE_and_WIND_1998_2016_DF)

exit()
















