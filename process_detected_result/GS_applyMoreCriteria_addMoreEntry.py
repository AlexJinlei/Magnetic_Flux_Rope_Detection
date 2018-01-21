# V4.2
# 1) Apply more criteria to the flux rope database.
# 2) Add wait time. (Adding wait time is always the last step, after selecting.)
# 3) Add shock label. Label the flux ropes after shock within 12 hours. Add \theta_{nb}.
# 4) Add daysToNearestHCS. turnTime to HCS time.
# 5) Add daysToExhaust. turnTime to Exhaust time.

import os
import sys
import glob
import pickle
import numpy as np
import pandas as pd
import datetime
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')

AU = 149597870700 # meters. Astronomical Unit.

pd.set_option('display.max_rows', 200)
pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 120)

# Specify input and output path
rootDir = '/Users/jz0006/GoogleDrive/GS/'
inputListDir = 'GS_SearchResult/detailed_info/'
inputShockDir = 'GS_DataCsv/'
inputShockFile = 'shocks_20170303_035000.csv'
inputHcsDir = 'GS_DataCsv/'
inputHcsFile = 'Leif_Svalgaard_IMF_Sector_Boundaries_1926_2017.csv'
inputExhaustDir = 'GS_DataCsv/'
inputExhaustFile = 'ACE_ExhaustList.csv'

outputDir = 'GS_SearchResult/selected_events/'
outputPlotDir = 'GS_statistical_plot/'

# Read Shock Data.
IPShock_DF = pd.read_csv(rootDir+inputShockDir+inputShockFile, header=0)
shockTime = IPShock_DF.apply(lambda row: datetime(row['Year'], row['Month'], row['Day'], row['Hour'], row['Minute'], row['Second']), axis=1)
IPShock_DF.insert(0, 'shockTime', shockTime) # This command is able to specify column position.
IPShock_DF.drop(['Year', 'Month', 'Day', 'Hour', 'Minute', 'Second'],inplace=True,axis=1)
# Pick only Wind events, and restrict the time range from 1996 to 2016.
IPShock_WIND_DF = IPShock_DF[IPShock_DF['Spacecraft'].str.contains('Wind')]
IPShock_WIND_1996_2016_DF = IPShock_WIND_DF[(IPShock_WIND_DF['shockTime']>=datetime(1996,1,1,0,0,0))&(IPShock_WIND_DF['shockTime']<=datetime(2016,12,31,23,59,59))]
IPShock_WIND_1996_2016_DF = IPShock_WIND_1996_2016_DF.sort_values(by='shockTime').copy()
IPShock_WIND_1996_2016_DF.reset_index(drop=True, inplace=True)
#print(IPShock_WIND_1996_2016_DF[['shockTime', 'Spacecraft']])

# Read HCS Data.
HCS_DF = pd.read_csv(rootDir+inputHcsDir+inputHcsFile, header=0)
hcsTime = HCS_DF.apply(lambda row: datetime(row['Year'], row['Month'], row['Day'], 12, 0, 0), axis=1)
HCS_DF.insert(0, 'hcsTime', hcsTime) # This command is able to specify column position.
HCS_DF.drop(['Year', 'Month', 'Day'],inplace=True,axis=1)
# Restrict the time range from 1996 to 2016.
HCS_1996_2016_DF = HCS_DF[(HCS_DF['hcsTime']>=datetime(1996,1,1,0,0,0))&(HCS_DF['hcsTime']<=datetime(2016,12,31,23,59,59))]
HCS_1996_2016_DF = HCS_1996_2016_DF.sort_values(by='hcsTime').copy()
HCS_1996_2016_DF.reset_index(drop=True, inplace=True)
#print(HCS_1996_2016_DF)
#exit()

# Read HCS Data.
month2Num = {'Jan':1,'Feb':2,'Mar':3,'Apr':4,'May':5,'Jun':6,'Jul':7,'Aug':8,'Sep':9,'Oct':10,'Nov':11,'Dec':12}
Exhaust_DF = pd.read_csv(rootDir+inputExhaustDir+inputExhaustFile, header=0)
#print(month2Num[Exhaust_DF.loc[1,'Month']])
exhaustTime = Exhaust_DF.apply(lambda row: datetime(int(row['Year']), int(month2Num[row['Month']]), int(row['Day']), int(row['Hour']), int(row['Minute']), 0), axis=1)
Exhaust_DF.insert(0, 'exhaustTime', exhaustTime) # This command is able to specify column position.
Exhaust_DF.drop(['Year', 'Month', 'Day', 'Hour', 'Minute'],inplace=True,axis=1)
# Restrict the time range from 1996 to 2016. Actually, the range is only from 1997 to 2006.
Exhaust_1996_2016_DF = Exhaust_DF[(Exhaust_DF['exhaustTime']>=datetime(1996,1,1,0,0,0))&(Exhaust_DF['exhaustTime']<=datetime(2016,12,31,23,59,59))]
Exhaust_1996_2016_DF = Exhaust_1996_2016_DF.sort_values(by='exhaustTime').copy()
Exhaust_1996_2016_DF.reset_index(drop=True, inplace=True)
#print(Exhaust_1996_2016_DF)
#exit()

isVerbose = False

year_start = int(sys.argv[1])
year_end   = int(sys.argv[2])

B_threshold = ('(|B|>0)', '(|B|<5)', '(|B|>=5)')

isPlotDaysToHCS = 0
isPlotDaysToExhaust = 0
isPlot = isPlotDaysToHCS or isPlotDaysToExhaust

if isPlot:
    # This is the plot layout, 7 rows, 3 columns.
    max_row = 7
    max_column = 3

for B_lim in B_threshold:
#for B_lim in ['(|B|>=5)']:

    print('\nB_lim = {}'.format(B_lim))

    n_total = 0
    
    if isPlot:
        fig1,ax1 = plt.subplots(max_row, max_column, figsize=(8, 12))
        row = 0
        column = 0

    # Generate dataset with all |B| value.
    for year in xrange(year_start, year_end+1):
        
        year_str = str(year)
        inputFileName = year_str+'_detailed_info.p'

        # If outputWebDir does not exist, create it.
        if not os.path.exists(rootDir + outputDir):
            os.makedirs(rootDir + outputDir)

        #Load DataFrame
        FR_selected_events_DF = pd.read_pickle(open(rootDir + inputListDir + inputFileName,'rb'))
        #FR_selected_events_DF = pickle.load(open(rootDir + inputListDir + inputFileName,'rb')) # Not compatible with different pandas version.
        
        # Remove events with duration less than 10.
        #FR_selected_events_DF = FR_selected_events_DF[FR_selected_events_DF['duration']>=10].copy()

        # Apply Criteria No.1
        if B_lim == '(|B|>0)':
            FR_selected_events_DF = FR_selected_events_DF[FR_selected_events_DF['B_abs_mean']>=0]
        elif B_lim == '(|B|<5)':
            FR_selected_events_DF = FR_selected_events_DF[FR_selected_events_DF['B_abs_mean']<5]
        elif B_lim == '(|B|>=5)':
            FR_selected_events_DF = FR_selected_events_DF[FR_selected_events_DF['B_abs_mean']>=5]
        else:
            print('Apply Criteria No.1 error!')
            exit()
        FR_selected_events_DF.reset_index(drop=True, inplace=True)
        
        #n_total += len(FR_selected_events_DF)
        print('year = {}, counts = {}'.format(year, len(FR_selected_events_DF)))

        # Calculate scale size in AU.
        '''
        # Loop operation is very slow.
        for index_temp, record_temp in FR_selected_events_DF.iterrows():
            # Get X_unitVector.
            X_unitVector = np.array(record_temp[['X_unitVector[0]', 'X_unitVector[1]', 'X_unitVector[2]']])
            # Get Y_unitVector.
            Y_unitVector = np.array(record_temp[['Y_unitVector[0]', 'Y_unitVector[1]', 'Y_unitVector[2]']])
            # Get Z_unitVector.
            Z_unitVector = np.array(record_temp[['Z_unitVector[0]', 'Z_unitVector[1]', 'Z_unitVector[2]']])
            # Find flux rope frame.
            transToFRframe = np.array([X_unitVector, Y_unitVector, Z_unitVector]).T
            # Get VHT.
            VHT = np.array(record_temp[['VHT_inGSE[0]', 'VHT_inGSE[1]', 'VHT_inGSE[2]']])
            # Project VHT into FR Frame.
            VHT_inFR = VHT.dot(transToFRframe)
            # Get time duration
            duration = int(record_temp['duration'])*60.0 # Convert minutes to seconds.
            size = - VHT_inFR[0] * 1000.0 * duration # Space increment along X axis. Convert km/s to m/s.
            size_inAU = size/AU # Divided by Earth radius, 6371km.

            FR_selected_events_DF.loc[index_temp, 'size_inAU'] = size_inAU
        '''
        # Following is matrix operation, very fast.
        # Get Unit vectors for flux rope frame.
        X_unitVector = np.array(FR_selected_events_DF[['X_unitVector[0]', 'X_unitVector[1]', 'X_unitVector[2]']])
        Y_unitVector = np.array(FR_selected_events_DF[['Y_unitVector[0]', 'Y_unitVector[1]', 'Y_unitVector[2]']])
        Z_unitVector = np.array(FR_selected_events_DF[['Z_unitVector[0]', 'Z_unitVector[1]', 'Z_unitVector[2]']])
        # Construct transformation matrices.
        Matrix_transToFRframe = np.zeros((len(FR_selected_events_DF),3,3))
        Matrix_transToFRframe[:,:,0] = X_unitVector
        Matrix_transToFRframe[:,:,1] = Y_unitVector
        Matrix_transToFRframe[:,:,2] = Z_unitVector
        # Get VHT
        VHT = np.array(FR_selected_events_DF[['VHT_inGSE[0]', 'VHT_inGSE[1]', 'VHT_inGSE[2]']])
        VHT_inFR = np.zeros((len(FR_selected_events_DF),3))

        # Project VHT into FR Frame.
        for i in xrange(len(VHT)):
            VHT_inFR[i,:] = VHT[i,:].dot(Matrix_transToFRframe[i,:,:])
        # Get time duration
        duration = np.array(FR_selected_events_DF['duration'])*60.0 # Convert minutes to seconds.
        size = - VHT_inFR[:,0] * 1000.0 * duration # Space increment along X axis. Convert km/s to m/s.
        size_inAU_array = size/AU # Divided by AU.
        # Add one more attribute: size_inRE (flux rope cross-section size in earth radius.)
        FR_selected_events_DF.insert(4, 'size_inAU', size_inAU_array) # After insertion, the column index of size_inRE is 4.
        
        # Add one more attribute: wait_time (time interval between two flux ropes.)
        startTimeSeries = FR_selected_events_DF['startTime'].copy()
        endTimeSeries = FR_selected_events_DF['endTime'].copy()
        # Drop first record in startTime list.
        startTimeSeries.drop(startTimeSeries.index[[0]], inplace=True)
        # Reset index.
        startTimeSeries.reset_index(drop=True, inplace=True)
        # Drop last record in endTime list.
        endTimeSeries.drop(endTimeSeries.index[[-1]], inplace=True)
        # Reset index.
        endTimeSeries.reset_index(drop=True, inplace=True)
        # Calculate wait time.
        waitTime = startTimeSeries - endTimeSeries
        # Convert wait time to list.
        waitTimeList = []
        for record in waitTime:
            waitTime_temp = int(record.total_seconds()//60)
            waitTimeList.append(waitTime_temp)
        # Add np.nan as first element.
        waitTimeList = [np.nan] + waitTimeList
        # Add one new column 'waitTime' to dataframe
        #FR_detailed_info['waitTime'] = waitTimeList
        FR_selected_events_DF.insert(5, 'waitTime', waitTimeList) # This command is able to specify column position.
        #print('waitTime column is added.')
        
        # Add daysToHCS.
        FR_selected_events_DF.insert(6, 'daysToHCS', [np.nan]*len(FR_selected_events_DF))
        # Extend one month time boundary.
        if (year == year_start)&(year != year_end):
            HCS_currentYear_temp_DF = HCS_1996_2016_DF[(HCS_1996_2016_DF['hcsTime']>=datetime(year,1,1,0,0,0))&(HCS_1996_2016_DF['hcsTime']<=datetime(year+1,1,31,23,59,59))]
        elif (year != year_start)&(year == year_end):
            HCS_currentYear_temp_DF = HCS_1996_2016_DF[(HCS_1996_2016_DF['hcsTime']>=datetime(year-1,12,1,0,0,0))&(HCS_1996_2016_DF['hcsTime']<=datetime(year,12,31,23,59,59))]
        elif (year == year_start)&(year == year_end):
            HCS_currentYear_temp_DF = HCS_1996_2016_DF[(HCS_1996_2016_DF['hcsTime']>=datetime(year,1,1,0,0,0))&(HCS_1996_2016_DF['hcsTime']<=datetime(year,12,31,23,59,59))]
        elif (year != year_start)&(year != year_end):
            HCS_currentYear_temp_DF = HCS_1996_2016_DF[(HCS_1996_2016_DF['hcsTime']>=datetime(year-1,12,1,0,0,0))&(HCS_1996_2016_DF['hcsTime']<=datetime(year+1,1,31,23,59,59))]
        else:
            print('Something wrong, please check.')
        # Loop flux rope events list, calculate tempral distance of each event to HCS.
        for event_index in FR_selected_events_DF.index:
            if isVerbose:
                print('event_index = {}'.format(event_index))
            oneEventRecord = FR_selected_events_DF.loc[event_index] # Note the difference of iloc and loc.
            FluxRopeTurnTime = oneEventRecord['turnTime']
            #print(FluxRopeTurnTime)
            HCS_time_list = HCS_currentYear_temp_DF['hcsTime']
            #print(HCS_time_list)
            # Time difference in days(rounded). Negative value indicate the flux rope preceded the HCS.
            timeDiff_list = [round(x.total_seconds()/(24*3600), 0) for x in (FluxRopeTurnTime - HCS_time_list)]
            #print(timeDiff_list)
            daysToHCS = min(timeDiff_list, key=lambda x:abs(x))
            #print(daysToHCS)
            FR_selected_events_DF.set_value(event_index, 'daysToHCS', daysToHCS)
            #print(FR_selected_events_DF.loc[event_index])
                
        if isPlotDaysToHCS:
            # plot separate Pandas DataFrames as subplots.
            print('row = {}, column = {}'.format(row, column))
            FR_selected_events_DF['daysToHCS'].plot.hist(ax=ax1[row][column],bins=np.arange(-30.5,30.5), ylim=[0,900], color='#0077C8')
            ax1[row][column].set_xlim([-30,30])
            ax1[row][column].set_ylim([0,700]) # For B>=5
            ax1[row][column].set_xlabel(str(year))
            column += 1
            if column == max_column:
                row += 1
                column = 0
        
        # Add daysToExhaust.
        FR_selected_events_DF.insert(7, 'daysToExhaust', [np.nan]*len(FR_selected_events_DF))
        # Extend one month time boundary.
        if (year == year_start)&(year != year_end):
            Exhaust_currentYear_temp_DF = Exhaust_1996_2016_DF[(Exhaust_1996_2016_DF['exhaustTime']>=datetime(year,1,1,0,0,0))&(Exhaust_1996_2016_DF['exhaustTime']<=datetime(year+1,1,31,23,59,59))]
        elif (year != year_start)&(year == year_end):
            Exhaust_currentYear_temp_DF = Exhaust_1996_2016_DF[(Exhaust_1996_2016_DF['exhaustTime']>=datetime(year-1,12,1,0,0,0))&(Exhaust_1996_2016_DF['exhaustTime']<=datetime(year,12,31,23,59,59))]
        elif (year == year_start)&(year == year_end):
            Exhaust_currentYear_temp_DF = Exhaust_1996_2016_DF[(Exhaust_1996_2016_DF['exhaustTime']>=datetime(year,1,1,0,0,0))&(Exhaust_1996_2016_DF['exhaustTime']<=datetime(year,12,31,23,59,59))]
        elif (year != year_start)&(year != year_end):
            Exhaust_currentYear_temp_DF = Exhaust_1996_2016_DF[(Exhaust_1996_2016_DF['exhaustTime']>=datetime(year-1,12,1,0,0,0))&(Exhaust_1996_2016_DF['exhaustTime']<=datetime(year+1,1,31,23,59,59))]
        else:
            print('Something wrong, please check.')
        
        # If Exhaust_currentYear_temp_DF is not empty, Loop flux rope events list, calculate tempral distance of each event to Exhaust.
        if not Exhaust_currentYear_temp_DF.empty:
            for event_index in FR_selected_events_DF.index:
                if isVerbose:
                    print('event_index = {}'.format(event_index))
                oneEventRecord = FR_selected_events_DF.loc[event_index] # Note the difference of iloc and loc.
                FluxRopeTurnTime = oneEventRecord['turnTime']
                #print(FluxRopeTurnTime)
                Exhaust_time_list = Exhaust_currentYear_temp_DF['exhaustTime']
                #print(Exhaust_time_list)
                # Time difference in days(rounded). Negative value indicate the flux rope preceded the exhaust.
                timeDiff_list = [round(x.total_seconds()/(24*3600), 0) for x in (FluxRopeTurnTime - Exhaust_time_list)]
                #print(timeDiff_list)
                daysToExhaust = min(timeDiff_list, key=lambda x:abs(x))
                #print(daysToExhaust)
                FR_selected_events_DF.set_value(event_index, 'daysToExhaust', daysToExhaust)
                #print(FR_selected_events_DF.loc[event_index])
        
        #print(FR_selected_events_DF[['turnTime', 'daysToExhaust']])
        
        if isPlotDaysToExhaust:
            ax = FR_selected_events_DF['daysToExhaust'].plot.hist(bins=np.arange(-30.5,30.5), ylim=[0,400], color='#0077C8')
            ax.set_xlim([-30,30])
            ax.set_ylim([0,350]) # For B>=5
            fig = ax.get_figure()
            plt.title('Time to Reconnection Exhaust (Year '+str(year)+')')
            plt.xlabel('Day to Reconnection Exhaust')
            plt.ylabel('Flux Rope Occurrence Counts')
            fig.savefig('/Users/jz0006/Desktop/plot_temp/TimeToExhaust_'+str(year)+'_'+B_lim+'.eps', format='eps')
            plt.close('all')
        
        # Add shock information.
        FR_selected_events_DF.insert(8, 'afterShock', [False]*len(FR_selected_events_DF))
        FR_selected_events_DF.insert(9, 'shockTheta_deg', [np.nan]*len(FR_selected_events_DF))

        # Loop Shock List. Add info to FR_selected_events_DF.
        IPShock_WIND_currentYear_temp_DF = IPShock_WIND_1996_2016_DF[(IPShock_WIND_1996_2016_DF['shockTime']>=datetime(year,1,1,0,0,0))&(IPShock_WIND_1996_2016_DF['shockTime']<=datetime(year,12,31,23,59,59))]
        #IPShock_WIND_currentYear_temp_DF.reset_index(inplace=True, drop=True)

        #print(IPShock_WIND_currentYear_temp_DF)
        for shock_index in range(0, len(IPShock_WIND_currentYear_temp_DF)):
            if isVerbose:
                print('shock_index = {}'.format(shock_index))
            oneShockRecord = IPShock_WIND_currentYear_temp_DF.iloc[shock_index]
            shockStartTime = oneShockRecord['shockTime']
            shockEndTime = shockStartTime + timedelta(hours=12)
            shockTheta_deg = oneShockRecord['ShockTheta(degrees)']
            shockSpeedJump = oneShockRecord['SpeedJump(km/s)']
            shockProtonTemperatureRatio = oneShockRecord['ProtonTemperatureRatio']
            shockShockSpeed = oneShockRecord['ShockSpeed(km/s)']
            shockPlasmaBetaUpstream = oneShockRecord['PlasmaBetaUpstream']
            if isVerbose:
                print('shock time : {}'.format(shockStartTime))
        
            # Strange!! have to keep this.
            afterShockMask = ((FR_selected_events_DF.loc[:,['startTime']]>shockStartTime).values)&((FR_selected_events_DF.loc[:,['endTime']]<=shockEndTime).values)
            #print(afterShockMask)
            
            #print(FR_selected_events_DF.keys())

            index_tobe_set = FR_selected_events_DF[afterShockMask].index
            FR_selected_events_DF.set_value(index_tobe_set, 'afterShock', True)
            FR_selected_events_DF.set_value(index_tobe_set, 'shockTheta_deg', shockTheta_deg)
            FR_selected_events_DF.set_value(index_tobe_set, 'SpeedJump(km/s)', shockSpeedJump)
            FR_selected_events_DF.set_value(index_tobe_set, 'ProtonTemperatureRatio', shockProtonTemperatureRatio)
            FR_selected_events_DF.set_value(index_tobe_set, 'ShockSpeed(km/s)',  shockShockSpeed)
            FR_selected_events_DF.set_value(index_tobe_set, 'PlasmaBetaUpstream',  shockPlasmaBetaUpstream)

        # Save all events to pickle file.
        FR_selected_events_DF.to_pickle(rootDir + outputDir + B_lim + year_str + '_selected_events.p')
        #print(FR_selected_events_DF)
        # Pick the events after shock.
        FR_selected_events_DF = FR_selected_events_DF[FR_selected_events_DF['afterShock']==True]
        FR_selected_events_DF.reset_index(inplace=True, drop=True) # Reset index.
        # Save events after shock to pickle file.
        FR_selected_events_DF.to_pickle(rootDir + outputDir + B_lim + year_str + '_selected_events_afterShock.p')
    
        if isPlot:
            # Save Plot.
            # If plotFolder does not exist, create it.
            if not os.path.exists(rootDir + outputPlotDir):
                os.makedirs(rootDir + outputPlotDir)
            fig1.savefig(rootDir + outputPlotDir + 'TimeToHCS_'+B_lim+'.eps', format='eps')
            plt.close('all')

#print('n_total = {}'.format(n_total))
print('Done.')





