# Save the date used in webpage.
# Version 4.0. Combine plots into two plots. Loop all years.

import os
import sys
import glob
import pickle
import numpy as np
import pandas as pd
from datetime import datetime

print('\nCheck the python version on Mac OS X, /usr/local/bin/python should be used:')
os.system('which python')

year_start = int(sys.argv[1])
year_end   = int(sys.argv[2])

for year in xrange(year_start,year_end+1,1):
    print('\nyear = {}'.format(year))
    year_str = str(year)
    # Read year string from command line argument.
    #year_str = sys.argv[1]
    #year = int(year_str)
    # Specify input and output path
    rootDir = '/Users/jz0006/GoogleDrive/GS/'
    inputListDir = 'GS_SearchResult/selected_events/'
    inputFileName = year_str+'_selected_events_B_abs_mean>=5.p'
    outputDir = 'GS_WEB_html/'+year_str+'/events/'
    # If outputDir does not exist, create it.
    if not os.path.exists(rootDir + outputDir):
        os.makedirs(rootDir + outputDir)

    # Load web data pickle file.
    FR_web_data = pickle.load(open(rootDir + inputListDir + inputFileName,'rb'))
    eventList_length = len(FR_web_data)
    print('eventList_length = {}'.format(eventList_length))

    # Read and Check one year data.
    print('Reading data...')
    GS_oneYearData_DataFrame = pd.read_pickle(rootDir + 'GS_DataPickleFormat/preprocessed/GS_' + year_str + '_AllData_DataFrame_preprocessed.p')
    # Check data property.
    print('Checking DataFrame keys... {}'.format(GS_oneYearData_DataFrame.keys()))
    print('Checking DataFrame shape... {}'.format(GS_oneYearData_DataFrame.shape))
    print('Data Time start: {}'.format(GS_oneYearData_DataFrame.index[0]))
    print('Data Time end: {}'.format(GS_oneYearData_DataFrame.index[-1]))
    # Check the NaNs in each variable.
    print('Checking the number of NaNs in DataFrame...')
    len_GS_oneYearData_DataFrame = len(GS_oneYearData_DataFrame)
    for key in GS_oneYearData_DataFrame.keys():
        num_notNaN = GS_oneYearData_DataFrame[key].isnull().values.sum()
        percent_notNaN = 100.0 - num_notNaN * 100.0 / len_GS_oneYearData_DataFrame
        print('The number of NaNs in {} is {}, integrity is {}%'.format(key, num_notNaN, round(percent_notNaN, 2)))

    # Generate single webpage for one event.
    for i in range(eventList_length):
        print('i = {}'.format(i))
        # Prepare data for one event.
        currentRecord_DF = FR_web_data.iloc[[i]]
        # timeRange for current record.
        startTime = currentRecord_DF['startTime'].iloc[0]
        endTime = currentRecord_DF['endTime'].iloc[0]
        duration = currentRecord_DF['duration'].iloc[0]
        startTime_str = str(startTime.strftime('%Y%m%d%H%M'))
        endTime_str = str(endTime.strftime('%Y%m%d%H%M'))
        duration_str = str(int(duration))
        timeRange_str = startTime_str + '_' + endTime_str
        content_title = 'Flux Rope Event ' +  str(startTime.strftime('%Y-%m-%d %H:%M')) + ' ~ ' + str(endTime.strftime('%Y-%m-%d %H:%M')) + ' (' + duration_str + ' minutes)'
        #img01_name = timeRange_str + '.png'
        
        # Constructing webpage.
        html_filename = timeRange_str + '.html'
        html_webpage_file = open(rootDir + outputDir + html_filename,'w')
        f = html_webpage_file
        
        # Generate filename for previous and next webpage, used for link.
        if i==0: # First record.
            # Next record.
            nextRecord_DF = FR_web_data.iloc[[i+1]]
            # timeRange for next record.
            nextStartTime = nextRecord_DF['startTime'].iloc[0]
            nextEndTime = nextRecord_DF['endTime'].iloc[0]
            nextStartTime_str = str(nextStartTime.strftime('%Y%m%d%H%M'))
            nextEndTime_str = str(nextEndTime.strftime('%Y%m%d%H%M'))
            nextTimeRange_str = nextStartTime_str + '_' + nextEndTime_str
            # Next webpage filename.
            next_html_filename = nextTimeRange_str + '.html'
        elif i==(eventList_length-1): # Last record.
            # previous record.
            previousRecord_DF = FR_web_data.iloc[[i-1]]
            # timeRange for next record.
            previousStartTime = previousRecord_DF['startTime'].iloc[0]
            previousEndTime = previousRecord_DF['endTime'].iloc[0]
            previousStartTime_str = str(previousStartTime.strftime('%Y%m%d%H%M'))
            previousEndTime_str = str(previousEndTime.strftime('%Y%m%d%H%M'))
            previousTimeRange_str = previousStartTime_str + '_' + previousEndTime_str
            # Previous webpage filename.
            previous_html_filename = previousTimeRange_str + '.html'
        else: # Neither first nor last record.
            # previous record.
            previousRecord_DF = FR_web_data.iloc[[i-1]]
            # timeRange for next record.
            previousStartTime = previousRecord_DF['startTime'].iloc[0]
            previousEndTime = previousRecord_DF['endTime'].iloc[0]
            previousStartTime_str = str(previousStartTime.strftime('%Y%m%d%H%M'))
            previousEndTime_str = str(previousEndTime.strftime('%Y%m%d%H%M'))
            previousTimeRange_str = previousStartTime_str + '_' + previousEndTime_str
            # Previous webpage filename.
            previous_html_filename = previousTimeRange_str + '.html'
            # Next record.
            nextRecord_DF = FR_web_data.iloc[[i+1]]
            # timeRange for next record.
            nextStartTime = nextRecord_DF['startTime'].iloc[0]
            nextEndTime = nextRecord_DF['endTime'].iloc[0]
            nextStartTime_str = str(nextStartTime.strftime('%Y%m%d%H%M'))
            nextEndTime_str = str(nextEndTime.strftime('%Y%m%d%H%M'))
            nextTimeRange_str = nextStartTime_str + '_' + nextEndTime_str
            # Next webpage filename.
            next_html_filename = nextTimeRange_str + '.html'

        # html heading.
        f.write('<html>\n') # python will convert \n to os.linesep
        # Fixed link on top.
        f.write('<div style="width:100px; height:100px; position:fixed;">\n')
        f.write('<a href="../../index.html" style="font-family:arial; text-decoration:none; font-size:13;"> HOME </a> &nbsp;\n')
        f.write('<a href="../year' + year_str + '.html" style="font-family:arial; text-decoration:none; font-size:13;"> LIST </a>\n')
        f.write('<br />\n')
        if i==0: # First record.
            f.write('<a href="../year' + year_str + '.html" style="font-family:arial; text-decoration:none;"> &lsaquo;&ndash; </a> &nbsp;\n')
            f.write('<a href="' + next_html_filename + '" style="font-family:arial; text-decoration:none;"> &ndash;&rsaquo; </a>\n')
        elif i==(eventList_length-1): # Last record.
            f.write('<a href="' + previous_html_filename + '" style="font-family:arial; text-decoration:none;"> &lsaquo;&ndash; </a> &nbsp;\n')
            f.write('<a href="../year' + year_str + '.html" style="font-family:arial; text-decoration:none;"> &ndash;&rsaquo; </a> &nbsp;\n')
        else: # Neither first nor last record.
            f.write('<a href="' + previous_html_filename + '" style="font-family:arial; text-decoration:none;"> &lsaquo;&ndash; </a> &nbsp;\n')
            f.write('<a href="' + next_html_filename + '" style="font-family:arial; text-decoration:none;"> &ndash;&rsaquo; </a>\n')
        f.write('</div>\n')
        f.write('<br />\n')

        # Webpage head, showed on tab.
        f.write('<head>\n')
        f.write('<title>Flux Rope in ' + year_str + '</title>\n')
        f.write('</head>\n')

        # Open webpage body.
        f.write('<body>\n')
        # Content title
        f.write('<h2><center>' + content_title + '<center></h2>\n')
        f.write('<hr />\n')

        # Insert image.
        f.write('<center>\n')
        f.write('<img src="../images/' + timeRange_str + '.png' + '" alt="plot" width="900" height="1650" />\n')
        #f.write('<img src="../images/' + timeRange_str + '_timeSeries_plot.png' + '" alt="timeSeries_plot" width="900" height="1050" />\n')
        f.write('<center>\n')
        # Close body.
        f.write('</body>\n')
        # Close html.
        f.write('</html>\n')

        # Close file.
        f.close()
















