#!/usr/local/bin/python

'''
Given time range, plot detailed information of fluxrope. 
Do not trim data.
Previous version will check the number of turning points. From this version(v1.2), the task of checking will be finished by GS_getMoreFluxRopeInfo. The data 1996_detailed_info.p is already refined.
In Verion 1.3, put ploting process into loop. Produce plots in bulk.
Version 1.4. try small size pic format, pdf.
Version 1.4. turnPoint is read from detailed_info.
Version 1.5. Fix VA and V_remaining label in walen test plot.
Version 1.6. Add Hodogram.
Version 1.7. change pic format (png).
Version 3.3. Split plots panels.
Version 3.5. Merge plots into two to reduce file numbers.
Version 4.0. Merge plots into one to further reduce file numbers.

'''
from __future__ import division # Treat integer as float.
import os
import sys
import gc
import pickle
import numba
import numpy as np # Scientific calculation package.
from numpy import linalg as la
import pandas as pd
import scipy as sp
from scipy.signal import savgol_filter # Savitzky-Golay filter
from scipy import integrate
from scipy import stats
from datetime import datetime # Import datetime class from datetime package.
from datetime import timedelta
import matplotlib
import matplotlib.pyplot as plt # Plot package, make matlab style plots.
from matplotlib.ticker import AutoMinorLocator
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import FuncFormatter
import matplotlib.dates as mdates
from scipy import stats
import multiprocessing
import time
sys.path.append('/Users/jz0006/GoogleDrive/GS/GS_FluxRopeDetectionPackage')
import MyPythonPackage.fluxrope as FR

###############################################################################

# Calculate the velocity of deHoffmann-Teller frame, VHT.
def findVHT(B_DF_inGSE, Vsw_DF_inGSE):
    N = len(B_DF_inGSE)
    B_square = np.square(B_DF_inGSE).sum(axis=1) # Take squre and sum row (axis=1 for row, axis=0 for column)
    KN = np.zeros((N,3,3)) # np.zeros((layer, row, column)). Right most index change first.
    for n in range(N):
        for i in range(3):
            for j in range(3):
                if i == j:
                    KN[n,i,j] = B_square.iloc[n] - B_DF_inGSE.iloc[n][i] * B_DF_inGSE.iloc[n][j]
                else:
                    KN[n,i,j] = - B_DF_inGSE.iloc[n][i] * B_DF_inGSE.iloc[n][j]
    K = np.mean(KN, axis=0) # Take average on layer (axis=1 for row, axis=2 for column).
    KVN = np.zeros((N,3)) # np.zeros((row, column)). Right most index change first.
    for n in range(N):
        KVN[n,:] = np.dot(KN[n,:], Vsw_DF_inGSE.iloc[n])
    # Average KVN over N to get KV.
    KV = np.mean(KVN, axis=0) # Take average on column.
    VHT = np.dot(np.linalg.inv(K), KV)
    return VHT

#############################################################################################################


def plotFluxRopeInfo(oneRecord_DF):
    # Set plot flag.
    '''
    isPlot_TransversePressure = True
    isPlot_Hodogram_B2_B1 = True
    isPlot_Hodogram_B3_B1 = True
    isPlot_Hodogram_Bz_By = True
    isPlot_Hodogram_Bx_By = True
    isPlot_WalenTest = True
    isPlot_B_inGSE = True
    isPlot_B_inFR = True
    isPlot_Vsw_inGSE = True
    isPlot_Np = True
    isPlot_Tp = True
    isPlot_Beta = True
    isPlot_Beta_p = True
    isPlot_TotalPressure = True
    isPlot_Te_to_Tp = True
    isPlot_CounterstreamingElectrons = True
    isPlot_EnergeticParticles = False
    isPlot_PitchAngle = True
    '''
    
    # Prepare data for one event.
    #oneRecord_DF = FR_web_data_specificTimeInterval.iloc[[i]]
    # timeRange
    startTime = oneRecord_DF['startTime'].iloc[0]
    turnTime = oneRecord_DF['turnTime'].iloc[0]
    endTime = oneRecord_DF['endTime'].iloc[0]
    duration = int(oneRecord_DF['duration'].iloc[0])
    residue_diff = oneRecord_DF['residue_diff'].iloc[0]
    residue_fit = oneRecord_DF['residue_fit'].iloc[0]
    theta_deg = int(oneRecord_DF['theta_deg'].iloc[0])
    phi_deg = int(oneRecord_DF['phi_deg'].iloc[0])
    VHT_inGSE = np.array([oneRecord_DF['VHT_inGSE[0]'].iloc[0], oneRecord_DF['VHT_inGSE[1]'].iloc[0], oneRecord_DF['VHT_inGSE[2]'].iloc[0]])
    
    X_unitVector = np.array([oneRecord_DF['X_unitVector[0]'].iloc[0], oneRecord_DF['X_unitVector[1]'].iloc[0], oneRecord_DF['X_unitVector[2]'].iloc[0]])
    Y_unitVector = np.array([oneRecord_DF['Y_unitVector[0]'].iloc[0], oneRecord_DF['Y_unitVector[1]'].iloc[0], oneRecord_DF['Y_unitVector[2]'].iloc[0]])
    Z_unitVector = np.array([oneRecord_DF['Z_unitVector[0]'].iloc[0], oneRecord_DF['Z_unitVector[1]'].iloc[0], oneRecord_DF['Z_unitVector[2]'].iloc[0]])
    walenTest_slope = oneRecord_DF['walenTest_slope'].iloc[0]
    walenTest_intercept = oneRecord_DF['walenTest_intercept'].iloc[0]
    walenTest_r_value = oneRecord_DF['walenTest_r_value'].iloc[0]
    B_abs_mean = oneRecord_DF['B_abs_mean'].iloc[0]
    Bx_abs_mean = oneRecord_DF['Bx_abs_mean'].iloc[0]
    By_abs_mean = oneRecord_DF['By_abs_mean'].iloc[0]
    Bz_abs_mean = oneRecord_DF['Bz_abs_mean'].iloc[0]
    B_std = oneRecord_DF['B_std'].iloc[0]
    Bx_std = oneRecord_DF['Bx_std'].iloc[0]
    By_std = oneRecord_DF['By_std'].iloc[0]
    Bz_std = oneRecord_DF['Bz_std'].iloc[0]
    B_magnitude_max = oneRecord_DF['B_magnitude_max'].iloc[0]
    Vsw_magnitude_mean = oneRecord_DF['Vsw_magnitude_mean'].iloc[0]
    Tp_mean = oneRecord_DF['Tp_mean'].iloc[0]
    Beta_mean = oneRecord_DF['Beta_mean'].iloc[0]
    Beta_p_mean = oneRecord_DF['Beta_p_mean'].iloc[0]
    lambda1 = oneRecord_DF['lambda1'].iloc[0]
    lambda2 = oneRecord_DF['lambda2'].iloc[0]
    lambda3 = oneRecord_DF['lambda3'].iloc[0]
    eigenVectorMaxVar_lambda1 = np.array([oneRecord_DF['eigenVectorMaxVar_lambda1[0]'].iloc[0], oneRecord_DF['eigenVectorMaxVar_lambda1[1]'].iloc[0], oneRecord_DF['eigenVectorMaxVar_lambda1[2]'].iloc[0]])
    eigenVectorInterVar_lambda2 = np.array([oneRecord_DF['eigenVectorInterVar_lambda2[0]'].iloc[0], oneRecord_DF['eigenVectorInterVar_lambda2[1]'].iloc[0], oneRecord_DF['eigenVectorInterVar_lambda2[2]'].iloc[0]])
    eigenVectorMinVar_lambda3 = np.array([oneRecord_DF['eigenVectorMinVar_lambda3[0]'].iloc[0], oneRecord_DF['eigenVectorMinVar_lambda3[1]'].iloc[0], oneRecord_DF['eigenVectorMinVar_lambda3[2]'].iloc[0]])
    
    startTime_str = str(startTime.strftime('%Y%m%d%H%M'))
    #turnTime_str = str(turnTime.strftime('%Y-%m-%d %H:%M'))
    endTime_str = str(endTime.strftime('%Y%m%d%H%M'))
    duration_str = str(duration)
    timeRange_str = startTime_str + '_' + endTime_str
    theta_deg_str = str(theta_deg)
    phi_deg_str = str(phi_deg)
    residue_fit_str = str(round(residue_fit,4))

    # Grab data in specific range.
    selectedRange_mask = (GS_oneYearData_DataFrame.index >= startTime) & (GS_oneYearData_DataFrame.index <= endTime)
    # The data of one fluxrope.
    GS_fluxRopeCandidate_DF = GS_oneYearData_DataFrame.iloc[selectedRange_mask]
    # The pitch angle data of one fluxrope.
    GS_fluxRopeCandidate_PitchAngle_DF = GS_PitchAngle_DataFrame.iloc[selectedRange_mask]

    matrix_transToMVAB = np.array([eigenVectorMaxVar_lambda1, eigenVectorInterVar_lambda2, eigenVectorMinVar_lambda3]).T
    matrix_transToFR = np.array([X_unitVector, Y_unitVector, Z_unitVector]).T
    #print(matrix_transToMVAB)
    #print(matrix_transToFR)

    # Keys: Index([u'Bx', u'By', u'Bz', u'Vx', u'Vy', u'Vz', u'Np', u'Tp', u'Te'], dtype='object')
    # Get Magnetic field slice.
    B_inGSE = GS_fluxRopeCandidate_DF.ix[:,['Bx', 'By', 'Bz']] # Produce a reference.
    # Get the solar wind slice.
    Vsw_inGSE = GS_fluxRopeCandidate_DF.ix[:,['Vx', 'Vy', 'Vz']] # Produce a reference.
    # Get the proton number density slice.
    Np_inGSE = GS_fluxRopeCandidate_DF.ix[:,['Np']] # Produce a reference.
    # Get the proton temperature slice. In Kelvin.
    Tp_inGSE = GS_fluxRopeCandidate_DF.ix[:,['Tp']] # Produce a reference.
    # Convert thermal speed to 10^6 Kelvein. Thermal speed is in km/s. Vth = sqrt(2KT/M)
    Tp_MK = Tp_inGSE.copy(deep=True)/1e6
    #Tp_MK['Tp'] = m_proton*np.square(np.array(Tp_MK['Tp'])*1e3)/(2.0*k_Boltzmann)/1e6
    if 'Te' in GS_fluxRopeCandidate_DF.keys():
        # Get the electron temperature slice. In Kelvin.
        Te_inGSE = GS_fluxRopeCandidate_DF.ix[:,['Te']] # Produce a reference.
        # Convert Te to 10^6 Kelvien.
        Te_MK = Te_inGSE.copy(deep=True)
        Te_MK['Te'] = np.array(Te_MK['Te'])/1e6
    
    # If there is any NaN in B_inGSE, try to interpolate.
    if B_inGSE.isnull().values.sum():
        print('Found NaNs, interpolate B.')
        B_inGSE_copy = B_inGSE.copy(deep=True)
        # limit=3 means only interpolate the gap shorter than 4.
        B_inGSE_copy.interpolate(method='time', limit=None, inplace=True)
        # interpolate won't fill leading NaNs, so we use backward fill.
        B_inGSE_copy.bfill(inplace=True)
        B_inGSE_copy.ffill(inplace=True)
        if B_inGSE_copy.isnull().values.sum():
            print('Too many NaNs in B. Skip this record. If this situation happens, please check.')
        else:
            B_inGSE = B_inGSE_copy

    # If there is any NaN in Vsw_inGSE, try to interpolate.
    if Vsw_inGSE.isnull().values.sum():
        print('Found NaNs, interpolate Vsw.')
        Vsw_inGSE_copy = Vsw_inGSE.copy(deep=True)
        # limit=3 means only interpolate the gap shorter than 4.
        Vsw_inGSE_copy.interpolate(method='time', limit=None, inplace=True)
        # interpolate won't fill leading NaNs, so we use backward fill.
        Vsw_inGSE_copy.bfill(inplace=True)
        Vsw_inGSE_copy.ffill(inplace=True)
        if Vsw_inGSE_copy.isnull().values.sum():
            print('Too many NaNs in Vsw. Skip this record. If this situation happens, please check.')
        else:
            Vsw_inGSE = Vsw_inGSE_copy
            
    # If there is any NaN in Np_inGSE, try to interpolate.
    if Np_inGSE.isnull().values.sum():
        print('Found NaNs, interpolate Np.')
        Np_inGSE_copy = Np_inGSE.copy(deep=True)
        # limit=3 means only interpolate the gap shorter than 4.
        Np_inGSE_copy.interpolate(method='time', limit=None, inplace=True)
        # interpolate won't fill leading NaNs, so we use backward fill.
        Np_inGSE_copy.bfill(inplace=True)
        Np_inGSE_copy.ffill(inplace=True)
        if Np_inGSE_copy.isnull().values.sum():
            print('Too many NaNs in Vsw. Skip this record. If this situation happens, please check.')
        else:
            Np_inGSE = Np_inGSE_copy
            
    # If there is any NaN in Tp_MK, try to interpolate.
    isOK_Tp_data = True
    if Tp_MK.isnull().values.sum():
        print('Found NaNs, interpolate Np.')
        Tp_MK_copy = Tp_MK.copy(deep=True)
        # limit=3 means only interpolate the gap shorter than 4.
        Tp_MK_copy.interpolate(method='time', limit=None, inplace=True)
        # interpolate won't fill leading NaNs, so we use backward fill.
        Tp_MK_copy.bfill(inplace=True)
        Tp_MK_copy.ffill(inplace=True)
        if Tp_MK_copy.isnull().values.sum():
            print('Too many NaNs in Tp_MK. Fill NaN with -1.')
            Tp_MK_copy.fillna(value=-1, inplace=True) # To avoid plot error.
            isOK_Tp_data = False
        Tp_MK = Tp_MK_copy

    # If there is any NaN in Tp_MK, try to interpolate.
    isOK_Te_data = False
    if 'Te' in GS_fluxRopeCandidate_DF.keys():
        isOK_Te_data = True
        if Te_MK.isnull().values.sum():
            print('Found NaNs, interpolate Te.')
            Te_MK_copy = Te_MK.copy(deep=True)
            # limit=3 means only interpolate the gap shorter than 4.
            Te_MK_copy.interpolate(method='time', limit=None, inplace=True)
            # interpolate won't fill leading NaNs, so we use backward fill.
            Te_MK_copy.bfill(inplace=True)
            Te_MK_copy.ffill(inplace=True)
            if Te_MK_copy.isnull().values.sum():
                print('Too many NaNs in Te_MK. Skip this record. If this situation happens, please check.')
                isOK_Te_data = False
            else:
                Te_MK = Te_MK_copy

    #VHT_inGSE = findVHT(B_inGSE, Vsw_inGSE)
    # Project B_inGSE into FluxRope Frame.
    B_inFR = B_inGSE.dot(matrix_transToFR)
    # Project VHT_inGSE into FluxRope Frame.
    VHT_inFR = VHT_inGSE.dot(matrix_transToFR)
    # Project Vsw_inFR into FluxRope Frame.
    Vsw_inFR = Vsw_inGSE.dot(matrix_transToFR)

    B_inMVAB = B_inGSE.dot(matrix_transToMVAB)

    # Calculate A(x,0) by integrating By. A(x,0) = Integrate[-By(s,0)ds, {s, 0, x}], where ds = -Vht dot unit(X) dt.
    ds = - VHT_inFR[0] * 1000.0 * dt # Space increment along X axis. Convert km/s to m/s.
    # Don't forget to convert km/s to m/s, and convert nT to T. By = B_inFR[1]
    A = integrate.cumtrapz(-B_inFR[1]*1e-9, dx=ds, initial=0)
    # Calculate Pt(x,0). Pt(x,0)=p(x,0)+Bz^2/(2mu0). By = B_inFR[2]
    Pt = np.array((B_inFR[2] * 1e-9)**2 / (2.0*mu0) * 1e9) # 1e9 convert unit form pa to npa.
    # Find the index of turnPoint.
    index_turnTime = B_inFR.index.get_loc(turnTime)
    # Split A and Pt into two branches.
    A_sub1 = A[:index_turnTime+1]
    A_sub2 = A[index_turnTime:]
    Pt_sub1 = Pt[:index_turnTime+1]
    Pt_sub2 = Pt[index_turnTime:]
    
    z = np.polyfit(A, Pt, 3)
    Func_Pt_A = np.poly1d(z)

    startTime_str = str(startTime.strftime('%Y%m%d%H%M'))
    #turnTime_str = str(turnTime.strftime('%Y-%m-%d %H:%M'))
    endTime_str = str(endTime.strftime('%Y%m%d%H%M'))
    timeRange_str = startTime_str + '_' + endTime_str
    theta_deg_str = str(theta_deg)
    phi_deg_str = str(phi_deg)
    residue_fit_str = str(round(residue_fit,4))

    # Print information on screen.r'$\alpha > \beta$'
    time_str = str(startTime.strftime('%Y-%m-%d %H:%M'))+' ~ '+str(endTime.strftime('%Y-%m-%d %H:%M'))
    orientation_str = '('+theta_deg_str+','+phi_deg_str+')'
    fig_title = time_str+ '  ' + 'duration = ' + duration_str + ' minutes ' + ' (' + r'$\theta$' + ',' + r'$\phi$' + ') = ' + orientation_str + '  Residue=' + residue_fit_str
    #print('\nPlotting Flux Rope...')
    print('Time Range: {}'.format(time_str))
    #print('\n')
    
    # Plot.
    # Format defination.
    fig_formatter = mdates.DateFormatter('%m/%d %H:%M') # Full format is ('%Y-%m-%d %H:%M:%S').
    fig_title_fontsize = 11
    fig_xlabel_fontsize = 8
    fig_ylabel_fontsize = 8
    fig_xtick_fontsize = 8
    fig_ytick_fontsize = 8
    fig_linewidth = 1
    
    fig_dpi = 150
    fig_format = 'png'
    
    fig1,ax1 = plt.subplots(2, 3, figsize=(12, 8))
    # Plot Transverse Pressure (plot_Pt_vs_A).
    plot_Pt_vs_A = ax1[0][0]
    plot_Pt_vs_A.set_title('Transverse pressure', fontsize=fig_title_fontsize)
    plot_Pt_vs_A.plot(A_sub1, Pt_sub1, marker='o', color='r' ,label='1st half path')
    plot_Pt_vs_A.plot(A_sub2, Pt_sub2, marker='^', color='b' ,label='2nd half path')
    plot_Pt_vs_A.set_ylim(bottom=0)
    x_min, x_max = plot_Pt_vs_A.get_xlim()
    y_min, y_max = plot_Pt_vs_A.get_ylim()
    plot_Pt_vs_A.set_aspect(aspect=str((x_max-x_min)/(y_max-y_min)))
    plot_Pt_vs_A.set_xlabel(r'A(T$\cdot$m)',fontsize=fig_xlabel_fontsize)
    plot_Pt_vs_A.set_ylabel('Pt(nPa)',fontsize=fig_ylabel_fontsize)
    plot_Pt_vs_A.tick_params(axis='x', labelsize=fig_xtick_fontsize) # Tick font size.
    plot_Pt_vs_A.tick_params(axis='y', labelsize=fig_ytick_fontsize) # Tick font size.
    plot_Pt_vs_A.xaxis.set_major_locator(MaxNLocator(8))
    plot_Pt_vs_A.yaxis.set_major_locator(MaxNLocator(10))
    plot_Pt_vs_A.legend(loc='upper center',prop={'size':7})
    
    # Plot Hodogram_B2_B1.
    plot_Hodogram_B2_B1 = ax1[0][1]
    plot_Hodogram_B2_B1.set_aspect('equal')
    plot_Hodogram_B2_B1.set_title('Hodogram(B2_B1 MVAB frame)', fontsize=fig_title_fontsize)
    plot_Hodogram_B2_B1.plot(B_inMVAB[1], B_inMVAB[0], color = 'blue', marker='.')
    x_min, x_max = plot_Hodogram_B2_B1.get_xlim()
    y_min, y_max = plot_Hodogram_B2_B1.get_ylim()
    x_range = x_max - x_min
    y_range = y_max - y_min
    if y_range > x_range:
        axis_extension = (y_range - x_range)/2.0
        x_min_new = x_min - axis_extension
        x_max_new = x_max + axis_extension
        plot_Hodogram_B2_B1.set_xlim([x_min_new, x_max_new])
    elif y_range < x_range:
        axis_extension = (x_range - y_range)/2.0
        y_min_new = y_min - axis_extension
        y_max_new = y_max + axis_extension
        plot_Hodogram_B2_B1.set_ylim([y_min_new, y_max_new])
    plot_Hodogram_B2_B1.set_xlabel('B2(nT)',fontsize=fig_xlabel_fontsize)
    plot_Hodogram_B2_B1.set_ylabel('B1(nT)',fontsize=fig_ylabel_fontsize)
    plot_Hodogram_B2_B1.tick_params(axis='x', labelsize=fig_xtick_fontsize) # Tick font size.
    plot_Hodogram_B2_B1.tick_params(axis='y', labelsize=fig_ytick_fontsize) # Tick font size.

    # Plot Hodogram_B3_B1.
    plot_Hodogram_B3_B1 = ax1[0][2]
    plot_Hodogram_B3_B1.set_aspect('equal', adjustable='box')
    plot_Hodogram_B3_B1.set_title('Hodogram(B3_B1 MVAB frame)', fontsize=fig_title_fontsize)
    plot_Hodogram_B3_B1.plot(B_inMVAB[2], B_inMVAB[0], color = 'blue', marker='.')
    x_min, x_max = plot_Hodogram_B3_B1.get_xlim()
    y_min, y_max = plot_Hodogram_B3_B1.get_ylim()
    x_range = x_max - x_min
    y_range = y_max - y_min
    if y_range > x_range:
        axis_extension = (y_range - x_range)/2.0
        x_min_new = x_min - axis_extension
        x_max_new = x_max + axis_extension
        plot_Hodogram_B3_B1.set_xlim([x_min_new, x_max_new])
    elif y_range < x_range:
        axis_extension = (x_range - y_range)/2.0
        y_min_new = y_min - axis_extension
        y_max_new = y_max + axis_extension
        plot_Hodogram_B3_B1.set_ylim([y_min_new, y_max_new])
    plot_Hodogram_B3_B1.set_xlabel('B3(nT)',fontsize=fig_xlabel_fontsize)
    plot_Hodogram_B3_B1.set_ylabel('B1(nT)',fontsize=fig_ylabel_fontsize)
    plot_Hodogram_B3_B1.tick_params(axis='x', labelsize=fig_xtick_fontsize) # Tick font size.
    plot_Hodogram_B3_B1.tick_params(axis='y', labelsize=fig_ytick_fontsize) # Tick font size.
        
    # Plot walen test.
    plot_walenTest = ax1[1][0]
    plot_walenTest.set_aspect('equal', adjustable='box')
    # Proton mass density. Original Np is in #/cc ( cc = cubic centimeter). Multiply by 1e6 to convert cc to m^3.
    P_massDensity = Np_inGSE * m_proton * 1e6 # In kg/m^3.
    len_P_massDensity = len(P_massDensity)
    P_massDensity_array = np.array(P_massDensity)
    P_massDensity_array = np.reshape(P_massDensity_array, (len_P_massDensity, 1))
    # Alfven speed. Multiply by 1e-9 to convert nT to T. Divided by 1000.0 to convert m/s to km/s.
    VA_inFR = np.array(B_inFR * 1e-9) / np.sqrt(mu0 * P_massDensity_array) / 1000.0
    VA_inFR_1D = np.reshape(VA_inFR, VA_inFR.size)
    V_remaining = np.array(Vsw_inFR - VHT_inFR)
    V_remaining_1D = np.reshape(V_remaining, V_remaining.size)
    # Plot plot_walenTest.
    plot_walenTest.set_title('WalenTest', fontsize=fig_title_fontsize)
    # plot_walenTest.scatter(x, y, color='r')
    plot_walenTest.scatter(VA_inFR[:,0], V_remaining[:,0], color='r')
    plot_walenTest.scatter(VA_inFR[:,1], V_remaining[:,1], color='g')
    plot_walenTest.scatter(VA_inFR[:,2], V_remaining[:,2], color='b')
    # Set axis range.
    VA_inFR_1D = np.reshape(VA_inFR, VA_inFR.size)
    V_remaining_1D = np.reshape(V_remaining, V_remaining.size)
    x_min = np.nanmin([np.nanmin(VA_inFR_1D), np.nanmin(V_remaining_1D)])
    x_max = np.nanmax([np.nanmax(VA_inFR_1D), np.nanmax(V_remaining_1D)])
    y_min = x_min
    y_max = x_max
    if ~(np.isnan(x_min)&np.isnan(x_max)&np.isnan(y_min)&np.isnan(y_max)):
        plot_walenTest.set_xlim(x_min, x_max)
        plot_walenTest.set_ylim(y_min, y_max)
    else:
        x_min, x_max = plot_walenTest.get_xlim()
        y_min, y_max = plot_walenTest.get_ylim()
    # Fitted line start and end point.
    walenTestFittedLine_start_x = x_min
    walenTestFittedLine_start_y = walenTest_slope * x_min + walenTest_intercept
    walenTestFittedLine_end_x = x_max
    walenTestFittedLine_end_y = walenTest_slope * x_max + walenTest_intercept
    plot_walenTest.plot([walenTestFittedLine_start_x,walenTestFittedLine_end_x],\
    [walenTestFittedLine_start_y,walenTestFittedLine_end_y], color='black', linestyle='dashed')
    plot_walenTest.set_ylabel('V$_{remaining}$(km/s)',fontsize=fig_ylabel_fontsize)
    plot_walenTest.set_xlabel('V$_A$(km/s)',fontsize=fig_xlabel_fontsize)
    plot_walenTest.tick_params(axis='x', labelsize=fig_xtick_fontsize) # Tick font size.
    plot_walenTest.tick_params(axis='y', labelsize=fig_ytick_fontsize) # Tick font size.
    textInBox_temp = 'slope = '+str(round(walenTest_slope,3))+'\n'+'$r$ = '+str(round(walenTest_r_value,3))
    plot_walenTest.text(x_min+(x_max-x_min)*0.025, y_max-(y_max-y_min)*0.025,  \
    textInBox_temp, fontsize = 9, horizontalalignment='left', verticalalignment='top', \
    bbox={'boxstyle':'square','pad':0.2, 'facecolor':'white'})

    # Plot Hodogram_Bz_By:
    plot_Hodogram_Bz_By = ax1[1][1]
    plot_Hodogram_Bz_By.set_aspect('equal', adjustable='box')
    plot_Hodogram_Bz_By.set_title('Hodogram(Bz_By FR frame)', fontsize=fig_title_fontsize)
    plot_Hodogram_Bz_By.plot(B_inFR[2], B_inFR[1], color = 'blue', marker='.')
    x_min, x_max = plot_Hodogram_Bz_By.get_xlim()
    y_min, y_max = plot_Hodogram_Bz_By.get_ylim()
    x_range = x_max - x_min
    y_range = y_max - y_min
    if y_range > x_range:
        axis_extension = (y_range - x_range)/2.0
        x_min_new = x_min - axis_extension
        x_max_new = x_max + axis_extension
        plot_Hodogram_Bz_By.set_xlim([x_min_new, x_max_new])
    elif y_range < x_range:
        axis_extension = (x_range - y_range)/2.0
        y_min_new = y_min - axis_extension
        y_max_new = y_max + axis_extension
        plot_Hodogram_Bz_By.set_ylim([y_min_new, y_max_new])
    plot_Hodogram_Bz_By.set_xlabel('Bz(nT)',fontsize=fig_xlabel_fontsize)
    plot_Hodogram_Bz_By.set_ylabel('By(nT)',fontsize=fig_ylabel_fontsize)
    plot_Hodogram_Bz_By.tick_params(axis='x', labelsize=fig_xtick_fontsize) # Tick font size.
    plot_Hodogram_Bz_By.tick_params(axis='y', labelsize=fig_ytick_fontsize) # Tick font size.

    # Plot Hodogram_Bx_By:
    plot_Hodogram_Bx_By = ax1[1][2]
    plot_Hodogram_Bx_By.set_aspect('equal', adjustable='box')
    plot_Hodogram_Bx_By.set_title('Hodogram(Bx_By FR frame)', fontsize=fig_title_fontsize)
    plot_Hodogram_Bx_By.plot(B_inFR[0], B_inFR[1], color = 'blue', marker='.')
    x_min, x_max = plot_Hodogram_Bx_By.get_xlim()
    y_min, y_max = plot_Hodogram_Bx_By.get_ylim()
    x_range = x_max - x_min
    y_range = y_max - y_min
    if y_range > x_range:
        axis_extension = (y_range - x_range)/2.0
        x_min_new = x_min - axis_extension
        x_max_new = x_max + axis_extension
        plot_Hodogram_Bx_By.set_xlim([x_min_new, x_max_new])
    elif y_range < x_range:
        axis_extension = (x_range - y_range)/2.0
        y_min_new = y_min - axis_extension
        y_max_new = y_max + axis_extension
        plot_Hodogram_Bx_By.set_ylim([y_min_new, y_max_new])
    plot_Hodogram_Bx_By.set_xlabel('Bx(nT)',fontsize=fig_xlabel_fontsize)
    plot_Hodogram_Bx_By.set_ylabel('By(nT)',fontsize=fig_ylabel_fontsize)
    plot_Hodogram_Bx_By.tick_params(axis='x', labelsize=fig_xtick_fontsize) # Tick font size.
    plot_Hodogram_Bx_By.tick_params(axis='y', labelsize=fig_ytick_fontsize) # Tick font size.

    left_margin = 0.065    # the left side of the subplots of the figure
    right_margin = 0.935   # the right side of the subplots of the figure
    top_margin = 0.95     # the top of the subplots of the figure
    bottom_margin = 0.05  # the bottom of the subplots of the figure
    width_gap = 0.225       # the amount of width reserved for blank space between subplots
    height_gap = 0.15       # the amount of height reserved for white space between subplots
    plt.subplots_adjust(left=left_margin, bottom=bottom_margin, right=right_margin, top=top_margin, wspace=width_gap, hspace=height_gap)

    # Save plot. Supported formats: emf, eps, pdf, png, ps, raw, rgba, svg, svgz.
    grid_fig_filename = rootDir + outputDir + timeRange_str + '_' + 'grid_plot' + '.' + fig_format
    print(grid_fig_filename)
    fig1.savefig(grid_fig_filename, format=fig_format, dpi=fig_dpi)
    #plt.close(fig1)


    # Plot time series data.
    fig2,ax2 = plt.subplots(7,1, figsize=(12, 14))
    plot_B_GSE = ax2[0]
    plot_B_FR = ax2[1]
    e_pitch = ax2[2]
    plot_Vsw_GSE = ax2[3]
    plot_Beta_p = ax2[4]
    plot_Np = ax2[5]
    plot_Tp = ax2[6]

    # Plot plot_B_GSE.
    plot_B_GSE.set_title('Magnetic Field (GSE Frame)', fontsize=fig_title_fontsize)
    plot_B_GSE.xaxis.set_major_formatter(fig_formatter) # Set xlabel format.
    plot_B_GSE.plot(B_inGSE.index, B_inGSE['Bx'],\
                    color = 'red',label='Bx')
    plot_B_GSE.plot(B_inGSE.index, B_inGSE['By'],\
                    color = 'green', label='By')
    plot_B_GSE.plot(B_inGSE.index, B_inGSE['Bz'],\
                    color = 'blue', label='Bz')
    plot_B_GSE.axhline(0, color='black',linewidth=0.5,linestyle='dashed') # Zero line, must placed after data plot
    #plot_B_GSE.set_xlabel('Time',fontsize=fig_xlabel_fontsize)
    plot_B_GSE.set_ylabel('B(nT)',fontsize=fig_ylabel_fontsize)
    plot_B_GSE.tick_params(axis='x', labelsize=fig_xtick_fontsize) # Tick font size.
    plot_B_GSE.tick_params(axis='y', labelsize=fig_ytick_fontsize) # Tick font size.
    plot_B_GSE.legend(loc='upper right',prop={'size':7})
    #plt.subplots_adjust(left=left_margin_coord, right=right_margin_coord, top=top_margin_coord, bottom=bottom_margin_coord)

    # Plot plot_B_FR.
    plot_B_FR.set_title('Magnetic Field (Flux Rope Frame)', fontsize=fig_title_fontsize)
    plot_B_FR.xaxis.set_major_formatter(fig_formatter) # Set xlabel format.
    plot_B_FR.plot(B_inFR.index, B_inFR[0], color = 'red', label='Bx\'') # Minimum vairance.
    plot_B_FR.plot(B_inFR.index, B_inFR[1], color = 'green', label='By\'') # Maximum variance.
    plot_B_FR.plot(B_inFR.index, B_inFR[2], color = 'blue', label='Bz\'') # Intemediate variance.
    Btotal_fluxRope_inOptimalFrame = (B_inFR[0]**2 + B_inFR[1]**2 + B_inFR[2]**2)**0.5
    plot_B_FR.plot(Btotal_fluxRope_inOptimalFrame.index, Btotal_fluxRope_inOptimalFrame, color = 'black',label='|B|')
    plot_B_FR.axhline(0, color='black',linewidth=0.5,linestyle='dashed') # Zero line, must placed after data plot
    #plot_B_FR.set_xlabel('Time',fontsize=fig_xlabel_fontsize)
    plot_B_FR.set_ylabel('B(nT)',fontsize=fig_ylabel_fontsize)
    plot_B_FR.tick_params(axis='x', labelsize=fig_xtick_fontsize) # Tick font size.
    plot_B_FR.tick_params(axis='y', labelsize=fig_ytick_fontsize) # Tick font size.
    plot_B_FR.legend(loc='upper right',prop={'size':7})
    #plt.subplots_adjust(left=left_margin_coord, right=right_margin_coord, top=top_margin_coord, bottom=bottom_margin_coord)

    # Plot plot_Vsw_GSE.
    plot_Vsw_GSE.set_title('Solar Wind Velocity (GSE Frame)', fontsize=fig_title_fontsize)
    plot_Vsw_GSE.xaxis.set_major_formatter(fig_formatter) # Set xlabel format.
    # Plot solar wind speed in GSE frame.
    plot_Vsw_GSE.plot(Vsw_inGSE.index, Vsw_inGSE['Vx'], color = 'r', label='Vx', linewidth=fig_linewidth, linestyle='solid')
    plot_Vsw_GSE.plot(Vsw_inGSE.index, Vsw_inGSE['Vy'], color = 'g', label='Vy', linewidth=fig_linewidth, linestyle='solid')
    plot_Vsw_GSE.plot(Vsw_inGSE.index, Vsw_inGSE['Vz'], color = 'b', label='Vz', linewidth=fig_linewidth, linestyle='solid')
    plot_Vsw_GSE.legend(loc='upper right',prop={'size':7})
    #plot_Vsw_GSE.set_xlabel('Time',fontsize=fig_xlabel_fontsize)
    plot_Vsw_GSE.set_ylabel('Vsw(km/s)',fontsize=fig_ylabel_fontsize)
    plot_Vsw_GSE.tick_params(axis='x', labelsize=fig_xtick_fontsize) # Tick font size.
    plot_Vsw_GSE.tick_params(axis='y', labelsize=fig_ytick_fontsize) # Tick font size.
    #plt.subplots_adjust(left=left_margin_coord, right=right_margin_coord, top=top_margin_coord, bottom=bottom_margin_coord)

    # Plot Proton Number Density.
    plot_Np.set_title('Proton Number Density', fontsize=fig_title_fontsize)
    plot_Np.xaxis.set_major_formatter(fig_formatter) # Set xlabel format.
    # Plot Proton number density.
    plot_Np.plot(Np_inGSE.index, Np_inGSE, color = 'b', linewidth=fig_linewidth)
    #plot_Np.legend(loc='upper right',prop={'size':7})
    #plot_Np.set_xlabel('Time',fontsize=fig_xlabel_fontsize)
    plot_Np.set_ylabel('Np(#/cc)',fontsize=fig_ylabel_fontsize)
    plot_Np.set_ylim([0,30])
    plot_Np.tick_params(axis='x', labelsize=fig_xtick_fontsize) # Tick font size.
    plot_Np.tick_params(axis='y', labelsize=fig_ytick_fontsize) # Tick font size.
    #plt.subplots_adjust(left=left_margin_coord, right=right_margin_coord, top=top_margin_coord, bottom=bottom_margin_coord)

    # Plot plot_Tp.
    plot_Tp.set_title('Proton Temperature and $T_e/T_p$', fontsize=fig_title_fontsize)
    plot_Tp.xaxis.set_major_formatter(fig_formatter) # Set xlabel format.
    # Plot Proton number density.
    #print(Tp_MK.index)
    #print(Tp_MK)
    plot_Tp.plot(Tp_MK.index, Tp_MK, color = 'b', linewidth=fig_linewidth, label='Tp')
    #plot_Tp.set_xlabel('Time',fontsize=fig_xlabel_fontsize)
    plot_Tp.set_ylabel('Tp($10^6$K)',fontsize=fig_ylabel_fontsize)
    plot_Tp.set_ylim([0,0.5])
    plot_Tp.tick_params(axis='x', labelsize=fig_xtick_fontsize) # Tick font size.
    plot_Tp.tick_params(axis='y', labelsize=fig_ytick_fontsize) # Tick font size.
    plot_Tp.legend(loc='upper left',prop={'size':7})
    if ('Te' in GS_fluxRopeCandidate_DF.keys())&(isOK_Te_data)&(isOK_Tp_data):
        # Calculate Te/Tp.
        Te_Tp_ratio = Te_MK['Te']/Tp_MK['Tp']
        plot_Te_Tp_ratio = plot_Tp.twinx()
        plot_Te_Tp_ratio.set_ylabel(r'$T_e/T_p$', color='r', fontsize=fig_ylabel_fontsize)
        plot_Te_Tp_ratio.set_ylim([0,5])
        plot_Te_Tp_ratio.plot(Tp_MK.index, Te_Tp_ratio, color = 'r', linewidth=fig_linewidth, label=r'$T_e/T_p$')
        plot_Te_Tp_ratio.tick_params(axis='y', labelsize=fig_ytick_fontsize) # Tick font size.
        plot_Te_Tp_ratio.legend(loc='upper right',prop={'size':7})
        for ticklabel in plot_Te_Tp_ratio.get_yticklabels(): # Set label color to green
            ticklabel.set_color('r')
    #plt.subplots_adjust(left=left_margin_coord, right=right_margin_coord, top=top_margin_coord, bottom=bottom_margin_coord)

    # Plot plasma beta, including beta_p and beta_e
    # Calculate plasma beta_p.
    B_norm = (B_inFR[0]**2 + B_inFR[1]**2 + B_inFR[2]**2)**0.5
    # Original Np is in #/cc ( cc = cubic centimeter). Multiply by 1e6 to convert cc to m^3.
    Beta_p = (Np_inGSE['Np']*1e6) * k_Boltzmann * (Tp_MK['Tp']*1e6) / (np.square(B_norm*1e-9)/(2.0*mu0))
    plot_Beta_p.set_title('Plasma Beta', fontsize=fig_title_fontsize)
    plot_Beta_p.xaxis.set_major_formatter(fig_formatter) # Set xlabel format.
    # Plot plasma beta (including electron and proton).
    plot_Beta_p.plot(Beta_p.index, Beta_p, color = 'b', linewidth=fig_linewidth, label=r'$\beta_p$')
    #plot_Beta_p.set_xlabel('Time',fontsize=fig_xlabel_fontsize)
    plot_Beta_p.set_ylabel(r'$\beta_p$ and $\beta$',fontsize=fig_ylabel_fontsize)
    plot_Beta_p.set_ylim([0,5])
    plot_Beta_p.tick_params(axis='x', labelsize=fig_xtick_fontsize) # Tick font size.
    plot_Beta_p.tick_params(axis='y', labelsize=fig_ytick_fontsize) # Tick font size.
    # If Te data is good, plot Beta(proton and electron)
    if ('Te' in GS_fluxRopeCandidate_DF.keys())&(isOK_Te_data):
        # Calculate plasma beta.
        # Original Np is in #/cc ( cc = cubic centimeter). Multiply by 1e6 to convert cc to m^3.
        Beta = (Np_inGSE['Np']*1e6) * k_Boltzmann * ((Tp_MK['Tp']*1e6)+Te_MK['Te']*1e6) / (np.square(B_norm*1e-9)/(2.0*mu0))
        plot_Beta_p.xaxis.set_major_formatter(fig_formatter) # Set xlabel format.
        # Plot plasma beta (including electron and proton).
        plot_Beta_p.plot(Beta.index, Beta, color = 'g', linewidth=fig_linewidth, label=r'$\beta$')
        plot_Beta_p.tick_params(axis='y', labelsize=fig_ytick_fontsize) # Tick font size.
    # Set legend.
    plot_Beta_p.legend(loc='upper right',prop={'size':7})
    # Adjust plot margin.
    #plt.subplots_adjust(left=left_margin_coord, right=right_margin_coord, top=top_margin_coord, bottom=bottom_margin_coord)

    # Plot PitchAngle:
    e_pitch.set_title('Suprathermal Electrons Pitch Angle Distribution (' + energy_label + ')', fontsize=fig_title_fontsize)
    e_pitch.xaxis.set_major_formatter(fig_formatter) # Set xlabel format.
    e_pitch.yaxis.set_major_locator(MaxNLocator(3))
    #e_pitch.legend(loc='upper right',prop={'size':7})
    #e_pitch.set_xlabel('Time',fontsize=fig_xlabel_fontsize)
    e_pitch.set_ylabel('pitch angle (deg)',fontsize=fig_ylabel_fontsize) # Label font size.
    e_pitch.set_ylim([0,180])
    e_pitch.tick_params(axis='x', labelsize=fig_xtick_fontsize) # Tick font size.
    e_pitch.tick_params(axis='y', labelsize=fig_ytick_fontsize) # Tick font size.
    # Plot 2D data.
    ax_e_pitch = e_pitch.pcolormesh(GS_fluxRopeCandidate_PitchAngle_DF.index, GS_fluxRopeCandidate_PitchAngle_DF.columns.values, GS_fluxRopeCandidate_PitchAngle_DF.values.transpose(), cmap='jet', norm=matplotlib.colors.LogNorm(vmin=1e-31, vmax=1e-26))
    #plt.subplots_adjust(left=left_margin_coord, right=right_margin_coord, top=top_margin_coord, bottom=bottom_margin_coord)

    left_margin = 0.065    # the left side of the subplots of the figure
    right_margin = 0.935   # the right side of the subplots of the figure
    top_margin = 0.95     # the top of the subplots of the figure
    bottom_margin = 0.05  # the bottom of the subplots of the figure
    width_gap = None       # the amount of width reserved for blank space between subplots
    height_gap = 0.5       # the amount of height reserved for white space between subplots
    plt.subplots_adjust(left=left_margin, bottom=bottom_margin, right=right_margin, top=top_margin, wspace=width_gap, hspace=height_gap)

    # Plot color bar.
    box = e_pitch.get_position() # Get pannel position.
    pad, width = 0.005, 0.005 # pad = distance to panel, width = colorbar width.
    cax = fig2.add_axes([box.xmax + pad, box.ymin, width, box.height]) # Set colorbar position.
    ax_e_pitch_cbar = fig2.colorbar(ax_e_pitch, ax=e_pitch, cax=cax)
    #ax_e_pitch_cbar.set_label('#/[cc*(cm/s)^3]', rotation=270, fontsize=fig_ytick_fontsize)
    ax_e_pitch_cbar.ax.minorticks_on()
    ax_e_pitch_cbar.ax.tick_params(labelsize=fig_ytick_fontsize)
    ax_e_pitch_cbar.outline.set_linewidth(fig_linewidth)
    e_pitch.tick_params(axis='x', which='major', labelsize=fig_xtick_fontsize) # This is a shared axis for all subplot.

    #plt.tight_layout()

    # Save plot. Supported formats: emf, eps, pdf, png, ps, raw, rgba, svg, svgz.
    # Save as png format.
    timeSeries_fig_filename = rootDir + outputDir + timeRange_str + '_' + 'timeSeries_plot' + '.' + fig_format
    print(timeSeries_fig_filename)
    fig2.savefig(timeSeries_fig_filename, format=fig_format, dpi=fig_dpi)#, bbox_inches='tight')
    #plt.close(fig2)
    
    combined = FR.vstack_images(grid_fig_filename, timeSeries_fig_filename)
    combined_filename = rootDir + outputDir + timeRange_str + '.' + fig_format
    combined.save(combined_filename)
    print('{} is saved!'.format(combined_filename))
    os.remove(grid_fig_filename)
    os.remove(timeSeries_fig_filename)
    plt.close(fig1)
    plt.close(fig2)

    return True
###############################################################################

# Choose root directory according to environment.
def setRootDir(ENV):
    return {
        'macbook'    : '/Users/jz0006/GoogleDrive/GS/',
        'bladerunner': '/home/jinlei/gs/',
        'blueshark'  : '/udrive/staff/lzhao/jinlei/gs/',
    }.get(ENV, 0) # 0 is default if ENV not found

###############################################################################

# Physics constants.
mu0 = 4.0 * np.pi * 1e-7 #(N/A^2) magnetic constant permeability of free space vacuum permeability
m_proton = 1.6726219e-27 # Proton mass. In kg.
factor_deg2rad = np.pi/180.0 # Convert degree to rad.
k_Boltzmann = 1.3806488e-23 # Boltzmann constant, in J/K.
# Parameters.
dt = 60.0 # For 1 minute resolution data, the time increment is 60 seconds.

print('\nCheck the python version on Mac OS X, /usr/local/bin/python should be used:')
os.system('which python')

# Read year string from command line argument.
#year_str = sys.argv[1]
#year = int(year_str)

year_start = int(sys.argv[1])
year_end   = int(sys.argv[2])

for year in xrange(year_start,year_end+1,1):
    year_str = str(year)
    # Specify starting and ending time in a year.
    specified_startTime = datetime(year, 1,1,0,0,0)
    specified_endTime   = datetime(year, 12,31,23,59,59)
    duration_range_str = str(specified_startTime.strftime('%Y%m%d%H%M')) + '~' + str(specified_endTime.strftime('%Y%m%d%H%M'))


    # Specify input and output path
    rootDir = setRootDir('macbook') # 'macbook', 'bladerunner', or 'blueshark'.
    inputListDir = 'GS_SearchResult/selected_events/'
    inputFileName = year_str+'_selected_events_B_abs_mean>=5.p'
    outputDir = 'GS_WEB_html/'+year_str+'/images/'
    # If outputWebDir does not exist, create it.
    if not os.path.exists(rootDir + outputDir):
        os.makedirs(rootDir + outputDir)

    '''
    # Set log.
    frPlot_log_bufsize = 1 # 0 means unbuffered, 1 means line buffered.
    frPlot_log_filename = 'fr_' + duration_range_str +'.log'
    frPlot_log_path = rootDir +'frPlotLog/' # Log directory.
    # If frPlot_log_path does not exist, create it.
    if not os.path.exists(frPlot_log_path):
        os.makedirs(frPlot_log_path)
    frPlot_log_path_filename = frPlot_log_path + frPlot_log_filename
    # Create log file, all print statement will be writen into this log file.
    frPlot_log = open(frPlot_log_path_filename, 'w', frPlot_log_bufsize)
    sys.stdout = frPlot_log
    '''

    # Load web data pickle file.
    FR_web_data = pd.read_pickle(open(rootDir + inputListDir + inputFileName,'rb'))
    eventList_length = len(FR_web_data)
    print('eventList_length = {}'.format(eventList_length))
    print('Checking FR_web_data keys... {}'.format(FR_web_data.keys()))

    # Read and Check one year data.
    print('Reading data...')
    GS_oneYearData_DataFrame = pd.read_pickle(open(rootDir + 'GS_DataPickleFormat/preprocessed/GS_' + year_str + '_AllData_DataFrame_preprocessed.p','rb'))
    GS_PitchAngle_DataFrame = pd.read_pickle(open(rootDir + 'GS_DataPickleFormat/pitch_angle/GS_' + year_str + '_PitchAngle_DataFrame_WIND.p','rb'))
    if year in range(1996, 2002):
        energy_label = '94.02eV'
    elif year in range(2002, 2017):
        energy_label = '96.71eV'
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

    # Grab the time interval of interest.
    FR_web_data_specificTimeInterval = FR_web_data[(FR_web_data['startTime']>specified_startTime)&(FR_web_data['startTime']<specified_endTime)]
    FR_web_data_specificTimeInterval.reset_index(drop=True, inplace=True)

    '''
    # Multiprocessing
    num_cpus = multiprocessing.cpu_count()
    max_processes = num_cpus
    print '\nTotol CPU cores on this node = ', num_cpus
    # Create a multiprocessing pool with safe_lock.
    pool = multiprocessing.Pool(processes=max_processes)
    '''

    len_FR_web_data_specificTimeInterval = len(FR_web_data_specificTimeInterval)
    for i in range(len_FR_web_data_specificTimeInterval):
        print('\nplotting = {}/{}...'.format(i+1, len_FR_web_data_specificTimeInterval))
        # Prepare data for one event.
        oneRecord_DF = FR_web_data_specificTimeInterval.iloc[[i]]
        plotFluxRopeInfo(oneRecord_DF)
        gc.collect()


    '''
    # Close pool, prevent new worker process from joining.
    pool.close()
    # Block caller process until workder processes terminate.
    pool.join()
    '''









        
