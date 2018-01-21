#!/usr/local/bin/python
# V3.1

'''
2016-11-16
Use brute force method to detect flux rope. Window center sliding method.
1) v1.4 is able to handle multiple structure.
2) comment out print statement to accelerate.
3) improve the method to find location of turn point.
4) put true and false result into two subfolder, and change cycle counter to 10000.
2016-11-30
version 1.7
1) change turnPointOnTop = (Pt_turnPoint>(Pt_max-(Pt_max-Pt_min)*0.2)) to turnPointOnTop = (Pt_turnPoint>(Pt_max-(Pt_max-Pt_min)*0.15))

2017-01-03
version 1.8
1) use flexible smooth window when get A curve before testing turn point counts. savgol_filter_window = half_window, no less than 9.

2017-02-01
version 1.9
1) Implement VHT.

2017-02-03
version 2.0
1) Change the interval specifying method. half_window_list = [5, 10, 15, 20, 25, 30]

2017-02-07
version 2.1
1) Combine three version into one.(For macbook, blueshark, and bladerunner)
2) Remove the command to get Np_DataFrame. Np is not used in this code.
3) Create and destroy pool in inner for loop. Since we want to save result in file when we finish one inner loop(complete one window size), we need to block the processing until all parallel subproceure finish with in the inner loop, and then save file. After saving file, create a new pool and start a new inner loop.
4) Add searching time range printing in searchFluxRopeInWindow() function to indicate procedure.

2017-02-07
version 2.2
1) Add one more command line argument to specify the time range list. Command format is:
    python GS_detectFluxRope_multiprocessing_v2.2.py 1996 '((20,30),(30,40),(40,50))'

2017-02-09
version 2.3
1) In version 2.2, we used a outer for loop to iterate duration range, and a inner for loop to iterate sliding window. We create a multiprocessing pool before the beginning of inner for loop, and put all worker processes generated in inner for loop into the processes pool. When the inner for loop finish, we use pool.close() and pool.join() to wait for the worker porcesses to finish. Then save the results generated in innner for loop. After that, we delete pool object and collect memory. For the next iteration in outer for loop, we create a new pool object and do the same thing as above. This code works well on the bladerunner server in UAH, however, does not work properly on blueshark server in FIT. When running on blueshark, once the outer loop steps into the second iteration, the server becomes very slow. In first outer iteration, it takes 20 minutes to go through one month, but in the second outer iteration, it takes 2 hours or more. I think it may due to the version of multiprocessing package or python, which cannot handle mulitpy pools properly. In this version, we create only one multiprocessing pool outside the outer loop, and close it when outer for loop finish. On each end of inner loop, we use trick to block the main processe to wait for the worker processes to finish and to save the results from inner loop.

2017-02-10
version 2.4
1) Change the residue calculating formula. Divided by N within the square root.

2017-02-11
version 3.0
1) A new major version. Retrun more informations.

2017-06-10
version 3.1
1) In this version, we change the A value smoothing method. Firstly, downsample A to 20 points, then apply savgol_filter, then upsample to original data points number. In this way, we do not need to specify savgol_filter smoothing window size for different size detection windows.
2) Improve sliding window generating method.
3) Use a new duration tuple specify method. User provide the min and max duration, and window size range step, program calcuate the duration tuples.

'''

############################################ Command line format ############################################

# GS_detectFluxRope_multiprocessing_v3.1.py 1996 10 50 5
# First parameter is year, second is minimum size, third is maxmum size, fourth is size step.
# The command above generate the following size tuple:
# ((10, 15), (15, 20), (20, 25), (25, 30), (30, 35), (35, 40), (40, 45), (45, 50))

#############################################################################################################


from __future__ import division # Treat integer as float.
import os
import sys
import pickle
import math
import time
import numpy as np # Scientific calculation package.
from numpy import linalg as la
import pandas as pd
import scipy as sp
from scipy.signal import savgol_filter # Savitzky-Golay filter
from scipy import integrate
from scipy import stats
from datetime import datetime # Import datetime class from datetime package.
from datetime import timedelta
import multiprocessing
import gc # Garbage Collector.
from ast import literal_eval


############################################ User defined module ############################################

# Choose root directory according to environment.
def setRootDir(ENV):
    return {
        'macbook'    : '/Users/jz0006/GoogleDrive/GS/',
        'bladerunner': '/home/jinlei/gs/',
        'blueshark'  : '/udrive/staff/lzhao/jinlei/gs/',
    }.get(ENV, 0) # 0 is default if ENV not found

#############################################################################################################

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

# Calculate the eignvectors and eigenvaluse of input matrix dataframe. This module is Python style.
def eigenMatrix(matrix_DataFrame, **kwargs):
    # Calculate the eigenvalues and eigenvectors of covariance matrix.
    eigenValue, eigenVector = la.eig(matrix_DataFrame) # eigen_arg are eigenvalues, and eigen_vec are eigenvectors.
    # Sort the eigenvalues and arrange eigenvectors by sorted eigenvalues.
    eigenValue_i = np.argsort(eigenValue) # covM_B_eigenValue_i is sorted index of covM_B_eigenValue
    lambda3 = eigenValue[eigenValue_i[0]] # lambda3, minimum variance
    lambda2 = eigenValue[eigenValue_i[1]] # lambda2, intermediate variance.
    lambda1 = eigenValue[eigenValue_i[2]] # lambda1, maximum variance.
    eigenVector3 = pd.DataFrame(eigenVector[:, eigenValue_i[0]], columns=['minVar(lambda3)']) # Eigenvector 3, along minimum variance
    eigenVector2 = pd.DataFrame(eigenVector[:, eigenValue_i[1]], columns=['interVar(lambda2)']) # Eigenvector 2, along intermediate variance.
    eigenVector1 = pd.DataFrame(eigenVector[:, eigenValue_i[2]], columns=['maxVar(lambda1)']) # Eigenvector 1, along maximum variance.
    
    if kwargs['formXYZ']==True:
        # Form an eigenMatrix with the columns:
        # X = minimum variance direction, Y = Maximum variance direction, Z = intermediate variance direction.
        eigenMatrix = pd.concat([eigenVector3, eigenVector1, eigenVector2], axis=1)
        eigenValues = pd.DataFrame([lambda3, lambda1, lambda2], index=['X(min)', 'Y(max)', 'Z(inter)'], columns=['eigenValue'])
    else:
        # Form a sorted eigenMatrix using three sorted eigenvectors. Columns are eigenvectors.
        eigenMatrix = pd.concat([eigenVector3, eigenVector2, eigenVector1], axis=1)
        eigenValues = pd.DataFrame([lambda3, lambda2, lambda1], index=['lambda3', 'lambda2', 'lambda1'], columns=['eigenValue'])
    
    return eigenValues, eigenMatrix

################################################################################################################

# Find X axis according to Z axis and V. The X axis is the projection of V on the plane perpendicular to Z axis.
# For this function, numba is slower than python.
def findXaxis(Z, V):
    #import numpy as np # Scientific calculation package.
    #from numpy import linalg as la
    Z = np.array(Z)
    V = np.array(V)
    # Both Z and V are unit vector representing the directions. They are numpy 1-D arrays.
    z1 = Z[0]; z2 = Z[1]; z3 = Z[2]; v1 = V[0]; v2 = V[1]; v3 = V[2]
    # V, Z, and X must satisfy two conditions. 1)The are co-plane. 2)X is perpendicular to Z. These two conditions
    # lead to two equations with three unknow. We can solve for x1, x2, and x3, in which x1 is arbitrary. Let x1
    # equals to 1, then normalize X.
    # 1) co-plane : (Z cross V) dot X = 0
    # 2) Z perpendicular to X : Z dot X = 0
    x1 = 1.0 # Arbitray.
    x2 = -((x1*(v2*z1*z1 - v1*z1*z2 - v3*z2*z3 + v2*z3*z3))/(v2*z1*z2 - v1*z2*z2 + v3*z1*z3 - v1*z3*z3))
    x3 = -((x1*(v3*z1*z1 + v3*z2*z2 - v1*z1*z3 - v2*z2*z3))/(v2*z1*z2 - v1*z2*z2 + v3*z1*z3 - v1*z3*z3))
    # Normalization.
    X = np.array([float(x1), float(x2), float(x3)])
    X = X/(la.norm(X))
    if X.dot(V) < 0:
        X = - X
    return X

# Given two orthnormal vectors(Z and X), find the third vector(Y) to form right-hand side frame.
# For this function, numba is slower than python.
def formRighHandFrame(X, Z): # Z cross X = Y in right hand frame.
    #import numpy as np # Scientific calculation package.
    #from numpy import linalg as la
    X = np.array(X)
    Z = np.array(Z)
    Y = np.cross(Z, X)
    Y = Y/(la.norm(Y)) # Normalize.
    return Y

# Find how many turning points in an array.
# For this function, numba is slower than python.
def turningPoints(array):
    array = np.array(array)
    dx = np.diff(array)
    dx = dx[dx != 0] # if don't remove duplicate points, will miss the turning points with duplicate values.
    return np.sum(dx[1:] * dx[:-1] < 0)

################################################################################################################
# Loop for all directions to calculate residue, return the smallest residue and corresponding direction.
def searchFluxRopeInWindow(B_DataFrame, VHT, theta_stepLength, phi_stepLength, minDuration, dt, flag_smoothA):
    
    #t0 = datetime.now()
    print('{} - [{}~{} minutes] searching: ({} ~ {})'.format(time.ctime(), minDuration, len(B_DataFrame), B_DataFrame.index[0], B_DataFrame.index[-1]))
    #t1 = datetime.now()
    #print((t1-t0).total_seconds())
    
    # Initialization.
    # Caution: the type of return value will be different if the initial data is updated. If updated, timeRange_temp will become to tuple, plotData_dict_temp will becomes to dict, et, al.
    time_start_temp = np.nan
    time_end_temp = np.nan
    time_turn_temp = np.nan
    turnPointOnTop_temp = np.nan
    Residue_diff_temp = np.inf
    Residue_fit_temp = np.inf
    duration_temp = np.nan
    theta_temp = np.nan
    phi_temp = np.nan
    time_start, time_end, time_turn, turnPointOnTop, Residue_diff, Residue_fit, duration = getResidueForCurrentAxial(0, 0, minDuration, B_DataFrame, VHT, dt, flag_smoothA)
    #print('For current orientation, the returned residue is {}'.format(Residue))
    #print('For current orientation, the returned duration is {}'.format(duration))
    if  Residue_diff < Residue_diff_temp:
        time_start_temp = time_start
        time_end_temp = time_end
        time_turn_temp = time_turn
        turnPointOnTop_temp = turnPointOnTop
        Residue_diff_temp = Residue_diff
        Residue_fit_temp = Residue_fit
        theta_temp = 0
        phi_temp = 0
        duration_temp = duration
    
    # This step loops all theta and phi except for theta = 0.
    # Bug found! Theta loop only from 0 to 80. Should be 0 to 90. 2017/10/10.
    for theta_deg in range(int(theta_stepLength), 90-int(theta_stepLength)+1, int(theta_stepLength)): # Not include theta = 0.
        # Phi loop from 0 to 340. It's OK. Because 0 and 360 are the same angle. 2017/10/10.
        for phi_deg in range(0, 360-int(phi_stepLength)+1, int(phi_stepLength)): # Include phi = 0.
            #print('theta_deg = {}, phi_deg = {}'.format(theta_deg, phi_deg))
            time_start, time_end, time_turn, turnPointOnTop, Residue_diff, Residue_fit, duration = getResidueForCurrentAxial(theta_deg, phi_deg, minDuration, B_DataFrame, VHT, dt, flag_smoothA)
            #print('For current orientation, the returned residue is {}'.format(Residue))
            #print('For current orientation, the returned duration is {}'.format(duration))
            if Residue_diff < Residue_diff_temp:
                time_start_temp = time_start
                time_end_temp = time_end
                time_turn_temp = time_turn
                turnPointOnTop_temp = turnPointOnTop
                Residue_diff_temp = Residue_diff
                Residue_fit_temp = Residue_fit
                theta_temp = theta_deg
                phi_temp = phi_deg
                duration_temp = duration

    #print('Residue_diff = {}'.format(Residue_diff_temp))
    #print('Residue_fit  = {}\n'.format(Residue_fit_temp))
    # Round some results.
    #print((time_start_temp, time_turn_temp, time_end_temp, duration_temp, turnPointOnTop_temp, Residue_diff_temp, Residue_fit_temp, (theta_temp, phi_temp), (round(VHT[0],5),round(VHT[1],5),round(VHT[2],5))))
    return time_start_temp, time_turn_temp, time_end_temp, duration_temp, turnPointOnTop_temp, Residue_diff_temp, Residue_fit_temp, (theta_temp, phi_temp), (round(VHT[0],5),round(VHT[1],5),round(VHT[2],5))

################################################################################################################
# Calculate the residue for given theta and phi.
def getResidueForCurrentAxial(theta_deg, phi_deg, minDuration, B_DataFrame, VHT, dt, flag_smoothA):
    # Initialize
    time_start = np.nan
    time_end = np.nan
    time_turn = np.nan
    Residue_diff = np.inf
    Residue_fit = np.inf
    duration = np.nan
    turnPointOnTop = np.nan
    # Loop for half polar angle (theta(0~90 degree)), and azimuthal angle (phi(0~360 degree)) for Z axis orientations.
    # Direction cosines:
    # x = rcos(alpha) = rsin(theta)cos(phi) => cos(alpha) = sin(theta)cos(phi)
    # y = rcos(beta)  = rsin(theta)sin(phi) => cos(beta)  = sin(theta)sin(phi)
    # z = rcos(gamma) = rcos(theta)         => cos(gamma) = cos(theta)
    # Using direction cosines to form a unit vector.
    theta_rad = factor_deg2rad * theta_deg
    phi_rad   = factor_deg2rad * phi_deg
    # Form new Z_unitVector according to direction cosines.
    Z_unitVector = np.array([np.sin(theta_rad)*np.cos(phi_rad), np.sin(theta_rad)*np.sin(phi_rad), np.cos(theta_rad)])
    # Find X axis from Z axis and -VHT.
    X_unitVector = findXaxis(Z_unitVector, -VHT)
    # Find the Y axis to form a right-handed coordinater with X and Z.
    Y_unitVector = formRighHandFrame(X_unitVector, Z_unitVector)

    # Project B_DataFrame into new trial Frame.
    transToTrialFrame = np.array([X_unitVector, Y_unitVector, Z_unitVector]).T
    B_inTrialframe_DataFrame = B_DataFrame.dot(transToTrialFrame)
    # Project VHT into new trial Frame.
    VHT_inTrialframe = VHT.dot(transToTrialFrame)

    # Calculate A(x,0) by integrating By. A(x,0) = Integrate[-By(s,0)ds, {s, 0, x}], where ds = -Vht dot unit(X) dt.
    ds = - VHT_inTrialframe[0] * 1000.0 * dt # Space increment along X axis. Convert km/s to m/s.
    # Don't forget to convert km/s to m/s, and convert nT to T. By = B_inTrialframe_DataFrame[1]
    A = integrate.cumtrapz(-B_inTrialframe_DataFrame[1]*1e-9, dx=ds, initial=0)
    # Calculate Pt(x,0). Pt(x,0)=p(x,0)+Bz^2/(2mu0). By = B_inTrialframe_DataFrame[2]
    Pt = np.array((B_inTrialframe_DataFrame[2] * 1e-9)**2 / (2.0*mu0) * 1e9) # 1e9 convert unit form pa to npa.
    # Check how many turning points in original data.
    num_A_turningPoints = turningPoints(A)
    #print('num_A_turningPoints = {}'.format(num_A_turningPoints))
    
    '''
    if flag_smoothA == True:
        # Smooth A series to find the number of the turning point in main trend.
        # Because the small scale turning points are not important.
        # We only use A_smoothed to find turning points, when fitting Pt with A, original A is used.
        #savgol_filter_window = 9
        order = 3
        A_smoothed = savgol_filter(A, savgol_filter_window, order)
    else:
        A_smoothed = A
    '''
    
    if flag_smoothA == True:
        # Smooth A series to find the number of the turning point in main trend.
        # Because the small scale turning points are not important.
        # We only use A_smoothed to find turning points, when fitting Pt with A, original A is used.
        # Firstly, downsample A to 20 points, then apply savgol_filter, then upsample to original data points number.
        index_A = range(len(A))
        # Downsample A to 20 points.
        index_downsample = np.linspace(index_A[0],index_A[-1], 20)
        A_downsample = np.interp(index_downsample, index_A, A)
        # Apply savgol_filter. Set smooth window size to half of 20 but odd, i.e. 11.
        A_downsample = savgol_filter(A_downsample, 11, 3) # 11 is smooth window size, 3 is polynomial order.
        # Upsample to original data points amount.
        A_upsample = np.interp(index_A, index_downsample, A_downsample)
        # The smoothed A is just upsampled A.
        A_smoothed = A_upsample
    else:
        A_smoothed = A
    
    # Check how many turning points in smoothed data.
    num_A_smoothed_turningPoints = turningPoints(A_smoothed)
    #print('num_A_smoothed_turningPoints = {}'.format(num_A_smoothed_turningPoints))

    # num_A_smoothed_turningPoints==0 means the A value is not double folded. It's monotonous. Skip.
    # num_A_smoothed_turningPoints > 1 means the A valuse is 3 or higher folded. Skip.
    # continue # Skip the rest commands in current iteration.
    if (num_A_smoothed_turningPoints==0)|(num_A_smoothed_turningPoints>1):
        #return timeRange, Residue, duration, plotData_dict, transToTrialFrame, turnPoint_dict # Skip the rest commands in current iteration.
        return time_start, time_end, time_turn, turnPointOnTop, Residue_diff, Residue_fit, duration # Skip the rest commands in current iteration.
    #print('Theta={}, Phi={}. Double-folding feature detected!\n'.format(theta_deg, phi_deg))
    
    # Find the boundary of A.
    A_smoothed_start = A_smoothed[0] # The first value of A.
    A_smoothed_end = A_smoothed[-1] # The last value of A.
    A_smoothed_max_index = A_smoothed.argmax() # The index of max A, return the index of first max(A).
    A_smoothed_max = A_smoothed[A_smoothed_max_index] # The max A.
    A_smoothed_min_index = A_smoothed.argmin() # The index of min A, return the index of first min(A).
    A_smoothed_min = A_smoothed[A_smoothed_min_index] # The min A.

    if (A_smoothed_min == min(A_smoothed_start, A_smoothed_end))&(A_smoothed_max == max(A_smoothed_start, A_smoothed_end)):
        # This means the A value is not double folded. It's monotonous. Skip.
        # Sometimes num_A_smoothed_turningPoints == 0 does not work well. This is double check.
        return time_start, time_end, time_turn, turnPointOnTop, Residue_diff, Residue_fit, duration
    elif abs(A_smoothed_min - ((A_smoothed_start + A_smoothed_end)/2)) < abs(A_smoothed_max - ((A_smoothed_start + A_smoothed_end)/2)):
        # This means the turning point is on the right side.
        A_turnPoint_index = A_smoothed_max_index
        turnPointOnRight = True
    elif abs(A_smoothed_min - ((A_smoothed_start + A_smoothed_end)/2)) > abs(A_smoothed_max - ((A_smoothed_start + A_smoothed_end)/2)):
        # This means the turning point is on the left side.
        A_turnPoint_index = A_smoothed_min_index
        turnPointOnLeft = True

    # Split A into two subarray from turning point.
    A_sub1 = A[:A_turnPoint_index+1]
    Pt_sub1 = Pt[:A_turnPoint_index+1] # Pick corresponding Pt according to index of A.
    A_sub2 = A[A_turnPoint_index:]
    Pt_sub2 = Pt[A_turnPoint_index:] # Pick corresponding Pt according to index of A.

    # Get time stamps.
    timeStamp = B_inTrialframe_DataFrame.index
    # Split time stamps into two subarray from turning point.
    timeStamp_sub1 = timeStamp[:A_turnPoint_index+1]
    timeStamp_sub2 = timeStamp[A_turnPoint_index:]
    
    # Keep the time of turn point and the value of Pt turn point.
    Pt_turnPoint = Pt[A_turnPoint_index]
    timeStamp_turnPoint = timeStamp[A_turnPoint_index]

    # This block is to find the time range.
    # Put two branches into DataFrame.
    Pt_vs_A_sub1_DataFrame = pd.DataFrame(np.array([Pt_sub1, timeStamp_sub1]).T, index=A_sub1, columns=['Pt_sub1','timeStamp_sub1'])
    Pt_vs_A_sub2_DataFrame = pd.DataFrame(np.array([Pt_sub2, timeStamp_sub2]).T, index=A_sub2, columns=['Pt_sub2','timeStamp_sub2'])
    # Sort by A. A is index in Pt_vs_A_sub1_DataFrame.
    Pt_vs_A_sub1_DataFrame.sort_index(ascending=True, inplace=True, kind='quicksort')
    Pt_vs_A_sub2_DataFrame.sort_index(ascending=True, inplace=True, kind='quicksort')

    # Trim two branches to get same length.
    A_sub1_boundary_left = Pt_vs_A_sub1_DataFrame.index.min()
    A_sub1_boundary_right = Pt_vs_A_sub1_DataFrame.index.max()
    A_sub2_boundary_left = Pt_vs_A_sub2_DataFrame.index.min()
    A_sub2_boundary_right = Pt_vs_A_sub2_DataFrame.index.max()

    A_boundary_left = max(A_sub1_boundary_left, A_sub2_boundary_left)
    A_boundary_right = min(A_sub1_boundary_right, A_sub2_boundary_right)

    #Pt_vs_A_sub1_trimmed_DataFrame = Pt_vs_A_sub1_DataFrame.loc[A_boundary_left:A_boundary_right]
    #Pt_vs_A_sub2_trimmed_DataFrame = Pt_vs_A_sub2_DataFrame.loc[A_boundary_left:A_boundary_right]
    Pt_vs_A_sub1_trimmed_DataFrame = Pt_vs_A_sub1_DataFrame.iloc[Pt_vs_A_sub1_DataFrame.index.get_loc(A_boundary_left,method='nearest'):Pt_vs_A_sub1_DataFrame.index.get_loc(A_boundary_right,method='nearest')+1]
    Pt_vs_A_sub2_trimmed_DataFrame = Pt_vs_A_sub2_DataFrame.iloc[Pt_vs_A_sub2_DataFrame.index.get_loc(A_boundary_left,method='nearest'):Pt_vs_A_sub2_DataFrame.index.get_loc(A_boundary_right,method='nearest')+1]
    # Get the time range of trimmed A.
    timeStamp_start = min(Pt_vs_A_sub1_trimmed_DataFrame['timeStamp_sub1'].min(skipna=True), Pt_vs_A_sub2_trimmed_DataFrame['timeStamp_sub2'].min(skipna=True))
    timeStamp_end = max(Pt_vs_A_sub1_trimmed_DataFrame['timeStamp_sub1'].max(skipna=True), Pt_vs_A_sub2_trimmed_DataFrame['timeStamp_sub2'].max(skipna=True))
    #timeRange = [timeStamp_start, timeStamp_end]
    time_start = int(timeStamp_start.strftime('%Y%m%d%H%M'))
    time_end = int(timeStamp_end.strftime('%Y%m%d%H%M'))
    time_turn = int(timeStamp_turnPoint.strftime('%Y%m%d%H%M'))
    duration = int((timeStamp_end - timeStamp_start).total_seconds()/60)+1

    # Skip if shorter than minDuration.
    if duration < minDuration:
        return time_start, time_end, time_turn, turnPointOnTop, Residue_diff, Residue_fit, duration

    # Calculate two residues respectively. Residue_fit and Residue_diff.
    # Preparing for calculating Residue_fit, the residue of all data sample w.r.t. fitted PtA curve.
    # Combine two trimmed branches.
    A_sub1_array = np.array(Pt_vs_A_sub1_trimmed_DataFrame.index)
    A_sub2_array = np.array(Pt_vs_A_sub2_trimmed_DataFrame.index)
    Pt_sub1_array = np.array(Pt_vs_A_sub1_trimmed_DataFrame['Pt_sub1'])
    Pt_sub2_array = np.array(Pt_vs_A_sub2_trimmed_DataFrame['Pt_sub2'])
    # The order must be in accordance.
    Pt_array = np.concatenate((Pt_sub1_array, Pt_sub2_array))
    A_array = np.concatenate((A_sub1_array, A_sub2_array))
    # Sort index.
    sortedIndex = np.argsort(A_array)
    A_sorted_array = A_array[sortedIndex]
    Pt_sorted_array = Pt_array[sortedIndex]
    # Fit a polynomial function (3rd order). Use it to calculate residue.
    Pt_A_coeff = np.polyfit(A_array, Pt_array, 3)
    Pt_A = np.poly1d(Pt_A_coeff)

    # Preparing for calculating Residue_diff, the residue get by compare two branches.
    # Merge two subset into one DataFrame.
    Pt_vs_A_trimmed_DataFrame = pd.concat([Pt_vs_A_sub1_trimmed_DataFrame, Pt_vs_A_sub2_trimmed_DataFrame], axis=1)
    # Drop timeStamp.
    Pt_vs_A_trimmed_DataFrame.drop(['timeStamp_sub1', 'timeStamp_sub2'], axis=1, inplace=True) # axis=1 for column.
    #print('\n')
    #print('duration = {}'.format(duration))
    #print('A_boundary_left = {}'.format(A_boundary_left))
    #print('A_boundary_right = {}'.format(A_boundary_right))
    # Interpolation.
    # "TypeError: Cannot interpolate with all NaNs" can occur if the DataFrame contains columns of object dtype. Convert data to numeric type. Check data type by print(Pt_vs_A_trimmed_DataFrame.dtypes).
    for one_column in Pt_vs_A_trimmed_DataFrame:
        Pt_vs_A_trimmed_DataFrame[one_column] = pd.to_numeric(Pt_vs_A_trimmed_DataFrame[one_column], errors='coerce')
    # Interpolate according to index A.
    Pt_vs_A_trimmed_DataFrame.interpolate(method='index', axis=0, inplace=True) # axis=0:fill column-by-column
    # Drop leading and trailing NaNs. The leading NaN won't be filled by linear interpolation, however,
    # the trailing NaN will be filled by forward copy of the last non-NaN values. So, for leading NaN,
    # just use pd.dropna, and for trailing NaN, remove the duplicated values.
    Pt_vs_A_trimmed_DataFrame.dropna(inplace=True) # Drop leading NaNs.
    trailing_NaN_mask_DataFrame = (Pt_vs_A_trimmed_DataFrame.diff()!=0) # Get duplicate bool mask.
    trailing_NaN_mask = np.array(trailing_NaN_mask_DataFrame['Pt_sub1'] & trailing_NaN_mask_DataFrame['Pt_sub2'])
    Pt_vs_A_trimmed_DataFrame = Pt_vs_A_trimmed_DataFrame.iloc[trailing_NaN_mask]

    # Get Pt_max and Pt_min. They will be used to normalize Residue for both Residue_fit and Residue_diff.
    Pt_max = Pt_sorted_array.max()
    Pt_min = Pt_sorted_array.min()
    Pt_max_min_diff = abs(Pt_max - Pt_min)
    # Check if turn point is on top.
    turnPointOnTop = (Pt_turnPoint>(Pt_max-(Pt_max-Pt_min)*0.15))

    # Use two different defination to calculate Residues. # Note that, the definition of Residue_diff is different with Hu's paper. We divided it by 2 two make it comparable with Residue_fit. The definition of Residue_fit is same as that in Hu2004.
    if Pt_max_min_diff == 0:
        Residue_diff = np.inf
        Residue_fit = np.inf
    else:
        Residue_diff = 0.5 * np.sqrt((1.0/len(Pt_vs_A_trimmed_DataFrame))*((Pt_vs_A_trimmed_DataFrame['Pt_sub1'] - Pt_vs_A_trimmed_DataFrame['Pt_sub2']) ** 2).sum()) / Pt_max_min_diff
        Residue_fit = np.sqrt((1.0/len(A_array))*((Pt_sorted_array - Pt_A(A_sorted_array)) ** 2).sum()) / Pt_max_min_diff
        # Round results.
        Residue_diff = round(Residue_diff, 5)
        Residue_fit = round(Residue_fit, 5)
    
    return time_start, time_end, time_turn, turnPointOnTop, Residue_diff, Residue_fit, duration


################################################################################################################
# Command line argument.
year_str = sys.argv[1]
size_start_str = sys.argv[2]
size_end_str   = sys.argv[3]
size_step_str  = sys.argv[4]
overlap_str    = sys.argv[5]

size_start = int(size_start_str)
size_end = int(size_end_str)
size_step = int(size_step_str)
overlap = int(overlap_str)

duration_range_list = []
size_start_one_iter_temp = size_start
size_end_one_iter_temp = size_start
while size_end_one_iter_temp < size_end:
    size_end_one_iter_temp = size_start_one_iter_temp + size_step
    #print([size_start_one_iter_temp, size_end_one_iter_temp])
    duration_range_list.append([size_start_one_iter_temp, size_end_one_iter_temp])
    size_start_one_iter_temp = size_end_one_iter_temp
if (duration_range_list[-1][-1] > size_end):
    duration_range_list[-1][-1] = size_end
'''
# Shift boundary to make overlap.
for i in range(len(duration_range_list)):
    if i==0:
        duration_range_list[i][1] += overlap
    elif i==len(duration_range_list)-1:
        duration_range_list[i][0] -= overlap
    else:
        duration_range_list[i][0] -= overlap
        duration_range_list[i][1] += overlap
'''
# Shift boundary to make overlap.
for i in range(len(duration_range_list)):
    duration_range_list[i][0] -= overlap
    duration_range_list[i][1] += overlap

duration_range_tuple = tuple(duration_range_list)
duration_range_tuple = tuple(tuple(element) for element in duration_range_list)


# Set root directory.
rootDir = setRootDir('macbook') # macbook, blueshark, or bladerunner.

# Set log.
fr_log_bufsize = 1 # 0 means unbuffered, 1 means line buffered.
fr_log_filename = 'fr' + year_str + '_' + size_start_str + '_' + size_end_str + '_' + size_step_str + '.log'
fr_log_path = rootDir +'log/' # For blueshark server.
fr_log_path_filename = fr_log_path + fr_log_filename
fr_log = open(fr_log_path_filename, 'w', fr_log_bufsize)
sys.stdout = fr_log

print('\nDuration range is: {}'.format(duration_range_tuple))
print('\nCheck the python version on Mac OS X, /usr/local/bin/python should be used:')
os.system('which python')
#homedir = os.environ['HOME']

year = int(year_str)
print('Searching fluxrope in {}.'.format(year_str))

# If year folder does not exist, create it.
if not os.path.exists(rootDir + 'output/' + year_str):
    os.makedirs(rootDir + 'output/' + year_str)

# =================================== Read and Check data =======================================
# Read in one year data.
print('Reading data...')
GS_AllData_DataFrame = pd.read_pickle(rootDir + 'input/GS_' + year_str + '_AllData_DataFrame_preprocessed.p')
# Check data property.
print('Checking DataFrame keys... {}'.format(GS_AllData_DataFrame.keys()))
print('Checking DataFrame shape... {}'.format(GS_AllData_DataFrame.shape))
print('Data Time start: {}'.format(GS_AllData_DataFrame.index[0]))
print('Data Time end: {}'.format(GS_AllData_DataFrame.index[-1]))
# Check the NaNs in each variable.
print('Checking the number of NaNs in DataFrame...')
len_GS_AllData_DataFrame = len(GS_AllData_DataFrame)
for key in GS_AllData_DataFrame.keys():
    num_notNaN = GS_AllData_DataFrame[key].isnull().values.sum()
    percent_notNaN = 100.0 - num_notNaN * 100.0 / len_GS_AllData_DataFrame
    print('The number of NaNs in {} is {}, integrity is {}%'.format(key, num_notNaN, round(percent_notNaN, 2)))

# Set search range. Though we read one year data, we may not want to search the whole year.

searchDatetimeStart = datetime(year,  1, 1, 0, 0, 0)
searchDatetimeEnd   = datetime(year, 12,31,23,59,59)

selectedRange_mask = (GS_AllData_DataFrame.index >= searchDatetimeStart) & (GS_AllData_DataFrame.index <= searchDatetimeEnd)
GS_DataFrame = GS_AllData_DataFrame.iloc[selectedRange_mask]
# Get Magnetic field slice.
B_DataFrame = GS_DataFrame.ix[:,['Bx', 'By', 'Bz']] # Produce a reference.
# Get the solar wind slice.
Vsw_DataFrame = GS_DataFrame.ix[:,['Vx', 'Vy', 'Vz']] # Produce a reference.
# Get the proton number density slice.
#Np_DataFrame = GS_DataFrame.ix[:,['Np']] # Produce a reference.

'''
# DO NOT DELETE THIS COMMENT.
# Good for insertion, not good for range selection. if datetime = datetime(year, 1,1,2,59,59), 
# returned index is from datetime(year, 1,1,2,59,0). However, if datetime = datetime(year, 1,1,2,59,0),
# returned index is from datetime(year, 1,1,2,58,0). Uncertain, do not use.
# Get the start and end DataFrame indices according to the start and end datetime.
index_start = GS_AllData_DataFrame.index.searchsorted(searchDatetimeStart)
index_end = GS_AllData_DataFrame.index.searchsorted(searchDatetimeEnd) # index_end not include searchDatetimeEnd.
# Get the records between start and end time from DataFrame.
GS_DataFrame = GS_AllData_DataFrame.iloc[index_start:index_end+1] #.iloc works on location.
'''

import warnings
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")

# Physics constants.
mu0 = 4.0 * np.pi * 1e-7 #(N/A^2) magnetic constant permeability of free space vacuum permeability
m_proton = 1.6726219e-27 # Proton mass. In kg.
dt = 60.0 # For 1 minute resolution data, the time increment is 60 seconds.
factor_deg2rad = np.pi/180.0 # Convert degree to rad.

flag_smoothA = True

# Multiprocessing
num_cpus = multiprocessing.cpu_count()
max_processes = num_cpus
print '\nTotol CPU cores on this node = ', num_cpus
# Create a multiprocessing pool with safe_lock.
pool = multiprocessing.Pool(processes=max_processes)
# Create a list to save result.
results = []

# Apply GS detection in sliding window.
# Set searching parameters.
theta_stepLength = 10 # In degree.
phi_stepLength = 2 * theta_stepLength # In degree.
# First integer in tuple is minimum duration threshold, second integer in tuple is searching window width.
# duration_range_tuple = ((20,30), (30,40), (40,50), (50,60)) #
print('\nDuration range tuple is: {}'.format(duration_range_tuple))
num_FluxRope = 0
totalStartTime = datetime.now()
for duration_range in duration_range_tuple: # Loop different window width.
    startTime = datetime.now()
   
    print('\n{}'.format(time.ctime()))
    minDuration = duration_range[0]
    maxDuration = duration_range[1]
    print('Duration : {} ~ {} minutes.'.format(minDuration, maxDuration))
    
    '''
    # Choose a flexible savgol filter window width based on the length of minDuration.
    half_minDuration = minDuration//2
    half_maxDuration = maxDuration//2
    if (half_minDuration) % 2 == 0: # filter window must be odd.
        savgol_filter_window = half_minDuration + 1
    else:
        savgol_filter_window = half_minDuration
    print('savgol_filter_window = {}'.format(savgol_filter_window))
    '''
    
    # The maximum gap tolerance is up to 30% of total points count.
    interp_limit = int(math.ceil(minDuration*3.0/10)) # Flexible interpolation limit based on window length.
    print('interp_limit = {}'.format(interp_limit))

    # Sliding window. If half_window=15, len(DataFrame)=60, range(15,45)=[15,...,44].
    for indexFluxRopeStart in xrange(len(B_DataFrame) - maxDuration): # in minutes.
        indexFluxRopeEnd = indexFluxRopeStart + maxDuration - 1  # The end point is included, so -1.
        
        # Grab the B slice within the window. Change the slice will change the original DataFrame.
        B_inWindow = B_DataFrame.iloc[indexFluxRopeStart : indexFluxRopeEnd+1] # End is not included.
        
        # If there is any NaN in this range, try to interpolate.
        if B_inWindow.isnull().values.sum():
            B_inWindow_copy = B_inWindow.copy(deep=True)
            # For example, limit=3 means only interpolate the gap shorter than 4.
            B_inWindow_copy.interpolate(method='time', limit=interp_limit, inplace=True)
            if B_inWindow_copy.isnull().values.sum():
                #print('Encounter NaN in B field data, skip this iteration.')
                continue # If NaN still exists, skip this loop.
            else:
                B_inWindow = B_inWindow_copy

        # Grab the Vsw slice within the window. Change the slice will change the original DataFrame.
        Vsw_inWindow = Vsw_DataFrame.iloc[indexFluxRopeStart : indexFluxRopeEnd+1] # End is not included.
        # If there is any NaN in this range, try to interpolate.
        if Vsw_inWindow.isnull().values.sum():
            Vsw_inWindow_copy = Vsw_inWindow.copy(deep=True)
            # limit=3 means only interpolate the gap shorter than 4.
            Vsw_inWindow_copy.interpolate(method='time', limit=interp_limit, inplace=True)
            if Vsw_inWindow_copy.isnull().values.sum():
                #print('Encounter NaN in Vsw data, skip this iteration.')
                continue # If NaN still exists, skip this loop.
            else:
                Vsw_inWindow = Vsw_inWindow_copy
                
        # Grab the Np slice within the window. Change the slice will change the original DataFrame.
        # Np_inWindow = Np_DataFrame.iloc[indexFluxRopeStart : indexFluxRopeEnd+1]
        
        # Calculate VHT in GSE frame.
        #VHT_inGSE = findVHT(B_inWindow, Vsw_inWindow) # Very slow.
        # Calculating VHT takes very long time(0.02748s for 14 data points), we use mean Vsw as VHT.
        VHT_inGSE = np.array(Vsw_inWindow.mean())
        
        # Return value: timeRange, Residue, orientation
        result_temp = pool.apply_async(searchFluxRopeInWindow, args=(B_inWindow, VHT_inGSE, theta_stepLength, phi_stepLength, minDuration, dt, flag_smoothA,))
        # print(result_temp.get()) # This statement will cause IO very slow.
        results.append(result_temp)
        # DO NOT unpack result here. It will block IO. Unpack in bulk.

    # Next we are going to save file We have to wait for all worker processes to finish.
    # Block main process to wait for worker processes to finish. This while loop will execute almost immediately when the innner for loop goes through. The inner for loop is non-blocked, so it finish in seconds.
    while len(pool._cache)!=0:
        #print('{} - Waiting... There are {} worker processes in pool.'.format(time.ctime(), len(pool._cache)))
        time.sleep(5)
    print('{} - len(pool._cache) = {}'.format(time.ctime(), len(pool._cache)))
    print('{} - Duration range {}~{} minutes is completed!'.format(time.ctime(), minDuration, maxDuration))

    # Save result. One file per window size.
    results_true_tuple_list = []
    results_false_tuple_list = []
    # Unpack results. Convert to tuple, and put into list.
    for one_result in results:
        results_tuple_temp = (one_result.get())
        #print(results_tuple_temp)
        if not np.isinf(results_tuple_temp[5]): # Check residue.
            #print(results_tuple_temp)
            if results_tuple_temp[4]: #if True, turn point on top.
                results_true_tuple_list.append(results_tuple_temp)
            else: # Turn point on bottom.
                results_false_tuple_list.append(results_tuple_temp)
    # Save results to pickle file. One file per window size.
    pickle_filename_true = rootDir + 'output/' + year_str + '_true_' + str(minDuration) + '~' + str(maxDuration) + 'min.p'
    print('Save file to: {}'.format(pickle_filename_true))
    pickle.dump(results_true_tuple_list, open(pickle_filename_true, 'wb'))
    pickle_filename_false = rootDir + 'output/' + year_str + '_false_' + str(minDuration) + '~' + str(maxDuration) + 'min.p'
    print('Save file to: {}'.format(pickle_filename_false))
    pickle.dump(results_false_tuple_list, open(pickle_filename_false, 'wb'))
    # Empty container results[].
    results = []

    endTime = datetime.now()
    time_spent_in_seconds = (endTime - startTime).total_seconds()
    print('Time spent on this window: {} seconds ({} hours).'.format(time_spent_in_seconds, time_spent_in_seconds/3600.0))

# Close pool, prevent new worker process from joining.
pool.close()
# Block caller process until workder processes terminate.
pool.join()

totalEndTime = datetime.now()
time_spent_in_seconds = (totalEndTime - totalStartTime).total_seconds()
print('\n{}'.format(time.ctime()))
print('All duration ranges are completed!')
print('Number of CPU cores per node: {}.'.format(num_cpus))
print('Max number of workder processes in pool: {}.'.format(max_processes))
print('Total Time spent: {} seconds ({} hours).'.format(time_spent_in_seconds, time_spent_in_seconds/3600.0))

exit()

# Load pickle file.
result = pickle.load(open(pickle_filename_true, 'rb'))
for item in result:
    print(item)

