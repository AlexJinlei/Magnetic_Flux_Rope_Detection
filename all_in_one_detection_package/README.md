# DESCRIPTION
This is a magnetic flux rope all in one detection package. This package is a wrap-up of all the functions contained in other folders located in the same directory as all_in_one_detection_package folder. This all in one detection package is intended to used for short range \(several hours \~ several days\) detection. All the function is intended to run on local computer.
# CODE LIST
## 1. FUNCTION MODULE
### MyPythonPackage
This module contains all functions that is needed for flux rope detection.
### fluxrope_check_given_range.py
This code can detect the flux ropes within the given time range.
### fluxrope_check_shock_downstream.py
This code can detect the flux ropes downstream the shock \(observed by Wind or ACE spacecraft\) with in the given time range. Shock list is needed.
### fluxrope_check_shock_downstream_ACE_and_WIND.py
This code can detect the flux ropes downstream the shock \(observed by Wind or ACE spacecraft\) with in the given time range. Shock list is needed. The differce with fluxrope_check_shock_downstream.py is that this code check shock observed by both Wind and ACE.

## 2. USAGE
Specify the spacecraft, the start time, and end time in the code. The run the python code.

## 3. RESULT
The following plot is the detection result which show the detected flux ropes observed by ACE and Wind spacecrafts. The yellow shaded intervals represent flux rope time range. 

![result](all_in_one_detection_package/result_sample/combined/200208181810.png)
