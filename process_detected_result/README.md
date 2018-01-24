# Process Detected Result
## 1. DESCRIPTION
Once you finishe the flux rope detection step, you will end up with a huge amount of raw records. Thes records are saved in the files named by \[year\]_true_\[x\]\~\[x\]min.p \(See README.md under flux_rope_detection folder\). Detecting flux rope requires lots of CPU resources, so usually the task is done on cluster computer by parallel computing. For the sake of efficiency, the flux rope detection code minimize the IO workload. As a result, the output records contain only ciritical information, such as flux rope start and end time, and the duration. After we get these records, we have to do more calculation to get more flux rope information.

This package contains all needed code to process raw data. Firstly combine all raw data in one year into one single file. Then we clean the records, clean the records with overlapping time range. After that we do some calculation to get more flux rope information. The last step, we apply some selection criteria to refine the result. 

## 2. EXECUTION ORDER
### 1) GS_combine_raw_result_to_single_file.py
Input the raw results, output the combined raw results. 
### 2) GS_combineDuplicatedEvent_pickle.py
Input the combined raw resluts, ouput flux rope list without overlapped records.
### 3) GS_getMoreFluxRopeInfo.py
Input the non-overlapped records, output flux rope record list with detailed information.
### 4) GS_applyMoreCriteria_addMoreEntry.py
Input the flux rope record list with detailed information, add more entries, such as wait time, shock time, days to nearest HCS, days to reconnection exhaust. Then apply some criteria, such as duration. Output the flux rope record list with more detailed information.


