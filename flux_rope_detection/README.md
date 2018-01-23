# Flux Rope Detection
## 1. DESCRIPTION
This code is the core module which applys the flux rope detection algorithm based on Grad–Shafranov reconstruciton technique. Please read the related papers mentioned in README.md file under the root directory of this project to get familiar with the Grad–Shafranov reconstruciton technique. Also please read the comment in the code to learn the implementation details.
## 2. COMMAND LINE ARGUMENTS
Command format: python GS_detectFluxRope_multiprocessing.py 1996 10 50 5 1  
First parameter is year, second is minimum size, third is maxmum size, fourth is size step, fifth is overlap width.  
The command above generate the following size tuple:  
((9, 16), (14, 21), (19, 26), (24, 31), (29, 36), (34, 41), (39, 46), (44, 51))  
If you input command as "python GS_detectFluxRope_multiprocessing.py 1996 10 50 5 0", then the generated duration tuple is: ((10, 15), (15, 20), (20, 25), (25, 30), (30, 35), (35, 40), (40, 45), (45, 50))  
## 3. INPUT AND OUTPUT
### Input
The input file has the name format GS_1996_AllData_DataFrame_preprocessed.p, where 1996 is year. It contains all needed data to detect flux ropes.
### Output
The output files have the name format 1996_true_9~16min.p or 1996_false_9~16min.p, where 1996 is year, 9~16min is the duration range in minutes, true or false indicate whether they can be flux rope candidate. All results labeled as true have turning point on top, and al results labeled as false have turning point on bottom, which means the structure center has lower total pressure than the outer part, which does not consistent with flux rope feature.
