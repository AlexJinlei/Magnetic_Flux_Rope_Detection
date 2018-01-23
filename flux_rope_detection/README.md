# Flux Rope Detection
## 1. DESCRIPTION
This code is the core module which applys the flux rope detection algorithm based on Grad–Shafranov reconstruciton technique. Please read the related papers mentioned in README.md file under the root directory of this project to get familiar with the Grad–Shafranov reconstruciton technique. Also please read the comment in the code to learn the implementation details.
## 2. Command line arguments
Command format: GS_detectFluxRope_multiprocessing.py 1996 10 50 5  
First parameter is year, second is minimum size, third is maxmum size, fourth is size step.  
The command above generate the following size tuple:  
((10, 15), (15, 20), (20, 25), (25, 30), (30, 35), (35, 40), (40, 45), (45, 50))
