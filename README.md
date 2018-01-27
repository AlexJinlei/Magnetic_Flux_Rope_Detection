# Magnetic Flux Rope Detection
## 1. DESCRIPTION
This is an automated magnetic flux rope detection package. This project is part of my PhD research. The main purpose of this project is to automatically detect the small-scale magnetic flux ropes from in-situ spacecraft obervational data.
## 2. SPACE PHYSICS BACKGROUND
To get farmiliar with the physics of magnetic flux ropes, please refer to my PhD dissertation: [OBSERVATIONAL ANALYSIS OF TRANSIENT AND COHERENT STRUCTURES IN SPACE PLASMAS](phd_dissertation/). It will be discoverable soon on ProQuest dissertation database. I have one journal paper and two conference proceedings which talk about the statistical analysis and case studies on small-scale magnetic flux ropes. These papers are in the same folder of my dissertation.
## 3. CORE ALGORITHM FLOWCHART
Following is the flowchart of core detection algorithm. Please refer to chapter 2 and chapter 3 of my dissertation for detail.
![flowchart](phd_dissertation/GS_flowchart_www_draw_io_v3_1.png)
## 4. CODE SETS
### 1) Code Used for Large Scale Detection
This project contains two sets of code. One set is used for detecting huge number of small-scale flux ropes across tens of years. The core detection code is high CPU intensive, which is intended to run on cluster server. Other tasks, such as downloading data, preprocessing data, postprocessing detected results, generating website, need less computing resource, which are intended to run on desktop computer or laptop computer.  This code set contains 4 folders: a) [data_processing](data_processing/), b) [flux_rope_detection](\flux_rope_detection), c) [process_detected_result](process_detected_result/), d) [generate_website](generate_website/).
- a) 

