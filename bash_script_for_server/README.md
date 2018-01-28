# BASH SCRIPT FOR JOB SUBMITTING TO SERVER
## 1. DESCRIPTION
To simplify the job submitting process, I wrote this script. This script will generate bash script according to the parameters specified by user, and then submit the bash script to Slurm Workload Manager automatically. 
## 2. USAGE
You should be familiar with the hardware and software configuration of the server that you are going to use. If you have no idea on them, please consult your administrator. You may want to make changes to this code according to specific platform.

[batch_auto_submit_job.sh](/batch_auto_submit_job.sh) is the bash script to generate bash file to be submitted to Slurm.  
[fr1996_24core_((60,80),(80,100)).sh](/fr1996_24core_((60,80),(80,100)).sh) is a sample code created by batch_auto_submit_job.sh.
