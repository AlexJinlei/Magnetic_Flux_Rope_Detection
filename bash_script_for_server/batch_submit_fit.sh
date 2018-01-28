#!/bin/sh
# 2017-02-09
# For blueshark server.
# This script creates a bash script first, and then submit this script to slurm by sbatch command.


nodes=1
ntasks=24
mem="3G" #Allocate 4GB for 24 cores, allocate 6GB for 40 cores.
time="7-00:00:00"
partition="long"

# Duration range, must be in python tuple format, no blank space.
duration="((30,40),(40,50))" # No blank space within string. The trailing "," is required if there is only one element in tuple, i.e., "((60,80),)"

# Root Directory
rootDir="/your_path/" # For blueshark.

# Log path.
log_path="${rootDir}log/"
# script_filename path
script_filename_path="${rootDir}sh_script/"

# Create and execute sbatsh scripte for each year in loop.
#for year in 1996
for year in {1996..2016}
    do
        # Create a temp script file for sbatch.
        script_filename="${script_filename_path}fr${year}_${ntasks}core_${duration}.sh"
        # Check if file already exists. If it is, delete.
        if [ -f $script_filename ]; then
            rm $script_filename
        fi

        job_name="fr${year}${duration}"
        error="${log_path}fr${year}${duration}.err"
        output="${log_path}fr${year}${duration}.out"

        # Write content to temp file.
        echo "#!/bin/sh" >> $script_filename
        echo "#SBATCH --job-name ${job_name}" >> $script_filename
        echo "#SBATCH --error=${error}" >> $script_filename
        echo "#SBATCH --output=${output}" >> $script_filename

        echo "#SBATCH --time=${time}" >> $script_filename
        echo "#SBATCH --nodes ${nodes}" >> $script_filename
        echo "#SBATCH --ntasks ${ntasks}" >> $script_filename
        echo "#SBATCH --partition=${partition}" >> $script_filename
        echo "#SBATCH --mem=${mem}" >> $script_filename

        echo "date">> $script_filename
        echo "python GS_detectFluxRope_multiprocessing.py ${year} '${duration}'">> $script_filename
        echo "date">> $script_filename

        # Submit script file to slurm.
        sbatch $script_filename
    done

