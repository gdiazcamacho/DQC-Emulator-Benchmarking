#!/bin/bash

# Base directory containing the emulators
base_dir="emulators"

# Loop through each emulator directory
for emulator_dir in "$base_dir"/*; do
    # Check if the directory exists and contains a jobs directory
    if [ -d "$emulator_dir/jobs" ]; then
        # Loop through each .sh file in the jobs directory
        for job_script in "$emulator_dir/jobs"/*.sh; do
            # Check if the file exists
            if [ -f "$job_script" ]; then
                # Submit the job using sbatch and set the working directory
                sbatch -D "$emulator_dir/jobs" "$job_script"
                echo "Submitted job script: $job_script (working dir: $emulator_dir/jobs)"
            fi
        done
    fi
done

