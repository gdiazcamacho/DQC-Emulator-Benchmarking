#!/bin/bash

# Check if emulator name is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <emulator_name>"
    exit 1
fi

emulator="$1"
emulator_dir="emulators/$emulator"
jobs_dir="$emulator_dir/jobs"

# Check if the jobs directory exists
if [ ! -d "$jobs_dir" ]; then
    echo "No jobs directory found for emulator: $emulator"
    exit 1
fi

# Submit all .sh job scripts for the specified emulator
for job_script in "$jobs_dir"/*.sh; do
    if [ -f "$job_script" ]; then
        sbatch -D "$jobs_dir" "$job_script"
        echo "Submitted job script: $job_script (working dir: $jobs_dir)"
    fi
done

