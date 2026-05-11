#!/bin/bash
#SBATCH --job-name=squid_two
#SBATCH --output=../jobs/%x_%j.out
#SBATCH --error=../jobs/%x_%j.err
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4  # Allocate 4 CPU cores
#SBATCH --mem-per-cpu=8G

# Load the necessary modules
module purge
module load miniconda3  # Adjust this to the correct module name if needed

# Activate the conda environment for the emulator
conda activate squidasm_conda

# Run the Python script
python ../scripts/qft_two_node.py --num_qubits=$1 --theta=$2 --shots=$3

# Summarize resource usage with seff
seff $SLURM_JOBID

