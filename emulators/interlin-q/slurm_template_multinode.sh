#!/bin/bash
#SBATCH --job-name=interlinq_multi
#SBATCH --output=../jobs/%x_%j.out
#SBATCH --error=../jobs/%x_%j.err
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4  # Allocate 4 CPU cores
#SBATCH --mem-per-cpu=4G

# Load the necessary modules
module purge
module load miniconda3  # Adjust this to the correct module name if needed

# Activate the conda environment for the emulator
conda activate interlinq_conda

# Run the Python script
python ../scripts/qft_multinode.py --num_qubits=$1 --num_nodes=$2 --theta=$3 --shots=$4

# Summarize resource usage with seff
seff $SLURM_JOBID

