#!/bin/bash
#SBATCH --job-name={{JOB_NAME}}
#SBATCH --output=../jobs/%x_%j.out
#SBATCH --error=../jobs/%x_%j.err
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G

# Load the necessary modules
module purge
module load miniconda3

# Activate the conda environment for the emulator
conda activate interlinq_conda

# The algorithm selects WHICH script runs; the communication mode selects a
# branch INSIDE that script.
python ../scripts/{{ALGORITHM}}_multinode.py \
    --num_qubits={{NUM_QUBITS}} \
    --num_nodes={{NUM_NODES}} \
    --theta={{THETA}} \
    --shots={{SHOTS}} \
    --comm_mode={{COMM_MODE}}

# Summarize resource usage with seff
seff $SLURM_JOBID
