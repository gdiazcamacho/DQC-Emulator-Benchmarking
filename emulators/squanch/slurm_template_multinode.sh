#!/bin/bash
#SBATCH --job-name={{JOB_NAME}}
#SBATCH --output=../jobs/%x_%j.out
#SBATCH --error=../jobs/%x_%j.err
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem={{MEM_GB}}G

# Provenance stamped by scripts/job_creator.py; written into every results row.
export DQC_COMMIT={{COMMIT}}
export DQC_CAMPAIGN={{CAMPAIGN}}
export LC_ALL=C

# Load the necessary modules
module purge
module load miniconda3

# Activate the conda environment for the emulator
conda activate squanch_conda

# The algorithm selects WHICH script runs; the communication mode selects a
# branch INSIDE that script.
python ../scripts/{{ALGORITHM}}_multinode.py \
    --num_qubits={{NUM_QUBITS}} \
    --num_nodes={{NUM_NODES}} \
    --theta={{THETA}} \
    --shots={{SHOTS}} \
    --comm_mode={{COMM_MODE}} \
    --noise={{NOISE}}

# Summarize resource usage with seff
seff $SLURM_JOBID
