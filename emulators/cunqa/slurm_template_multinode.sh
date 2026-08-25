#!/bin/bash
#SBATCH --job-name={{JOB_NAME}}
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G

# This job requests minimal resources for the *driver* process only. The
# driver calls qraise (inside qft_multinode.py), which submits its own SLURM
# allocation for the vQPU servers -- so the actual simulation resources are
# requested separately by qraise, not by this #SBATCH header.
#
# NOTE: nested SLURM job submission (a job that itself calls qraise/sbatch)
# is site-configuration-dependent. This has not been validated against
# CESGA's actual scheduler config -- if `qraise` fails to submit from within
# a running job here, run job_creator.py's CUNQA jobs interactively from a
# login node instead, or ask CESGA support whether nested submission is
# permitted on QMIO / FinisTerrae III.

module load cunqa 2>/dev/null || true   # adjust to the actual module name if different
conda activate cunqa_conda 2>/dev/null || true

python ../scripts/{{ALGORITHM}}_multinode.py \
    --num_qubits={{NUM_QUBITS}} \
    --num_nodes={{NUM_NODES}} \
    --theta={{THETA}} \
    --shots={{SHOTS}} \
    --comm_mode={{COMM_MODE}}

# Summarize resource usage with seff. Note this covers the driver process
# above; it does NOT include the vQPU server processes qraise spawns as a
# separate allocation (see script header for why memory figures are
# driver-only for this emulator).
seff $SLURM_JOBID
