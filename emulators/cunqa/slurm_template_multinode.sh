#!/bin/bash
#SBATCH --job-name={{JOB_NAME}}
#SBATCH --output=../jobs/%x_%j.out
#SBATCH --error=../jobs/%x_%j.err
#SBATCH --time=2-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G

# This allocation is for the DRIVER only. qft_multinode.py calls qraise, which
# submits a SEPARATE allocation for the vQPUs (quantum_comm for packed/basic,
# classical_comm for dynamic) with a generated backend JSON and a memory
# request sized from the statevector (see default_mem_per_qpu_gb), then qdrop.
# The driver blocks until the vQPU run ends, so its --time must cover the
# qraise budget (default_max_time caps that at 2 days).
# Simulation memory is therefore NOT in this job's seff: run seff on the qraise
# job id (printed by qraise) if you need it.
#
# Nested submission (sbatch from inside a running job) is site-configuration
# dependent and unverified on QMIO / FinisTerrae III. If qraise fails here, run
# the same command from a login node instead -- the script is identical.

module load cunqa/3.0.0 2>/dev/null || module load cunqa 2>/dev/null || true

python ../scripts/{{ALGORITHM}}_multinode.py \
    --num_qubits={{NUM_QUBITS}} \
    --num_nodes={{NUM_NODES}} \
    --theta={{THETA}} \
    --shots={{SHOTS}} \
    --comm_mode={{COMM_MODE}}

seff $SLURM_JOBID
