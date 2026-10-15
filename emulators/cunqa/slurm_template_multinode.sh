#!/bin/bash
#SBATCH --job-name={{JOB_NAME}}
#SBATCH --output=../jobs/%x_%j.out
#SBATCH --error=../jobs/%x_%j.err
#SBATCH --time=2-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem={{MEM_GB}}G

# Provenance stamped by scripts/job_creator.py; written into every results row.
export DQC_COMMIT={{COMMIT}}
export DQC_CAMPAIGN={{CAMPAIGN}}
export LC_ALL=C

# This allocation is for the DRIVER only. qft_multinode.py calls qraise, which
# submits a SEPARATE allocation for the vQPUs (quantum_comm for packed/basic,
# classical_comm for dynamic) with a generated backend JSON and a memory
# request sized from the statevector (see default_mem_per_qpu_gb), then qdrop.
# The driver blocks until the vQPU run ends, so its --time must cover the
# qraise budget (default_max_time caps that at 2 days).
# Simulation memory is therefore NOT in this job's seff: run seff on the qraise
# job id (printed by qraise) if you need it.
#
# Nested submission (qraise's sbatch from inside this job) works on QMIO
# (verified 2026-10-08, job 561229). It is site-configuration dependent
# elsewhere; if qraise fails here, run the same command from a login node.

# Same environment as install_cunqa.sh (modules, CUNQA_PREFIX, STORE, PATH).
source ../env/cunqa_env.sh

python3 ../scripts/{{ALGORITHM}}_multinode.py \
    --num_qubits={{NUM_QUBITS}} \
    --num_nodes={{NUM_NODES}} \
    --theta={{THETA}} \
    --shots={{SHOTS}} \
    --comm_mode={{COMM_MODE}} \
    --noise={{NOISE}}

seff $SLURM_JOBID
