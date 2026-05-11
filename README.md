# DQC Emulator Benchmarking

A benchmarking framework for **Distributed Quantum Computing (DQC) emulators**, using a distributed implementation of the inverse Quantum Fourier Transform (QFT†) as the test case. The benchmark evaluates emulators across execution time, memory usage, and fidelity relative to a monolithic baseline.

This repository accompanies the paper:

> *Benchmarking Distributed Quantum Computing Emulators*, G. Díaz-Camacho et al., PCCC Galicia 2025.

---

## Overview

The benchmark partitions the inverse QFT across quantum nodes using gate teleportation (cat-entangler/disentangler circuits). Four emulators are evaluated:

| Emulator | Version | Python | Conda env | Role in study |
|---|---|---|---|---|
| **Qiskit Aer** | 0.17.1 | 3.10 | `qiskit_conda` | Monolithic baseline adapted for DQC |
| **SquidASM** | 0.13.4 | 3.8 | `squidasm_conda` | Full discrete-event DQC emulator |
| **Interlin-q** | 0.0.1.post5 | 3.10 | `interlinq_conda` | Automatic circuit partitioning prototype |
| **SQUANCH** | 1.1.0 | 3.10 | `squanch_conda` | Agent-based lightweight DQC tool |

Each emulator runs in its own isolated conda environment to avoid dependency conflicts.

> **Note on SquidASM / NetSquid:** SquidASM depends on NetSquid, which requires a free account and license from [QuTech](https://netsquid.org). Register and install NetSquid before creating the `squidasm_conda` environment.

---

## Repository Structure

```
DQC-Emulator-Benchmarking/
│
├── configs/
│   └── sweep_params.yaml              # Parameter sweep configuration
│
├── emulators/
│   ├── qiskit_aer/
│   │   ├── env/environment.yml        # Pinned conda environment
│   │   ├── scripts/
│   │   │   └── qft_multinode.py       # Distributed inverse QFT implementation
│   │   ├── slurm_template_multinode.sh
│   │   ├── jobs/                      # Auto-generated SLURM scripts
│   │   └── results/                   # CSV output files 
│   │
│   ├── squidasm/
│   │   ├── env/environment.yml
│   │   ├── scripts/
│   │   │   ├── AbstractNode.py        # Base class: quantum node with EPR/teleportation helpers
│   │   │   ├── config_template.yaml   # NetSquid network topology template (auto-expanded at runtime)
│   │   │   ├── config-QFT-noiseless.yaml
│   │   │   ├── config-QFT-noisy.yaml
│   │   │   ├── qft_multinode.py       # Multi-node distributed QFT
│   │   ├── slurm_template_multinode.sh
│   │   ├── jobs/
│   │   └── results/
│   │
│   ├── interlin-q/
│   │   ├── env/environment.yml
│   │   ├── scripts/
│   │   │   └── qft_multinode.py
│   │   ├── slurm_template_multinode.sh
│   │   ├── jobs/
│   │   └── results/
│   │
│   └── squanch/
│       ├── env/environment.yml
│       ├── scripts/
│       │   └── qft_multinode.py
│       ├── slurm_template_multinode.sh
│       ├── jobs/
│       └── results/
│
├── scripts/
│   ├── job_creator.py                 # Generates SLURM scripts from sweep_params.yaml
│   ├── job_submitter_all.sh           # Submits all jobs for all emulators
│   └── job_submitter_one.sh           # Submits all jobs for a single emulator
│
└── plots/
    ├── plot_results.py                # Parses CSVs and generates figures
```

---

## Prerequisites

- Linux (tested on RHEL on the QMIO/FinisTerrae III cluster at CESGA)
- [Conda](https://docs.conda.io/en/latest/) (Miniconda or Anaconda)
- SLURM workload manager (for HPC use; scripts can be run directly for local testing)
- A NetSquid account for SquidASM — register at [netsquid.org](https://netsquid.org) before creating the environment

---

## Environment Setup

Each emulator has its own pinned conda environment. Create them individually:

```bash
conda env create -f emulators/qiskit_aer/env/environment.yml
conda env create -f emulators/squidasm/env/environment.yml
conda env create -f emulators/interlin-q/env/environment.yml
conda env create -f emulators/squanch/env/environment.yml
```

---

## Running Benchmarks

### 1. Configure the parameter sweep

Edit `configs/sweep_params.yaml`:

```yaml
emulators: ['qiskit_aer', 'squidasm', 'interlin-q', 'squanch']
num_qubits_range: [4, 6, 8, 10, 12, 14, 16, 18, 20]
num_nodes_range: [1, 2, 4, 8]
theta_range: [0.0, 0.33, 0.66]
shots: 100
```

> **Interlin-q constraint:** `num_qubits` must be divisible by `num_nodes`.

### 2. Generate SLURM job scripts

Run from the repository root:

```bash
python scripts/job_creator.py
```

This reads `configs/sweep_params.yaml` and writes one `.sh` script per parameter combination into `emulators/<name>/jobs/`, filling in positional placeholders from each emulator's SLURM template:

| Placeholder | Value |
|---|---|
| `$1` | `num_qubits` |
| `$2` | `num_nodes` |
| `$3` | `theta` |
| `$4` | `shots` |

### 3. Submit jobs

Submit all emulators at once:

```bash
bash scripts/job_submitter_all.sh
```

Or a single emulator:

```bash
bash scripts/job_submitter_one.sh qiskit_aer
# Valid names: qiskit_aer, squidasm, interlin-q, squanch
```

### 4. Run locally (without SLURM)

Activate the environment and run the script directly from its `scripts/` directory:

```bash
conda activate qiskit_conda
cd emulators/qiskit_aer/scripts
python qft_multinode.py --num_qubits 8 --num_nodes 2 --theta 0.33 --shots 100
```

For SquidASM, `qft_multinode.py` auto-generates a temporary NetSquid network config at runtime from `config_template.yaml`, which must be present in the same directory.

---

## Output Format

Each run writes a CSV to `emulators/<name>/results/<name>_<nodes>_q<qubits>_a<theta>.csv`:

| Column | Description |
|---|---|
| `Emulator` | Emulator name |
| `Num Qubits` | Total logical qubits |
| `Nodes` | Number of distributed nodes |
| `Theta` | Phase angle for Fourier state preparation |
| `Execution Time (s)` | Wall-clock simulation time |
| `Peak Memory (KB)` | Peak memory via `tracemalloc` |
| `Fidelity` | Classical fidelity vs. monolithic reference |

---

## Plotting Results

Once results are collected:

```bash
cd plots
python plot_results.py
```

The script scans all `emulators/*/results/*.csv`, aggregates them, and produces the scaling figures. An interactive version is available in `emulators/benchmark_multinode_figures.ipynb`.

---

## Hardware

Benchmarks were run on the **QMIO HPC cluster** at [CESGA](https://www.cesga.es):

- Nodes: 2× Intel Xeon Ice Lake 8352Y (64 cores/node)
- RAM: 256 GB per node
- Scheduler: SLURM
- Per job: 1 core, 4 GB RAM, up to 6 h walltime

---

## Extending the Framework

To add a new emulator:

1. Create `emulators/<new_emulator>/` with the standard subdirectory layout (`scripts/`, `env/`, `jobs/`, `results/`).
2. Add an `environment.yml` pinning all dependencies.
3. Implement `scripts/qft_multinode.py` with the same CLI (`--num_qubits`, `--num_nodes`, `--theta`, `--shots`) and CSV output format.
4. Add a `slurm_template_multinode.sh` using `$1`–`$4` placeholders.
5. Add the emulator name to `configs/sweep_params.yaml`.

---

## Citation

If you use this benchmarking suite, please cite:

```bibtex
@article{diaz2025dqcbenchmark,
  title   = {Benchmarking Distributed Quantum Computing Emulators},
  author  = {D{\'i}az-Camacho, Guillermo and Llovo, Iago F. and Cardama, F. Javier
             and Bautista, Irais and Fa{\'i}lde, Daniel and Mussa Juane, Mariamo
             and V{\'a}zquez-P{\'e}rez, Jorge and Costas, Natalia
             and Pena, Tom{\'a}s F. and G{\'o}mez, Andr{\'e}s},
  journal = {PCCC Galicia 2025},
  year    = {2025}
}
```

---

## Acknowledgements

This work was supported by MICINN through the European Union NextGenerationEU recovery plan (PRTR-C17.I1) and the Galician Regional Government through "Planes Complementarios de I+D+I con las Comunidades Autónomas" in Quantum Communication. The authors acknowledge CESGA for access to FinisTerrae III and the QMIO quantum computing infrastructure.

---

## License

See `LICENSE` for details.
