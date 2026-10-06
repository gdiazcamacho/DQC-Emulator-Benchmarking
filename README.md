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
├── common/
│   └── dqc_common.py                  # Shared partitioning, fidelity and CSV helpers
│                                      # (imported by all four emulator scripts)
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
│   ├── squanch/
│   │   ├── env/environment.yml
│   │   ├── scripts/
│   │   │   └── qft_multinode.py
│   │   ├── slurm_template_multinode.sh
│   │   ├── jobs/
│   │   └── results/
│   │
│   └── cunqa/                          # In the sweep; shadow-validated, not yet run live
│       ├── env/README.md               # CUNQA is built from source, not conda/pip
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
├── tests/
│   └── test_primitives.py             # Gate identities, partitioning, fidelity
│                                      # (NumPy only; no emulator install needed)
│
└── plots/
    ├── plot_results.py                # Parses CSVs and generates figures
```

---

## Emulator status

| Emulator | In default sweep | Noise | Notes |
|---|---|---|---|
| Qiskit Aer | Yes | Yes | Reference / monolithic baseline |
| SquidASM | Yes | Yes | NetSquid-based |
| Interlin-q | Yes | Yes | `num_qubits` must be divisible by `num_nodes` |
| SQUANCH | Yes | Yes | |
| CUNQA 3.1.0 (pinned, built from source: `emulators/cunqa/env/`) | Yes (`basic`, `packed`, `dynamic`) | **No** (no noise on communication paths upstream) | Shadow-validated only (`--shadow`, `tests/test_cunqa_adapter.py`); hand-check 4q/2n on QMIO before trusting the sweep. Memory column is the driver process only. Quantum-comm points simulate ONE global state of N+M qubits; 8-node 18-20-qubit points take hours |

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
min_qubits_per_node: 2
```

> **Interlin-q constraint:** `num_qubits` must be divisible by `num_nodes`.

**Grid feasibility.** The sweep is a full cartesian product, so it contains
points that are not valid distributed benchmarks:

- *Impossible* — `num_qubits < num_nodes` (e.g. 4 qubits over 8 nodes). These
  are always excluded; earlier versions accepted them and produced partitions
  containing zero-qubit nodes.
- *Degenerate* — a node holding a single qubit performs no local controlled
  rotations, so the run measures communication overhead against an empty local
  circuit. `min_qubits_per_node` controls these: `2` (default) excludes them,
  `1` reproduces the original grid.

`job_creator.py` prints every skipped combination and the reason, so the
exclusions are visible rather than silent. With the defaults above, 336 of the
432 grid points are emitted.

### Telegate packing ablation

`emulators/qiskit_aer/scripts/qft_multinode.py` accepts `--telegate_baseline`,
which switches the distributed ladder from the packed cat-comm construction
(one Bell pair per column) to a naive baseline (one Bell pair per individual
gate, no packing). Not part of the sweep grid — it's a separate, small
comparison run producing `plots/ebit_packing_ablation.png`:

```bash
python emulators/qiskit_aer/scripts/qft_multinode.py \
    --num_qubits 8 --num_nodes 2 --theta 0.33 --shots 100                     # packed (default)
python emulators/qiskit_aer/scripts/qft_multinode.py \
    --num_qubits 8 --num_nodes 2 --theta 0.33 --shots 100 --telegate_baseline # baseline
```

Both write to `emulators/qiskit_aer/results/`, distinguished by the
`Algorithm` column (`qft` vs. `qft_telegate_baseline`) so they never mix into
the main scaling figures. The `Ebits` column records the number of Bell pairs
consumed. In the bipartite (2-node) case this fits exactly: packed = n/2,
baseline = (n/2)², verified against hand-derived counts at n = 4, 6, 8, 10
before being trusted (see `CHANGELOG.md`). The multi-node case has not yet had
the same check.

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
python emulators/qiskit_aer/scripts/qft_multinode.py \
    --num_qubits 8 --num_nodes 2 --theta 0.33 --shots 100
```

All scripts resolve paths relative to their own location, so they can be
launched from the repository root, from their own directory, or from a SLURM
working directory. Results are always written to
`emulators/<name>/results/`.

For SquidASM, `qft_multinode.py` auto-generates a temporary NetSquid network config at runtime from `config_template.yaml`, which must be present in the same directory.

---

## Output Format

Each run writes a **single-row** CSV to
`emulators/<name>/results/<algorithm>_<name>_<nodes>_q<qubits>_a<theta>.csv`:

| Column | Description |
|---|---|
| `Algorithm` | Benchmark discriminator (`qft`, and future additions) |
| `Emulator` | Emulator name |
| `Nodes` | Number of distributed nodes |
| `Num Qubits` | Total logical qubits |
| `Theta` | Phase angle for Fourier state preparation |
| `Shots` | Number of repetitions |
| `Execution Time (s)` | Wall-clock simulation time |
| `Current Memory (KB)` | Memory at end of run via `tracemalloc` |
| `Peak Memory (KB)` | Peak memory via `tracemalloc` |
| `Fidelity` | Classical fidelity vs. the analytic inverse-QFT distribution |
| `Result` | Measured phase histogram |

Two conventions matter and are enforced by `common/dqc_common.write_result_row`:

- **One row per run.** `plots/plot_results.py` concatenates every CSV under
  `emulators/*/results/`, so a file with any other shape corrupts the combined
  DataFrame silently. Auxiliary outputs that are not one-row-per-run (for
  example cost landscapes) belong in a subdirectory such as
  `results/landscapes/`, outside that glob.
- **`Algorithm` is always written.** Without it, adding a second benchmark
  contaminates the QFT figures. Files predating this column are backfilled as
  `qft` when parsed.

`Fidelity` is computed per run, at the point where the raw counts exist, rather
than reconstructed later from the stringified histogram.

---

## Plotting Results

Once results are collected:

```bash
python plots/plot_results.py --save-path plots/scaling_2panel
```

Options:

| Flag | Default | Meaning |
|---|---|---|
| `--base-dir` | repo-relative `emulators/` | Where to look for result CSVs |
| `--algorithm` | `qft` | Which benchmark to plot |
| `--save-path` | none (interactive) | Path prefix for saved `.png` / `.svg` |

The script scans `emulators/*/results/*.csv`, filters by `Algorithm`, and
produces the scaling and fidelity figures. An interactive version is available
in `plots/benchmark_multinode_figures.ipynb`.

---

## Tests

```bash
python tests/test_primitives.py          # or: python -m pytest tests/ -v
```

The tests require only NumPy — no emulator installation and no NetSquid licence
— so they run from a fresh clone. They cover the controlled-phase gate
identities used by the distributed ladder, qubit partitioning, and the analytic
inverse-QFT reference distribution against which `Fidelity` is computed. The
Qiskit end-to-end check skips cleanly if Qiskit is absent.

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

Released under the MIT License. See `LICENSE` for the full text.
