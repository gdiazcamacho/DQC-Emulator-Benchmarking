# Running a benchmark campaign

A campaign is one set of results that can go in the paper: produced by one
tagged commit, from a clean checkout, with every run recording that commit.
`campaign-v1` is the first one; it supersedes every result produced before it
(the preprint/EuroQHPC numbers predate the SQUANCH channel fix, the CP fix,
the communication modes and CUNQA).

## What is measured

One algorithm, the distributed inverse QFT, run with three communication
schedules (plus Interlin-q's own):

| Mode | Cross-node rotation | Quantum communication | E-bits |
|---|---|---|---|
| `basic` | one telegate per gate | yes | number of cross-node gates |
| `packed` | one telegate per control qubit per lower node | yes | sum_i size_i * i |
| `dynamic` | measure, send the bit, classically conditioned phase | **no** | 0 |
| `native` | Interlin-q partitions and teleports itself | yes | not comparable |

`dynamic` is the semiclassical (deferred-measurement) iQFT. It is valid only
because the transform is immediately measured.

| Emulator | Modes | Noise models |
|---|---|---|
| Qiskit Aer | basic, packed, dynamic | none, depolarizing |
| SquidASM | packed, dynamic | none, device |
| SQUANCH | basic, packed, dynamic | none |
| CUNQA 3.1.0 | basic, packed, dynamic | none |
| Interlin-q | native | none |

## Waves

| Wave | Config | Content | Jobs |
|---|---|---|---|
| 1 | `configs/campaign_v1_wave1_noiseless.yaml` | every emulator and mode, noiseless | 840 |
| 2 | `configs/campaign_v1_wave2_noisy.yaml` | Aer (depolarizing), SquidASM (device) | 420 |

Waves never share a figure: the `Noise` column separates them.

Points that cannot run are excluded by `job_creator.py`, each with a printed
reason:
* impossible or degenerate partitions (`min_qubits_per_node: 2`);
* CUNQA `basic`/`packed` above 4 nodes (one global statevector, hours per point);
* SQUANCH `basic`/`packed` above a 10-qubit density matrix (`n + 2*nodes`;
  full-size gate operators: 10 qubits ~30 s per shot measured, 12 qubits
  extrapolate to ~20 min per shot);
* anything whose memory estimate exceeds `max_mem_gb`.

Fidelity at 100 shots is read against the ideal-sampling baseline
(`results/baselines/ideal_sampling.csv`): a perfect emulator scores
0.91-1.00 depending on `(n, theta)`, not 1.

## Procedure

### 0. Freeze the code (once)

On the development checkout:

```bash
git switch main
git merge --no-ff phase3-cunqa-v3           # brings phase2 + phase3
python3 -m pytest -q tests/                  # all pass, CUNQA tests not skipped
git tag -a campaign-v1 -m "Benchmark campaign v1"
git push origin main campaign-v1
```

### 1. Fresh clone in `$STORE` (not `$HOME`: file-count quota)

```bash
mkdir -p $STORE/dqc-campaigns
git clone --branch campaign-v1 https://github.com/gdiazcamacho/DQC-Emulator-Benchmarking.git \
    $STORE/dqc-campaigns/campaign-v1
cd $STORE/dqc-campaigns/campaign-v1
```

Environments: the conda envs (`qiskit_conda`, `squidasm_conda`,
`squanch_conda`, `interlinq_conda`) and CUNQA (`emulators/cunqa/env/`) are
the same as for development; nothing is installed in the clone.

### 2. Baseline and jobs

```bash
python3 scripts/ideal_baseline.py configs/campaign_v1_wave*.yaml
python3 scripts/job_creator.py configs/campaign_v1_wave1_noiseless.yaml
```

`job_creator.py` refuses a dirty tree, stamps the commit into every job
script, and writes `campaigns/campaign-v1/manifest_<config>.csv`.

### 3. Smoke test (one small job per emulator and mode)

```bash
python3 scripts/campaign_status.py campaign-v1 --missing | grep '_2_q4_a0.33$'
# sbatch those, wait, then:
python3 scripts/campaign_status.py campaign-v1
```

Every smoke job must produce a result, carry the campaign commit, and (being
noiseless) sit inside the ideal-sampling interval.

### 4. Submit, monitor, resubmit

```bash
python3 scripts/campaign_status.py campaign-v1 --submit --limit 200   # skips queued jobs
python3 scripts/campaign_status.py campaign-v1                         # progress + warnings
python3 scripts/campaign_status.py campaign-v1 --missing               # what is left
```

Re-running `--submit` only sends jobs that have no result and are not in the
queue, so it is safe to repeat. Do not use `scripts/job_submitter_*.sh` for a
campaign: they submit every script in `jobs/`, including stale ones.

### 5. Wave 2

```bash
python3 scripts/job_creator.py configs/campaign_v1_wave2_noisy.yaml
python3 scripts/campaign_status.py campaign-v1 --submit
```

Each wave gets its own manifest in `campaigns/campaign-v1/`; the status
report covers all of them and warns if they come from different commits.

### 6. Freeze the results

```bash
tar czf campaign-v1-results.tgz emulators/*/results campaigns/campaign-v1 results/baselines
```

Archive that file (and later the Zenodo record) with the tag.

## What every result row records

`Algorithm, Comm Mode, Noise, Emulator, Nodes, Num Qubits, Theta, Shots` (the
run coordinate, also encoded in the filename), timings, memory, `Fidelity`,
`Ebits`, `Campaign`, `Code Commit` and `Emulator Version`
(`qiskit-aer==…`, `squidasm==…`, `squanch==…`, `cunqa==3.1.0+<commit>`).
