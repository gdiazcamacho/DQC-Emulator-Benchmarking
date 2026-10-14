# Changelog

## [Unreleased] - campaign-v1 preparation

Everything needed to produce the paper's results as one traceable campaign
(see docs/CAMPAIGN.md). Results produced before this point are superseded.

### Added
- **`dynamic` comm mode on Qiskit Aer, SQUANCH and SquidASM** (CUNQA already
  had it): semiclassical inverse QFT, zero e-bits. SQUANCH uses private
  per-node streams in this mode (no joint density matrix). Interlin-q cannot
  support it (it partitions the circuit itself).
  Verified: exact output at non-palindromic exact theta for 2-4 nodes on Aer
  and SQUANCH; theta = 0.33 fidelities within ideal-sampling spread (Aer
  20000 shots, SQUANCH 3000 shots at n = 7, 8, 10). SquidASM: offline mock
  harness (tests/test_squidasm_dynamic_mock.py) that runs the real program
  generators with a NumPy backend; it catches a sign error in the correction
  angle. **Still needs a live NetSquid check at an exact theta.**
- **Noise is an explicit sweep axis**: `noise:` in the config, `--noise` in
  every template and script, `SUPPORTED_NOISE` capability matrix.
  `write_result_row` no longer defaults to "depolarizing".
- **Provenance**: `Campaign`, `Code Commit`, `Emulator Version` columns.
  job_creator.py stamps DQC_COMMIT/DQC_CAMPAIGN into every job script and
  refuses a dirty tree (`--allow-dirty` for tests only).
- **Shots in the results filename** (a 100-shot run had overwritten a
  1000-shot run of the same configuration).
- **Feasibility model in job_creator.py**: per-job memory request
  (`{{MEM_GB}}`) from the size of the simulated state, `max_mem_gb`, and
  per-emulator caps (`max_nodes`, `max_state_qubits`). Every exclusion is
  printed with its reason. `--dry-run` prints the plan only.
- `configs/campaign_v1_wave1_noiseless.yaml` (840 jobs) and
  `configs/campaign_v1_wave2_noisy.yaml` (420 jobs); per-wave manifests in
  `campaigns/<campaign>/`.
- `scripts/ideal_baseline.py`: expected fidelity of an ideal N-shot sample
  per (n, theta, shots) -> `results/baselines/ideal_sampling.csv`.
- `scripts/campaign_status.py`: progress per (emulator, mode, noise),
  commit check, noiseless results below the ideal-sampling interval,
  missing jobs, queue-aware `--submit`.
- `docs/CAMPAIGN.md`: the campaign procedure.
- Tests: tests/test_campaign_tools.py (templates vs scripts vs capability
  matrix, plan consistency, baseline), tests/test_squidasm_dynamic_mock.py.

### Fixed
- **Interlin-q: every sweep job failed.** The template passed `--comm_mode`,
  which the script did not accept. It now takes `--comm_mode native` and
  `--noise none` and records both (rows were labelled `packed` and
  `depolarizing`).
- **SquidASM: NameError after every simulation.** The main block used an
  undefined `total_qubits` to decode results (introduced by the hygiene
  pass); now `num_qubits`, with a shape assertion.
- **SquidASM noise was undocumented and unlabelled.** `config_template.yaml`
  was headed "Perfect ... no noise" but sets 1%/10% gate depolarising, link
  fidelity 0.9 and finite T1/T2, and was used for every run; rows were
  labelled by the writer's default. Now `--noise none` uses
  `config-QFT-noiseless.yaml` and `--noise device` the noisy template, whose
  header now says what it is. Check which one the EuroQHPC numbers used.
- SQUANCH rows wrote `noise="none"` correctly but SQUANCH quantum modes were
  unbounded in the sweep; at 100 shots a 10-qubit density matrix takes ~45
  min (measured), 12 qubits ~30 h (extrapolated), so they are capped.

### Findings for the paper
- SQUANCH's earlier apparent scalability came from not simulating
  inter-node entanglement at all. Simulating it correctly (one dense density
  matrix, full-size gate operators) limits SQUANCH's quantum modes to
  `n + 2*nodes <= 10`; its `dynamic` mode, with no entanglement across nodes,
  runs the whole grid.
- The 100-shot fidelity ceiling depends on how close theta * 2^n is to an
  integer, not on n monotonically: 0.91-0.99 for theta = 0.33, 0.66 over
  n = 4..20, exactly 1 for theta = 0.

## [Unreleased] - CUNQA integration against 3.0.0

Rewrites `emulators/cunqa/scripts/qft_multinode.py` against the CUNQA 3.0.0
source (python package and C++ executors) and the QCE26 tutorial. The previous
adapter had never run; checked on the new shadow executor, it could not have.

### Fixed (previous adapter)
- **Relayed bits never reached the counts.** CUNQA reports only clbits written
  by a `measure(save=True)`; a `recv` under quantum_comm forces save=False and
  under classical_comm never sets it. Node 0's counts therefore held only its
  own qubits and the decoder indexed past the end of the key. Node 0 now
  re-measures each relayed bit through a freed data qubit
  (`cif(b) x(r); measure(r, b); cif(b) x(r)`).
- **Classical-bit overflow at 8 nodes.** `cat_disentangler` receives one bit per
  receiver into distinct clbits; the fixed 2-bit scratch region overflowed the
  register at every 8-node point. Scratch width is now `num_nodes`.
- **Default backends rejected the circuits.** quantum_comm vQPUs default to
  (2 data, 1 comm) qubits and classical_comm to (6, 0); `run` raises for larger
  circuits. A per-run backend JSON with `num_qubits: [max node size, comm]` is
  now generated and passed to `qraise`.
- **No memory request.** quantum_comm simulates one centralized state of
  N + M qubits; `mem_per_qpu` is now sized from that (overridable).
- CSV rows were labelled `Noise=depolarizing` (the writer's default); now `none`.
- `Ebits` column now filled.

### Added
- `--comm_mode basic` (one two-party telegate per cross-node gate) and
  `--comm_mode dynamic` (semiclassical iQFT over classical_comm, zero e-bits:
  the tutorial's IPEA pattern applied to the benchmark's iQFT). Capability
  matrix: cunqa = {basic, packed, dynamic}.
- `emulators/cunqa/scripts/cunqa_shadow.py`: NumPy executor for the real
  `CunqaCircuit` instruction streams with CUNQA 3.0.0 semantics (per-shot
  lockstep, GHZ `gen_ent` with |0> assertion, save flags, counts key format,
  one-clbit classical messages). `qft_multinode.py --shadow` uses it.
- `estimate_seconds` / `default_max_time` size the qraise budget; driver
  template walltime raised to 2 days to cover it.
- Run-time check that each node's own counts equal the marginal of the relayed
  joint distribution (catches broken shot pairing).
- `tests/test_cunqa_adapter.py` (27 tests; circuit tests skip without the
  cunqa Python package).

### Verified (shadow executor, real cunqa 3.0.0 Python package)
- Exact output for a non-palindromic basis theta at (n,M) = (4,1) (4,2) (6,2)
  (6,3) (8,2) (8,4) (9,3) (10,2) (10,5) (8,8), all three modes.
- theta = 0.33, 0.66 at 2000 shots, (7,2) [4,3] and (10,4) [3,3,2,2]:
  fidelity within 2 sigma of an ideal multinomial sample of the same size.
- E-bits: gen_ent audit == closed form == CHANGELOG formulas
  (packed m*C(M,2), basic n^2(M-1)/(2M)).
- Node marginals equal joint marginals exactly in every run.
- No clbit index out of range anywhere in the sweep grid.

### Environment (pinned build, same on every cluster)
- CUNQA pinned to tag 3.1.0 (commit 29130f6) and built from source by
  `emulators/cunqa/env/install_cunqa.sh` into `$CUNQA_PREFIX`
  (default `$STORE/opt/cunqa-3.1.0`; source/build under `$STORE`,
  never `$HOME`, whose quota the build tree exceeds), recording `BUILD_INFO`.
  `pytest` is pip-installed with `--target` next to it. Site modules are
  not used: QMIO offered 2.4.0, whose telegate API (`expose`/`rcontrol`
  context managers, implicit comm qubits, no `save` flag) is incompatible.
- `emulators/cunqa/env/cunqa_env.sh` is the single environment for build,
  tests and jobs (modules mirrored from CUNQA's configure.sh for QMIO/FT3,
  `CUNQA_SITE_MODULES` elsewhere, `STORE`, PATH/PYTHONPATH). The SLURM
  template sources it instead of `module load cunqa`.
- Workaround for a CUNQA 3.1.0 bug (fixed upstream after the tag, PR #641):
  Python `qraise` initialises a missing `$STORE/.cunqa/qpus.json` as the JSON
  string `"{}"`, after which every `qraise` fails with json type_error.305.
  `cunqa_env.sh` now (re)initialises it to `{}` when it is not a JSON object.
- **Live-run fix, dynamic mode:** classical corrections used `p`, which CUNQA
  3.1.0's Aer adapter does not implement (it has `u1`, the same gate). The
  vQPU threw and its peer blocked on `recv` forever. Now `u1`; the shadow
  executor rejects any gate outside the Aer adapter's set so this is caught
  offline.
- `gather` runs under a watchdog (`--run_timeout`, default = qraise budget),
  so a dead vQPU produces an error and a `qdrop` instead of a hang.
- Live runs refuse to start unless `cunqa_env.sh` was sourced (Python >= 3.9,
  `CUNQA_PREFIX` set, and `cunqa` actually imported from it) -- a stale
  `~/cunqa` from an older release was being picked up via the `$HOME` fallback.
- **`--seed` refused for live runs.** CUNQA 3.1.0 re-seeds Aer at the start
  of every shot (`AERSimulatorAdapter::initialize`), so with a seed every shot
  of a dynamic circuit is identical. On QMIO at theta = 0.33 this gave
  fidelity 0.768 for packed and basic (exactly p_ideal of the modal outcome)
  and 0.003 for dynamic (each classical-comm vQPU replaying its own fixed
  stream). Worth reporting upstream.
- Live check on QMIO (4 qubits / 2 nodes, theta = 3/16): packed, basic and
  dynamic give fidelity 1.000000 with zero marginal mismatch.
- Live check on QMIO, theta = 0.33, 1000 shots: packed 0.9984, basic 0.9943,
  dynamic 0.9931, noiseless Aer packed 0.9955 -- all within the spread of
  ideal 1000-shot sampling (0.9953 +/- 0.0022).
- Nested submission verified on QMIO: the SLURM driver job's qraise submits
  the vQPU allocation from inside the job (job 561229, 100 shots, fidelity
  0.9559; ideal 100-shot sampling gives 0.9505 +/- 0.0125).
- The adapter and tests locate CUNQA via `CUNQA_PREFIX` (fallback `$HOME`) and
  put `$CUNQA_PREFIX/bin` on PATH for `qraise`.

### Fixed (repository)
- `.gitignore` still contained `test_*.py`, so `tests/` was untracked despite
  the phase-1 entry saying otherwise. Removed; `emulators/cunqa/backends/`
  ignored instead.

### Not yet done
- No live run. First cluster step: 4 qubits / 2 nodes, each mode, against
  noiseless Qiskit Aer.
- Nested `qraise` from inside a SLURM job is unverified on QMIO/FT3.
- The shadow models CUNQA's documented/observed semantics, not its timing; a
  live run can still disagree (e.g. FIFO order of classical messages).

## [Unreleased] - communication-mode axis

### Added
- `Comm Mode` and `Noise` as first-class CSV columns and as components of the
  results filename, so runs differing only in communication schedule or error
  model can no longer overwrite one another.
- Capability matrix in `scripts/job_creator.py`
  (`SUPPORTED_COMM_MODES` / `SUPPORTED_ALGORITHMS`): the sweep is a full
  cartesian product, and combinations no emulator implements are now excluded
  with a printed reason instead of generating jobs that cannot run.
- `--comm_mode {basic,packed}` on the Qiskit Aer script.
- `--noise {depolarizing,none}` on the Qiskit Aer script. The depolarizing
  model was previously applied unconditionally, so there was no way to check
  circuit correctness independently of the error model. Default is unchanged.
- Named `{{PLACEHOLDER}}` substitution in the SLURM templates, and a per-job
  `--job-name` (it was a constant, so `%x_%j` log files were not traceable to
  a configuration).

### Changed
- The communication schedule is no longer encoded in `Algorithm`
  (`qft_telegate_baseline`). It is its own axis; folding it into `Algorithm`
  multiplies the wrong dimension as soon as a second algorithm is added.
  `--telegate_baseline` is kept as a deprecated alias for `--comm_mode=basic`.
- `plots/plot_results.py` filters on `Algorithm`, `Comm Mode` and `Noise`, and
  warns when several are loaded at once. Without this, `fidelity_*` helpers
  select an arbitrary row and `time_memory_vs_qubits` reports the spread
  between modes as if it were run-to-run variance. Older CSVs are back-filled
  (`qft`, `packed` or the legacy `Packed` boolean, `depolarizing`).

### Fixed
- **SQUANCH: no quantum channel existed between nodes.** The runner allocated
  one `QStream` per node. SQUANCH's quantum channel transmits only a
  reference (`Qubit.serialize()` returns `(system_index, qubit_index)`) and
  the receiver resolves it against its OWN stream, so with per-node streams
  those indices named a different, untouched qubit: every Bell pair sent
  across a node boundary arrived unentangled and every cat-entangle
  teleported a |0>. Measured Bell correlation 0.500 (chance) versus 1.000 for
  the core simulator. Fixed with a single shared `QStream` holding every data
  qubit plus two communication ancillas per node.
  Reproduce with `emulators/squanch/scripts/diagnose_squanch.py`.
- **SQUANCH: communication ancilla never reset after measurement.** In
  `_cat_disentangle_target` the token was measured but left in |1> half the
  time, while the source correctly reset `b0`. The ancilla is reused per
  control qubit, so from the second teleportation onwards `H; CNOT` prepared
  |Phi-> instead of |Phi+>. Present independently of the channel defect, which
  masked it entirely.
- **SQUANCH: concurrent gate application on a shared qsystem.** `QSystem.apply`
  does `state[...] = U @ state @ U.conj().T`, a non-atomic read-modify-write
  of the whole density matrix; two agents corrupt each other even on disjoint
  qubits. Blocks of gate applications are now guarded by a
  `multiprocessing.Lock`, never held across a send/recv. Note this means the
  agents no longer run in parallel in any meaningful sense; they take turns on
  a shared state.

### Added (continued)
- `--comm_mode {basic,packed}` on SQUANCH, verified at fidelity 1.0 for both
  modes at (n,k) = (4,2), theta = 0.0 and 0.5, with e-bits 2 (packed) and
  4 (basic).
- `--comm_mode` accepted (packed only) on SquidASM and CUNQA so the scaffolding
  can pass the axis uniformly.
- `tests/test_results_io.py`: 10 tests pinning the results-CSV contract, in
  particular that runs differing only in communication schedule or error model
  land on distinct paths. That collision is silent -- the second run
  overwrites the first and the lost series looks like a job that never ran.

### Known cost change
- The corrected SQUANCH simulates the JOINT state: a dense density matrix of
  side 2^(n_total + 2P). It will not reach the qubit counts previously
  reported. Part of what made it appear to scale was that it was not modelling
  inter-node entanglement at all. Figures must be regenerated and the reduced
  range stated explicitly.

### Verified
- Noiseless correctness, Qiskit Aer, theta = 0: fidelity 1.0 at
  (n,k) = (4,1), (4,2), (6,2), (6,3), (8,2), both comm modes.
- E-bit counts match theory: packed = m*C(M,2), basic = n^2(M-1)/(2M),
  ratio n/M. Measured at k=2: 2/4, 3/9, 4/16 for n = 4, 6, 8.


## [Unreleased] — Phase 2, telegate packing ablation

Adds a config flag rather than new machinery, as planned: `--telegate_baseline`
on `emulators/qiskit_aer/scripts/qft_multinode.py` switches
`recursive_qft_telegate` from the packed cat-comm construction (one Bell pair
shared across a whole column of `cp` gates) to the naive baseline (one fresh
Bell pair per individual gate, no packing).

- `recursive_qft_telegate` now returns `(circuit, ebits_used)`; `run_DQFT`
  and the CLI thread both through.
- `common/dqc_common.write_result_row` gains optional `ebits` and `packed`
  columns (blank for scripts that don't report them, so the schema stays
  backwards-compatible without another backfill rule).
- Baseline runs write under `Algorithm=qft_telegate_baseline` rather than
  `qft`, so `plot_results.py --algorithm` cleanly separates the ablation from
  the main scaling figures without touching them.
- **Verified, not just implemented**: ran both variants at n = 4, 6, 8, 10
  qubits over 2 nodes and confirmed e-bit counts against hand-derived
  predictions before trusting the figure. Exact fit: packed = n/2,
  baseline = (n/2)², ratio 1.000 at every point (bipartite case only --
  the general multi-node scaling constant still needs the same treatment
  before being stated in the paper). Noiseless correctness at n=4/n=6 also
  checked: both variants agree with each other and with the analytic
  distribution up to shot noise.
- Figure at `plots/ebit_packing_ablation.png` / `.svg` — ready to drop into
  the iQFT results section as the nested ablation subsection per the project
  plan (one figure, not a co-equal contribution).

To reproduce or extend to more node counts / qubit counts:
```bash
python -c "
import sys; sys.path.insert(0, 'emulators/qiskit_aer/scripts')
import qft_multinode as m
phases, ebits = m.run_DQFT(num_qft_qubits=8, num_nodes=2, theta=0.33,
                           shots=1000, verbose=False, packing=False)
print(ebits)
"
```

### Not yet done

- Only the bipartite (2-node) case is verified. The multi-node packing
  constant (more than 2 nodes) should get the same hand-check before being
  claimed in the paper -- the packing loop structure differs when there is
  more than one target node per control.
- `--telegate_baseline` exists only on the Qiskit Aer script. Whether to add
  it to SquidASM's `AbstractNode`-based construction too depends on whether
  the paper wants the ablation cross-checked on a second backend or is
  content with one.

---



Adds `emulators/cunqa/` — a distributed inverse-QFT implementation on CUNQA
(github.com/CESGA-Quantum-Spain/cunqa), CESGA's own DQC emulator. Not yet added
to `configs/sweep_params.yaml`'s default emulator list; see the module
docstring in `emulators/cunqa/scripts/qft_multinode.py` for the validation
steps required before it is.

### Findings from reading the CUNQA source (not just its docs)

- **CUNQA's telegate primitive is named `cat_entangler`/`cat_disentangler`** and
  implements the same construction as this suite's `cat_entangle`/
  `cat_disentangle`: share a control onto every named remote circuit's comm
  qubit in one entangling round, apply the local controlled gate, then
  disentangle. This is a third independent implementation of the same
  primitive (alongside this suite and Vernec et al.), and CUNQA's multi-target
  `cat_entangler` gives packed-column behaviour natively.

- **Result correlation across nodes is not exposed by the public API as a
  joint distribution.** `Result.counts` (`cunqa/result.py`) is a per-circuit
  histogram. Reading the C++ executor
  (`src/sim/backends/quantum_comm/qc_executor.cpp`) confirms shots *are*
  executed in a correlated lockstep across all nodes internally, but that
  pairing is not surfaced back to Python. Fixed here by classically relaying
  every non-zero node's measured bits to node 0 with `send`/`recv` (the same
  primitives `cat_entangler` itself uses), so `results[0].counts` is directly
  the joint distribution. Classical-only, no extra e-bit cost, but is a
  benchmark-specific addition on top of the algorithm's own communication and
  should be described as such if reported in the paper.

- **Noise is not yet wired up for the quantum-communication path.** Confirmed
  by inspecting the backend JSON schema and the example backend files
  directly, not just the docs page's statement: `backend=` files passed to
  quantum-comm examples in CUNQA's own repository only declare qubit topology,
  and the `noise_model` key only appears in the no-communication example.
  CUNQA is therefore noiseless-only in this suite until that changes upstream.

- **No in-process memory hook on the vQPU.** `Result` carries `counts` and
  `time_taken` only. The vQPU runs as a SLURM-managed process separate from
  the driver script, so `tracemalloc` (used by the other four scripts) cannot
  see it. Current implementation reports `resource.getrusage` on the driver
  process only — a real methodological difference from the other four, to be
  disclosed rather than presented as directly comparable.

- **CUNQA's own distributed-QPE example** (`examples/qu_comm/04-qdistributed_QPE.py`)
  uses telegate-based control distribution with a single register qubit
  (Kitaev/IPEA style) — different from the target-shuttle-plus-semiclassical
  design already settled for the (separate, out-of-scope-for-this-paper) QPE
  work, so no prior-art conflict, but worth a differentiating footnote later.

### Execution model

`qraise` (allocate vQPUs) → `run` → `qdrop` (release) once per benchmark
point, matching the other four scripts' one-SLURM-job-per-run model rather
than CUNQA's normal amortized multi-circuit usage. The SLURM template requests
minimal resources for the driver process itself; `qraise` submits its own
allocation for the vQPUs. **Nested SLURM submission from within a running job
has not been validated against CESGA's scheduler configuration** — if it
fails, run the CUNQA jobs interactively from a login node instead.

### Not yet done

- On-cluster validation (module docstring, steps 1–3): hand-check the 2-qubit/
  2-node case, cross-check against noiseless Qiskit Aer, only then add to the
  sweep.
- Bit ordering in the final joint bitstring is implemented but **unverified**
  — flagged explicitly in the script rather than assumed, since every other
  emulator in this suite needed its own bit-order fix at some point.
- `emulators/cunqa/env/` has no conda spec (CUNQA is built from source); a
  short README points at CESGA's install docs instead.

---



Repository hygiene and correctness pass ahead of journal submission. **No
published result changes**; see "Verification" below.

### Fixed

- **`crz` → `cp` in the Qiskit Aer distributed ladder.** `recursive_qft` and
  `recursive_qft_telegate` applied `crz` for inter-node controlled rotations
  while the local `myqft_inverse` used `cp`. The inverse QFT is built from
  controlled-*phase* gates, so `cp` is correct at both sites.

  The two differ by a control-local phase, `crz(t) = cp(t) · P_control(−t/2)`.
  Because every control qubit is measured in the computational basis
  immediately after its packed column, that phase is diagonal and unobservable:
  measured distributions are identical (verified at 2×10⁵ shots, max
  |p_crz − p_cp| = 0). The discrepancy becomes physical the moment a control
  stays coherent — applying a single Hadamard afterwards gives
  |⟨ψ_cp|ψ_crz⟩|² = 0.691 — which is the regime any QPE or subroutine reuse
  would enter.

- **`AbstractNode.CRz` renamed to `CP`.** The function was misnamed: its gate
  sequence implements controlled-phase, not controlled-Rz. The leading
  `control.rot_Z(angle/2)` is exactly the correction taking CRz to CP.
  Confirmed two further ways: SQUANCH uses `CPHASE` for the same role, and the
  `if angle < 0: angle += 2π` branch is only valid for CP (2π-periodic) — the
  same wraparound on a true CRz (4π-periodic) would flip its sign. `CRz` is
  retained as a backwards-compatible alias. SquidASM results were always
  correct; only the name was wrong.

- **Infeasible sweep points no longer generate jobs.** `distribute_qubits`
  accepted `num_qubits < num_nodes` and returned partitions containing
  zero-qubit nodes (4 qubits over 8 nodes gave `[1,1,1,1,0,0,0,0]`). These are
  now rejected. Degenerate partitions where some node holds a single qubit —
  and therefore performs no local controlled rotations — are excluded by the
  new `min_qubits_per_node` config key (default `2`; set `1` for the original
  grid). `job_creator.py` reports every exclusion with its reason.

- **Variable shadowing.** `target_nodes = [n for n in ...]` rebound `n`, the
  qubit-count parameter, in both Qiskit ladder functions. `num_qubits` was
  likewise rebound from `combined.shape[1]` in the SquidASM main block.

- **`.gitignore` excluded the test suite.** `test_*.py` was in the ignore list,
  which would have silently kept `tests/` out of the repository — the same
  class of error as the earlier blanket `*.sh` / `*.yaml` rules.

- **A fresh clone could not run.** `emulators/*/results/` and
  `emulators/*/jobs/` did not exist, and scripts wrote to a hardcoded
  `../results/`. Directories are now tracked via `.gitkeep`, and all paths
  derive from the repository root.

- **Empty `LICENSE`** (referenced by the README) and an incomplete
  `CITATION.cff` missing `license`, `date-released`, `repository-code`, and
  `type`.

### Changed

- **New `common/dqc_common.py`**, imported by all four emulator scripts.
  Consolidates `distribute_qubits`, `partition_list`, the analytic inverse-QFT
  reference, fidelity computation, and the results-CSV writer. The four copies
  had already drifted — SQUANCH's `distribute_qubits` had lost the low-qubit
  warning — which undermines the claim that all four emulators run the same
  benchmark. Restricted to the standard library plus NumPy so it imports in the
  Python 3.8 SquidASM environment.

- **CSV schema**: added `Algorithm` (discriminator, so a second benchmark
  cannot contaminate the QFT figures) and `Fidelity` (computed per run where
  the raw counts exist, instead of reconstructed later from the stringified
  histogram). Filenames are now prefixed by algorithm. `parse_csv_results`
  backfills legacy files: absent `Algorithm` → `qft`, absent `Fidelity` →
  recomputed via the same shared function.

- **`plots/plot_results.py`** restructured: module-level driver code moved into
  `main()` behind an `if __name__ == "__main__"` guard, with `--base-dir`,
  `--algorithm`, and `--save-path` flags. Paths are repo-anchored, duplicate
  imports removed, helpers superseded by `common/` deleted, and the fidelity
  figure now saves rather than only calling `plt.show()`.

- **Path handling**: every script resolves paths from its own location, so runs
  from the repository root, from the script's directory, or from a SLURM
  working directory all behave identically. The commented-out `os.chdir` lines
  are gone.

- Removed the unused `qiskit_ibm_runtime` import, which pulled in an entire
  cloud SDK for nothing, and now-unused `csv` imports.

### Added

- **`tests/test_primitives.py`** — 11 tests requiring only NumPy, so they run
  from a fresh clone with no emulator installation and no NetSquid licence.
  Covers the CP/CRz identities and the CP-only validity of the 2π wraparound,
  partition conservation and balance, the analytic reference distribution, and
  an end-to-end check that the distributed circuit reproduces it (skips cleanly
  without Qiskit).

- `min_qubits_per_node` in `configs/sweep_params.yaml`.

- `results/landscapes/` ignore rule, for auxiliary outputs that are not
  one-row-per-run.

### Verification

| Check | Result |
|---|---|
| `tests/test_primitives.py` | 11/11 pass |
| Noiseless `recursive_qft`, 8 qubits / 2 nodes, fidelity vs analytic | 0.999827 |
| Noiseless `recursive_qft_telegate`, same parameters | 0.999815 |
| `crz` vs `cp` measured distributions, 2×10⁵ shots | max deviation 0 |
| Sweep grid at default `min_qubits_per_node: 2` | 336 of 432 points |
| Sweep grid at `min_qubits_per_node: 1` | 408 points (24 impossible always excluded) |
| Legacy-format CSV parsing | backfilled correctly alongside current-format rows |
| End-to-end plotting, headless | 4 figures produced |

### Open

- **Confirm the MIT licence** is what CESGA expects given the MICINN /
  NextGenerationEU funding. One-file swap if not.
- Add remaining paper co-authors to `CITATION.cff` and mint a Zenodo DOI
  (`TODO` markers are in place).
- Decide whether the degenerate `min_qubits_per_node: 1` points appear in the
  published figures; if so, either state the exclusion in the paper or set the
  key to `1`.
- Merge `grover_multinode.py` and `qaoa_multinode.py` into this repository. The
  `Algorithm` column and the `landscapes/` rule are in place for them.
