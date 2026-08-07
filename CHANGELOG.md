# Changelog

## [1.1.0] — 2026-08-07

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
