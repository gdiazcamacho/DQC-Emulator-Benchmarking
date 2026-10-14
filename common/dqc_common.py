"""
Shared helpers for the DQC emulator benchmarking suite.

This module is imported by every emulator script (Qiskit Aer, SquidASM,
Interlin-q, SQUANCH). It is deliberately restricted to the standard library
plus NumPy so that it imports cleanly in all four conda environments,
including the Python 3.8 SquidASM environment.

Rationale: `distribute_qubits` and the results-CSV writer were previously
copy-pasted into all four scripts. The copies had already drifted (SQUANCH's
`distribute_qubits` silently dropped the low-qubit-count warning), which is
exactly the failure mode that makes a cross-emulator comparison unfalsifiable:
if the partitioning differs between emulators, the emulators are not running
the same benchmark.

Usage from an emulator script:

    import os, sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "..", "..", ".."))
    from common.dqc_common import distribute_qubits, write_result_row
"""

import csv
import os
import warnings

import numpy as np

__all__ = [
    "distribute_qubits",
    "partition_list",
    "ideal_qft_probs",
    "classical_fidelity",
    "fidelity_from_counts",
    "repo_root",
    "results_dir",
    "write_result_row",
    "result_filename",
    "code_commit",
    "package_version",
    "CSV_COLUMNS",
    "COMM_MODES",
    "NOISE_SETTINGS",
]


# --------------------------------------------------------------------------- #
# Qubit partitioning
# --------------------------------------------------------------------------- #

def distribute_qubits(num_qubits, num_nodes):
    """Distribute `num_qubits` across `num_nodes`, giving the remainder to the
    last nodes.

    Returns a list of length `num_nodes` summing to `num_qubits`, with sizes
    differing by at most one and the larger nodes first.

    A node holding <= 1 qubit performs no local controlled rotations, so the
    partition degenerates. That is legal but usually not what was intended,
    hence the warning.
    """
    if num_nodes < 1:
        raise ValueError("num_nodes must be >= 1")
    if num_qubits < num_nodes:
        raise ValueError(
            "num_qubits ({0}) < num_nodes ({1}): cannot give every node a "
            "qubit".format(num_qubits, num_nodes)
        )

    base = (num_qubits + num_nodes - 1) // num_nodes  # ceiling division
    qubits = [base] * num_nodes
    over = sum(qubits) - num_qubits

    # Remove 1 from the last 'over' nodes (starting from the end)
    for i in range(over):
        qubits[-(i + 1)] -= 1

    assert sum(qubits) == num_qubits, "partition does not conserve qubit count"

    if any(q <= 1 for q in qubits):
        warnings.warn(
            "Partition {0} has a node with <= 1 qubit; that node contributes "
            "no local controlled rotations.".format(qubits),
            RuntimeWarning,
            stacklevel=2,
        )

    return qubits


def partition_list(lst, partition_spec):
    """Split `lst` into consecutive chunks.

    `partition_spec` is either an int (number of equal-ish chunks, computed via
    `distribute_qubits`) or an explicit list of chunk sizes.
    """
    if isinstance(partition_spec, int):
        partition_spec = distribute_qubits(len(lst), partition_spec)

    if not isinstance(partition_spec, (list, tuple)):
        raise TypeError("partition_spec must be an int or a list of ints.")

    if sum(partition_spec) != len(lst):
        raise ValueError("Sum of partition sizes does not match list length.")

    result = []
    idx = 0
    for size in partition_spec:
        result.append(lst[idx: idx + size])
        idx += size
    return result


# --------------------------------------------------------------------------- #
# Ideal reference distribution and fidelity
# --------------------------------------------------------------------------- #

def ideal_qft_probs(theta, num_qubits, eps=1e-12):
    """Exact output distribution of the inverse QFT applied to the Fourier
    state parameterised by `theta`.

    Returns (x, probs) where x = k / 2**num_qubits are the phase bins.
    """
    N = 2 ** num_qubits
    ks = np.arange(N)
    x = ks / N
    angle = np.pi * (theta - x)

    probs = np.zeros(N)
    small = np.abs(angle) < eps
    not_small = ~small

    probs[small] = 1.0
    probs[not_small] = (
        np.sin(N * angle[not_small]) ** 2
        / (N ** 2 * np.sin(angle[not_small]) ** 2)
    )
    return x, probs


def classical_fidelity(p_sim, p_ideal):
    """Bhattacharyya (classical) fidelity between two discrete distributions."""
    p_sim = np.asarray(p_sim, dtype=float)
    p_ideal = np.asarray(p_ideal, dtype=float)
    if p_sim.sum() <= 0 or p_ideal.sum() <= 0:
        raise ValueError("distributions must have positive total weight")
    p_sim = p_sim / p_sim.sum()
    p_ideal = p_ideal / p_ideal.sum()
    return float(np.sum(np.sqrt(p_sim * p_ideal)) ** 2)


def fidelity_from_counts(counts, num_qubits, theta):
    """Fidelity of a measured result against the ideal inverse-QFT output.

    `counts` maps phase values in [0, 1) (as produced by every emulator script
    in this suite) to shot counts. Missing outcomes are treated as zero.

    Computed per run, at the point where the raw counts exist, rather than
    reconstructed later from a stringified dict in the CSV.
    """
    N = 2 ** num_qubits
    p_sim = np.zeros(N)
    for k, v in counts.items():
        idx = int(round(float(k) * N))
        if not (0 <= idx < N):
            raise ValueError("phase key {0!r} outside [0, 1)".format(k))
        p_sim[idx] += v

    _, p_ideal = ideal_qft_probs(theta, num_qubits)
    return classical_fidelity(p_sim, p_ideal)


# --------------------------------------------------------------------------- #
# Paths and results CSV
# --------------------------------------------------------------------------- #

def repo_root():
    """Absolute path of the repository root, derived from this file's location.

    Everything downstream is anchored here so that scripts behave identically
    whether launched from the repo root, from `emulators/<name>/scripts/`, or
    from a SLURM working directory.
    """
    return os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def results_dir(emulator):
    """Return (and create) the results directory for a given emulator."""
    path = os.path.join(repo_root(), "emulators", emulator, "results")
    os.makedirs(path, exist_ok=True)
    return path


CSV_COLUMNS = [
    "Algorithm",
    "Comm Mode",
    "Emulator",
    "Nodes",
    "Num Qubits",
    "Theta",
    "Shots",
    "Execution Time (s)",
    "Current Memory (KB)",
    "Peak Memory (KB)",
    "Noise",
    "Fidelity",
    "Ebits",
    "Packed",
    "Campaign",
    "Code Commit",
    "Emulator Version",
    "Result",
]

# Communication modes. `Packed` (bool) cannot express more than two, so
# `Comm Mode` is the discriminating column; `Packed` is retained, derived, for
# back-compatibility with existing figures.
#   basic   - one teleportation per cross-node controlled-phase gate
#   packed  - cat-entangle/disentangle packing (one teleportation per control)
#   dynamic - deferred-measurement / semiclassical iQFT (zero e-bits)
#   native  - the emulator partitions and teleports on its own; the schedule is
#             not user-controlled, so its e-bit count is NOT comparable to ours
COMM_MODES = ("basic", "packed", "dynamic", "native")

_PACKED_FROM_MODE = {"packed": True, "basic": False, "dynamic": "", "native": ""}

# Noise setting, recorded per run. A fidelity figure is meaningless without
# knowing whether the run was noiseless, and previously nothing in the CSV
# distinguished the two. Rows written before this column existed came from
# runs that ALWAYS applied the depolarizing model, so that is the back-fill
# value used by plot_results.py.
NOISE_SETTINGS = ("none", "depolarizing", "device")


def code_commit():
    """Commit that produced a run.

    job_creator.py stamps DQC_COMMIT into every job script (and refuses to
    stamp a dirty tree), so campaign runs never depend on git being callable
    from a compute node. Interactive runs fall back to asking git, marking
    uncommitted changes with '-dirty'.
    """
    env = os.environ.get("DQC_COMMIT")
    if env:
        return env
    import subprocess
    try:
        root = repo_root()
        sha = subprocess.check_output(
            ["git", "-C", root, "rev-parse", "--short=12", "HEAD"],
            stderr=subprocess.DEVNULL, text=True).strip()
        dirty = subprocess.call(
            ["git", "-C", root, "diff", "--quiet", "HEAD", "--"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) != 0
        return sha + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def package_version(*dist_names):
    """Installed version of the first distribution found, e.g.
    package_version("qiskit-aer"). Returns "unknown" if none is installed."""
    try:
        from importlib.metadata import version, PackageNotFoundError
    except ImportError:                      # Python < 3.8
        return "unknown"
    for name in dist_names:
        try:
            return "{0}=={1}".format(name, version(name))
        except PackageNotFoundError:
            continue
    return "unknown"


def result_filename(algorithm, comm_mode, noise, emulator, num_nodes,
                    num_qubits, theta, shots):
    """One file per run coordinate. Every axis that can differ between two
    runs of the same configuration -- including shots -- is in the name, so
    no run can silently overwrite another."""
    return "{0}_{1}_{2}_{3}_{4}_q{5}_a{6}_s{7}.csv".format(
        algorithm, comm_mode, noise, emulator, num_nodes, num_qubits, theta,
        shots)


def write_result_row(emulator, num_nodes, num_qubits, theta, shots,
                     execution_time, current_kb, peak_kb, result,
                     algorithm="qft", fidelity=None, ebits=None, packed=None,
                     comm_mode=None, noise=None, path=None,
                     emulator_version=None):
    """Write a single benchmark run to its own one-row CSV.

    The suite's convention is strictly one row per run: `plot_results.py`
    concatenates every CSV under `emulators/*/results/`, so any file with a
    different shape corrupts the combined DataFrame silently.

    `Algorithm` discriminates between benchmarks (qft, grover, qaoa, qpe, ...)
    so that adding a second algorithm does not contaminate the QFT figures.

    `ebits` and `packed` support the telegate-packing ablation: `ebits` is the
    number of entangling rounds (Bell pairs) consumed by the run, and `packed`
    records whether the run used the packed cat-comm construction (True) or
    the naive one-Bell-pair-per-gate baseline (False). Both are left blank
    ("") when not applicable -- most emulator scripts do not yet report this
    and should not be assumed to be one or the other by a reader of the CSV.
    """
    # `comm_mode` MUST appear in the filename. Without it a basic run and a
    # packed run of the same (algorithm, emulator, nodes, qubits, theta) write
    # to the same path and the second silently overwrites the first.
    if comm_mode is None:
        # Infer from the legacy boolean so existing callers keep working.
        comm_mode = {True: "packed", False: "basic"}.get(packed, "packed")
    if comm_mode not in COMM_MODES:
        raise ValueError(
            "comm_mode {0!r} not in {1}".format(comm_mode, COMM_MODES))
    if packed is None:
        packed = _PACKED_FROM_MODE[comm_mode]

    # No default: every script must state its error model. A default of
    # "depolarizing" previously mislabelled noiseless SQUANCH/Interlin-q rows.
    if noise is None:
        raise ValueError("noise must be given explicitly ({0})".format(
            NOISE_SETTINGS))
    if noise not in NOISE_SETTINGS:
        raise ValueError("noise {0!r} not in {1}".format(noise, NOISE_SETTINGS))

    if path is None:
        path = os.path.join(
            results_dir(emulator),
            result_filename(algorithm, comm_mode, noise, emulator, num_nodes,
                            num_qubits, theta, shots),
        )

    with open(path, mode="w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(CSV_COLUMNS)
        writer.writerow([
            algorithm,
            comm_mode,
            emulator,
            num_nodes,
            num_qubits,
            theta,
            shots,
            execution_time,
            current_kb,
            peak_kb,
            noise,
            "" if fidelity is None else fidelity,
            "" if ebits is None else ebits,
            "" if packed is None else packed,
            os.environ.get("DQC_CAMPAIGN", ""),
            code_commit(),
            emulator_version or "unknown",
            result,
        ])
    return path
