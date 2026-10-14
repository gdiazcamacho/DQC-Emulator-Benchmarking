"""
Generate one SLURM script per benchmark point, plus a campaign manifest.

    python scripts/job_creator.py configs/campaign_v1_wave1_noiseless.yaml
    python scripts/job_creator.py <config> --dry-run        # print the plan only
    python scripts/job_creator.py <config> --allow-dirty    # NOT for campaigns

A point is (emulator, algorithm, comm_mode, noise, num_qubits, num_nodes,
theta). The grid in the config is a full cartesian product; this script keeps
only points that (1) the emulator implements, (2) form a valid partition and
(3) fit the emulator's memory model and the configured caps. Every exclusion is
printed with its reason, so the sweep stays auditable and the exclusions are
quotable in the paper.

Each generated script carries the code commit and campaign name
(DQC_COMMIT / DQC_CAMPAIGN), which every emulator script writes into its
results row. Campaign configs refuse a dirty working tree: a result must be
traceable to one commit.

The manifest (campaigns/<campaign>/manifest_<config>.csv, one per config so
waves do not overwrite each other) lists every emitted job and the results
file it must produce; scripts/campaign_status.py checks all of them.
"""
import argparse
import csv
import math
import os
import subprocess
import sys
import warnings

import yaml

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, REPO_ROOT)
from common.dqc_common import distribute_qubits, result_filename  # noqa: E402


# --------------------------------------------------------------------------- #
# Capability matrix
# --------------------------------------------------------------------------- #
# THIS REFLECTS WHAT THE SCRIPTS SUPPORT TODAY. Widen an entry only once the
# corresponding script accepts that value.
#
#   basic   one telegate per cross-node controlled phase
#   packed  one telegate per control qubit per lower node (cat-entangle)
#   dynamic semiclassical iQFT: measure + classically conditioned phases,
#           zero e-bits (valid only for a terminal transform)
#   native  emulator partitions/teleports by itself (not user-controlled)
SUPPORTED_COMM_MODES = {
    "qiskit_aer":  {"basic", "packed", "dynamic"},
    # basic is not implemented on SquidASM's AbstractNode construction.
    "squidasm":    {"packed", "dynamic"},
    "squanch":     {"basic", "packed", "dynamic"},
    "cunqa":       {"basic", "packed", "dynamic"},
    "interlin-q":  {"native"},
    "simdisq-net": {"native"},
}

# Error models each script can apply. "none" everywhere is what makes the
# noiseless wave a like-for-like comparison.
#   depolarizing  Aer's historical model: 1% on x/h, 2% on cx
#   device        SquidASM config_template.yaml: 1%/10% gate depolarising,
#                 link fidelity 0.9, finite T1/T2
SUPPORTED_NOISE = {
    "qiskit_aer":  {"none", "depolarizing"},
    "squidasm":    {"none", "device"},
    "squanch":     {"none"},
    "cunqa":       {"none"},        # no noise on communication paths upstream
    "interlin-q":  {"none"},
    "simdisq-net": {"none"},
}

SUPPORTED_ALGORITHMS = {e: {"qft"} for e in SUPPORTED_COMM_MODES}

# The deferred-measurement construction is only valid when the transform is
# TERMINAL (immediately followed by computational-basis measurement).
DYNAMIC_ONLY_FOR = {"qft"}

QUANTUM_MODES = {"basic", "packed", "native"}


def is_supported(emulator, algorithm, comm_mode, noise="none"):
    """Return (supported, reason). Reason is empty when supported."""
    if emulator not in SUPPORTED_COMM_MODES:
        return False, "unknown emulator {0!r}".format(emulator)
    if algorithm not in SUPPORTED_ALGORITHMS.get(emulator, set()):
        return False, "{0} has no {1!r} implementation".format(emulator, algorithm)
    if comm_mode not in SUPPORTED_COMM_MODES[emulator]:
        return False, "{0} does not support comm_mode={1!r} (supports {2})".format(
            emulator, comm_mode, sorted(SUPPORTED_COMM_MODES[emulator]))
    if comm_mode == "dynamic" and algorithm not in DYNAMIC_ONLY_FOR:
        return False, "comm_mode='dynamic' needs a terminal transform; {0!r} is not".format(algorithm)
    if noise not in SUPPORTED_NOISE[emulator]:
        return False, "{0} has no {1!r} error model (supports {2})".format(
            emulator, noise, sorted(SUPPORTED_NOISE[emulator]))
    return True, ""


def is_feasible(num_qubits, num_nodes, min_qubits_per_node=2):
    """Whether a (qubits, nodes) point is a valid benchmark partition.

    Impossible points (fewer qubits than nodes) are always rejected.
    Degenerate points (a node with fewer than `min_qubits_per_node` qubits,
    which performs no local controlled rotations) are rejected according to
    the config key of the same name.
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            partition = distribute_qubits(num_qubits, num_nodes)
    except ValueError as exc:
        return False, str(exc)
    if min(partition) < min_qubits_per_node:
        return False, (
            "partition {0} has a node below min_qubits_per_node={1}".format(
                partition, min_qubits_per_node))
    return True, ""


# --------------------------------------------------------------------------- #
# Memory model
# --------------------------------------------------------------------------- #
def _sv_gb(qubits):
    """Bytes of a complex128 statevector over `qubits`, in GB."""
    return 16.0 * 2 ** qubits / 1e9


def state_qubits(emulator, comm_mode, num_qubits, num_nodes):
    """Size (in qubits) of the largest state object the emulator holds.

    For SQUANCH's quantum modes this is the side of ONE dense density matrix
    over every data qubit plus two ancillas per node, and every gate is applied
    as a full-size operator, so run time explodes long before memory does
    (measured: n=4/2 nodes ~0.8 s/shot, n=6/2 nodes ~27 s/shot). Caps in the
    campaign config are expressed in these units (`max_state_qubits`)."""
    sizes = distribute_qubits(num_qubits, num_nodes)
    if comm_mode == "dynamic":
        return num_qubits if emulator == "qiskit_aer" else max(sizes)
    if emulator == "squanch":
        return num_qubits + 2 * num_nodes
    return num_qubits + num_nodes      # data qubits + one comm qubit per node


def estimated_mem_gb(emulator, comm_mode, num_qubits, num_nodes, shots=1):
    """Memory to request for one job (GB), from the state the emulator holds.

    A deliberately simple model (x3 working copies + 1 GB interpreter) whose
    job is to (a) avoid out-of-memory kills at the large end, (b) avoid
    reserving 16 GB for 4-qubit runs, and (c) flag points that cannot run.
    """
    sizes = distribute_qubits(num_qubits, num_nodes)
    n, M = num_qubits, num_nodes
    if emulator == "cunqa":
        return 2          # driver only; qraise sizes the vQPU allocation itself
    if emulator == "squanch":
        # A QStream preallocates one density matrix PER SHOT.
        if comm_mode == "dynamic":       # private per-node density matrices
            state = shots * sum(16.0 * 4 ** m / 1e9 for m in sizes)
        else:                            # one dense density matrix, n + 2M qubits
            state = shots * 16.0 * 4 ** (n + 2 * M) / 1e9
    elif comm_mode == "dynamic":
        state = _sv_gb(n) if emulator == "qiskit_aer" else sum(_sv_gb(m) for m in sizes)
    else:                                # data qubits + one comm qubit per node
        state = _sv_gb(n + M)
    return max(2, int(math.ceil(3 * state + 1)))


# --------------------------------------------------------------------------- #
# Provenance
# --------------------------------------------------------------------------- #
def git_commit(allow_dirty):
    try:
        sha = subprocess.check_output(
            ["git", "-C", REPO_ROOT, "rev-parse", "--short=12", "HEAD"],
            text=True).strip()
        dirty = subprocess.check_output(
            ["git", "-C", REPO_ROOT, "status", "--porcelain", "--untracked-files=no"],
            text=True).strip()
    except Exception as exc:
        if allow_dirty:
            return "unknown"
        sys.exit("Cannot read the git commit ({0}); campaigns must run from a "
                 "git checkout.".format(exc))
    if dirty:
        if not allow_dirty:
            sys.exit("Working tree has uncommitted changes:\n{0}\nCommit them "
                     "(campaign results must map to one commit) or pass "
                     "--allow-dirty for a test run.".format(dirty))
        return sha + "-dirty"
    return sha


# --------------------------------------------------------------------------- #
# Generation
# --------------------------------------------------------------------------- #
MANIFEST_COLUMNS = ["job_name", "emulator", "algorithm", "comm_mode", "noise",
                    "num_qubits", "num_nodes", "theta", "shots", "mem_gb",
                    "job_script", "result_file"]


def plan(config):
    """Return (jobs, excluded_combos, skipped_points) without writing files."""
    emulators = config["emulators"]
    algorithms = config.get("algorithms", ["qft"])
    comm_modes = config.get("comm_modes", ["packed"])
    noises = config.get("noise", ["none"])
    shots = config["shots"]
    min_q = config.get("min_qubits_per_node", 2)
    caps = config.get("caps", {}) or {}
    max_mem = config.get("max_mem_gb", 240)

    jobs, excluded, skipped = [], [], []
    for emulator in emulators:
        for algorithm in algorithms:
            for comm_mode in comm_modes:
                for noise in noises:
                    ok, why = is_supported(emulator, algorithm, comm_mode, noise)
                    if not ok:
                        excluded.append(("{0}/{1}/{2}/{3}".format(
                            emulator, algorithm, comm_mode, noise), why))
                        continue
                    cap = (caps.get(emulator) or {}).get(comm_mode) or {}
                    for n in config["num_qubits_range"]:
                        for M in config["num_nodes_range"]:
                            ok, why = is_feasible(n, M, min_q)
                            if not ok:
                                skipped.append((emulator, comm_mode, noise, n, M, why))
                                continue
                            if "max_nodes" in cap and M > cap["max_nodes"]:
                                skipped.append((emulator, comm_mode, noise, n, M,
                                                "cap: max_nodes={0}".format(cap["max_nodes"])))
                                continue
                            if "max_state_qubits" in cap:
                                sq = state_qubits(emulator, comm_mode, n, M)
                                if sq > cap["max_state_qubits"]:
                                    skipped.append((emulator, comm_mode, noise, n, M,
                                                    "cap: state of {0} qubits > max_state_qubits={1}".format(
                                                        sq, cap["max_state_qubits"])))
                                    continue
                            if "max_qubits" in cap and n > cap["max_qubits"]:
                                skipped.append((emulator, comm_mode, noise, n, M,
                                                "cap: max_qubits={0}".format(cap["max_qubits"])))
                                continue
                            mem = estimated_mem_gb(emulator, comm_mode, n, M, shots)
                            if mem > max_mem:
                                skipped.append((emulator, comm_mode, noise, n, M,
                                                "needs ~{0} GB > max_mem_gb={1}".format(mem, max_mem)))
                                continue
                            for theta in config["theta_range"]:
                                name = "{0}_{1}_{2}_{3}_{4}_q{5}_a{6}".format(
                                    emulator, algorithm, comm_mode, noise, M, n, theta)
                                jobs.append(dict(
                                    job_name=name, emulator=emulator,
                                    algorithm=algorithm, comm_mode=comm_mode,
                                    noise=noise, num_qubits=n, num_nodes=M,
                                    theta=theta, shots=shots, mem_gb=mem,
                                    job_script=os.path.join(
                                        "emulators", emulator, "jobs", name + ".sh"),
                                    result_file=os.path.join(
                                        "emulators", emulator, "results",
                                        result_filename(algorithm, comm_mode, noise,
                                                        emulator, M, n, theta, shots)),
                                ))
    return jobs, excluded, skipped


def write_jobs(config, jobs, commit, campaign, stem="sweep"):
    for job in jobs:
        tpl = os.path.join(REPO_ROOT, "emulators", job["emulator"],
                           "slurm_template_multinode.sh")
        with open(tpl) as fh:
            content = fh.read()
        subs = {
            "{{JOB_NAME}}": job["job_name"], "{{ALGORITHM}}": job["algorithm"],
            "{{COMM_MODE}}": job["comm_mode"], "{{NOISE}}": job["noise"],
            "{{NUM_QUBITS}}": str(job["num_qubits"]),
            "{{NUM_NODES}}": str(job["num_nodes"]), "{{THETA}}": str(job["theta"]),
            "{{SHOTS}}": str(job["shots"]), "{{MEM_GB}}": str(job["mem_gb"]),
            "{{COMMIT}}": commit, "{{CAMPAIGN}}": campaign,
        }
        for k, v in subs.items():
            content = content.replace(k, v)
        if "{{" in content:
            raise ValueError("Unresolved placeholder in {0}; check {1}".format(
                job["job_script"], tpl))
        path = os.path.join(REPO_ROOT, job["job_script"])
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as fh:
            fh.write(content)

    mdir = os.path.join(REPO_ROOT, "campaigns", campaign)
    os.makedirs(mdir, exist_ok=True)
    mpath = os.path.join(mdir, "manifest_{0}.csv".format(stem))
    with open(mpath, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=MANIFEST_COLUMNS)
        w.writeheader()
        for job in jobs:
            w.writerow(job)
    with open(os.path.join(mdir, "config_{0}.yaml".format(stem)), "w") as fh:
        yaml.safe_dump(dict(config, commit=commit), fh, sort_keys=False)
    return mpath


def summarise(jobs, excluded, skipped):
    from collections import Counter
    by = Counter((j["emulator"], j["comm_mode"], j["noise"]) for j in jobs)
    print("Jobs per (emulator, comm_mode, noise):")
    for k in sorted(by):
        print("  {0:<12} {1:<8} {2:<13} {3:>4}".format(*k, by[k]))
    print("Total jobs: {0}".format(len(jobs)))
    if excluded:
        print("\nExcluded {0} combination(s):".format(len(excluded)))
        for combo, why in excluded:
            print("  - {0}: {1}".format(combo, why))
    if skipped:
        print("\nSkipped {0} (emulator, mode, noise, qubits, nodes) point(s):".format(len(skipped)))
        for e, m, nz, n, M, why in sorted(skipped, key=lambda r: r[:5]):
            print("  {0:<12} {1:<8} {2:<13} q={3:>2} nodes={4}  --  {5}".format(e, m, nz, n, M, why))


def create_jobs(config_path, dry_run=False, allow_dirty=False):
    with open(config_path) as fh:
        config = yaml.safe_load(fh)
    campaign = config.get("campaign", "adhoc")
    jobs, excluded, skipped = plan(config)
    summarise(jobs, excluded, skipped)
    if dry_run:
        return jobs, excluded, skipped
    commit = git_commit(allow_dirty)
    stem = os.path.splitext(os.path.basename(config_path))[0]
    mpath = write_jobs(config, jobs, commit, campaign, stem)
    print("\nWrote {0} job scripts (commit {1}, campaign {2!r}); manifest: {3}".format(
        len(jobs), commit, campaign, os.path.relpath(mpath, REPO_ROOT)))
    return jobs, excluded, skipped


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("config", nargs="?",
                    default=os.path.join(REPO_ROOT, "configs", "sweep_params.yaml"))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--allow-dirty", action="store_true")
    a = ap.parse_args()
    create_jobs(os.path.abspath(a.config), dry_run=a.dry_run, allow_dirty=a.allow_dirty)
