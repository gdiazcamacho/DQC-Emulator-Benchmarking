import os
import sys
import warnings

import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))
from common.dqc_common import distribute_qubits

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


# --------------------------------------------------------------------------- #
# Capability matrix
# --------------------------------------------------------------------------- #
# Which (emulator, algorithm, comm_mode) triples actually exist. The sweep grid
# is a full cartesian product, so without this it emits jobs that cannot run --
# e.g. interlin-q + 'packed' (its partitioner inserts teleportation itself and
# exposes no hook for our packing), or any emulator + a comm_mode its script
# does not implement yet.
#
# Excluded combinations are reported rather than silently dropped, so the sweep
# stays auditable and the exclusions are quotable in the paper.
#
# THIS REFLECTS WHAT THE SCRIPTS SUPPORT TODAY. Widen an entry only once the
# corresponding script accepts that --comm_mode.
SUPPORTED_COMM_MODES = {
    # Aer already distinguishes the two via --telegate_baseline.
    "qiskit_aer":  {"basic", "packed"},
    "squidasm":    {"packed"},
    # Both modes verified at fidelity 1.0 after the shared-QStream and
    # ancilla-reset fixes (see CHANGELOG / diagnose_squanch.py).
    "squanch":     {"basic", "packed"},
    # CUNQA builds every schedule explicitly (user-controlled, so never
    # 'native'): packed/basic over quantum_comm telegates (cat_entangler ==
    # our cat_entangle == NetQMPI expose), dynamic as the semiclassical iQFT
    # over classical_comm. All three verified on the offline shadow executor
    # (emulators/cunqa/scripts/cunqa_shadow.py); not yet on a live deployment.
    "cunqa":       {"basic", "packed", "dynamic"},
    # Interlin-q partitions automatically; the teleportation schedule is not
    # user-controlled and its e-bit count is not comparable to ours.
    "interlin-q":  {"native"},
    "simdisq-net": {"native"},
}

SUPPORTED_ALGORITHMS = {
    "qiskit_aer":  {"qft"},
    "squidasm":    {"qft"},
    "squanch":     {"qft"},
    "cunqa":       {"qft"},
    "interlin-q":  {"qft"},
    "simdisq-net": {"qft"},
}

# The deferred-measurement construction is only valid when the transform is
# TERMINAL (immediately followed by computational-basis measurement).
DYNAMIC_ONLY_FOR = {"qft"}


def is_supported(emulator, algorithm, comm_mode):
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
    return True, ""


def is_feasible(num_qubits, num_nodes, min_qubits_per_node=2):
    """Whether a (qubits, nodes) point is a valid benchmark configuration.

    The sweep grid is a full cartesian product, so it contains points where the
    partition is impossible (fewer qubits than nodes) or degenerate (a node
    holding a single qubit performs no local controlled rotations, so the
    "distributed" run measures communication overhead against an empty local
    circuit).

    Impossible points are always rejected: the old code accepted them silently
    and returned partitions containing zero-qubit nodes.

    Degenerate points are rejected according to `min_qubits_per_node`, which is
    read from the sweep configuration. Set it to 1 in configs/sweep_params.yaml
    to reproduce the original grid.
    """
    try:
        # distribute_qubits warns about degenerate partitions, which is useful
        # when an emulator script calls it but pure noise here: this function
        # calls it precisely in order to REJECT those partitions, and the
        # rejection is already reported in the skipped-points summary.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            partition = distribute_qubits(num_qubits, num_nodes)
    except ValueError as exc:
        return False, str(exc)

    if min(partition) < min_qubits_per_node:
        return False, (
            "partition {0} has a node below min_qubits_per_node={1}".format(
                partition, min_qubits_per_node)
        )
    return True, ""


def create_jobs(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    emulators = config['emulators']
    algorithms = config.get('algorithms', ['qft'])
    comm_modes = config.get('comm_modes', ['packed'])
    num_qubits_range = config['num_qubits_range']
    num_nodes_range = config['num_nodes_range']
    theta_range = config['theta_range']
    shots = config['shots']
    # Minimum qubits a node must hold for the run to count as a distributed
    # benchmark. Declared in the config so the exclusion is auditable.
    min_qubits_per_node = config.get('min_qubits_per_node', 2)

    job_count = 0     # Initialize job counter
    skipped = {}      # (num_qubits, num_nodes) -> reason
    excluded = []     # (combination, reason)

    for emulator in emulators:
        slurm_template_path = os.path.join(
            REPO_ROOT, "emulators", emulator, "slurm_template_multinode.sh")
        if not os.path.exists(slurm_template_path):
            excluded.append((emulator, "no slurm_template_multinode.sh"))
            continue

        for algorithm in algorithms:
          for comm_mode in comm_modes:
            ok, reason = is_supported(emulator, algorithm, comm_mode)
            if not ok:
                excluded.append((f"{emulator}/{algorithm}/{comm_mode}", reason))
                continue

            for num_qubits in num_qubits_range:
              for theta in theta_range:
                for num_nodes in num_nodes_range:
                    ok, reason = is_feasible(num_qubits, num_nodes, min_qubits_per_node)
                    if not ok:
                        skipped[(num_qubits, num_nodes)] = reason
                        continue

                    job_name = f"{emulator}_{algorithm}_{comm_mode}_{num_nodes}_q{num_qubits}_a{theta}"
                    slurm_script_path = os.path.join(
                        REPO_ROOT, "emulators", emulator, "jobs", f"{job_name}.sh")

                    # Create the job script from the template
                    with open(slurm_template_path, 'r') as template_file:
                        template_content = template_file.read()

                    # Named placeholders: self-documenting, and immune to the
                    # ordering hazard of positional $1..$N substitution.
                    substitutions = {
                        "{{JOB_NAME}}":   job_name,
                        "{{ALGORITHM}}":  str(algorithm),
                        "{{COMM_MODE}}":  str(comm_mode),
                        "{{NUM_QUBITS}}": str(num_qubits),
                        "{{NUM_NODES}}":  str(num_nodes),
                        "{{THETA}}":      str(theta),
                        "{{SHOTS}}":      str(shots),
                    }
                    job_content = template_content
                    for placeholder, value in substitutions.items():
                        job_content = job_content.replace(placeholder, value)

                    if "{{" in job_content:
                        raise ValueError(
                            f"Unresolved placeholder in {slurm_script_path}; "
                            f"check {slurm_template_path}")

                    # Save the job script
                    os.makedirs(os.path.dirname(slurm_script_path), exist_ok=True)
                    with open(slurm_script_path, 'w') as job_file:
                        job_file.write(job_content)

                    print(f"Created job script: {slurm_script_path}")
                    job_count += 1  # Increment job counter

    print(f"Total jobs created: {job_count}")

    if excluded:
        print(f"\nExcluded {len(excluded)} (emulator, algorithm, comm_mode) "
              f"combination(s):")
        for combo, reason in excluded:
            print(f"  - {combo}: {reason}")

    if skipped:
        print(f"\nSkipped {len(skipped)} (qubits, nodes) combination(s) "
              f"at min_qubits_per_node={min_qubits_per_node}:")
        for (num_qubits, num_nodes) in sorted(skipped):
            print(f"  qubits={num_qubits:>3}, nodes={num_nodes:>2}  --  {skipped[(num_qubits, num_nodes)]}")
        print(
            "\nSet min_qubits_per_node: 1 in the sweep config to include "
            "single-qubit-node partitions (impossible partitions are always "
            "excluded)."
        )

    return job_count, skipped, excluded


if __name__ == "__main__":
    _default_config = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "configs", "sweep_params.yaml"
    )
    create_jobs(os.path.abspath(_default_config))

