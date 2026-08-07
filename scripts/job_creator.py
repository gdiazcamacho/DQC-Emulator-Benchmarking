import os
import sys

import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))
from common.dqc_common import distribute_qubits

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


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
    num_qubits_range = config['num_qubits_range']
    num_nodes_range = config['num_nodes_range']
    theta_range = config['theta_range']
    shots = config['shots']
    # Minimum qubits a node must hold for the run to count as a distributed
    # benchmark. Declared in the config so the exclusion is auditable.
    min_qubits_per_node = config.get('min_qubits_per_node', 2)

    job_count = 0     # Initialize job counter
    skipped = {}      # (num_qubits, num_nodes) -> reason

    for emulator in emulators:
        for num_qubits in num_qubits_range:
            for theta in theta_range:
                for num_nodes in num_nodes_range:
                    ok, reason = is_feasible(num_qubits, num_nodes, min_qubits_per_node)
                    if not ok:
                        skipped[(num_qubits, num_nodes)] = reason
                        continue

                    job_name = f"{emulator}_{num_nodes}_q{num_qubits}_a{theta}"
                    slurm_script_path = os.path.join(
                        REPO_ROOT, "emulators", emulator, "jobs", f"{job_name}.sh")
                    slurm_template_path = os.path.join(
                        REPO_ROOT, "emulators", emulator, "slurm_template_multinode.sh")

                    # Create the job script from the template
                    with open(slurm_template_path, 'r') as template_file:
                        template_content = template_file.read()

                    job_content = template_content.replace('$1', str(num_qubits)).replace('$2',str(num_nodes)).replace('$3', str(theta)).replace('$4', str(shots))

                    # Save the job script
                    os.makedirs(os.path.dirname(slurm_script_path), exist_ok=True)
                    with open(slurm_script_path, 'w') as job_file:
                        job_file.write(job_content)

                    print(f"Created job script: {slurm_script_path}")
                    job_count += 1  # Increment job counter

    print(f"Total jobs created: {job_count}")

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

    return job_count, skipped


if __name__ == "__main__":
    _default_config = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "configs", "sweep_params.yaml"
    )
    create_jobs(os.path.abspath(_default_config))

