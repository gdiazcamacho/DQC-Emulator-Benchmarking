"""
Distributed inverse QFT on CUNQA.

Follows the same packed-column construction as the other emulators in this
suite (see emulators/qiskit_aer/scripts/qft_multinode.py): qubits are
partitioned across `num_nodes` vQPUs; each node prepares its share of the
Fourier state locally; the ladder of controlled-phase rotations is then
applied node-by-node in descending order, with rotations that cross a node
boundary carried out via a telegate.

CUNQA's cat_entangler/cat_disentangler *are* your cat-entangle/cat-disentangle
primitive under a different name -- the entangler shares one control qubit
onto the comm qubits of every node named in `target_circuits` in a single
round, which is exactly the packed-column construction: a single entangling
round serving every remote target rotation in that column, rather than one
round per target. This script relies on that to reproduce the O(n) (not
O(n^2)) e-bit scaling already established for the other backends.

IMPORTANT -- validation status: this script is written against the CUNQA
1.x API (github.com/CESGA-Quantum-Spain/cunqa, read as of 2026-08-07) but has
not yet been run against a live deployment. Before it is trusted for the
sweep, follow the suite's layered validation approach:
  1. Run the 2-qubit / 2-node case and check the output distribution by hand.
  2. Cross-check against the noiseless Qiskit Aer distribution for the same
     (num_qubits, theta) at a matched partition.
  3. Only then fold into the full sweep.

Known gaps versus the other four scripts (see CHANGELOG.md / PR description):
  - No in-process memory hook: CUNQA's Result object carries counts and
    time_taken, not memory. Peak/current memory here are measured on the
    *driver* process only (this script's own resource.getrusage), which does
    NOT include the vQPU server processes qraise spawns. This is a real
    methodology difference from the other four scripts (which measure the
    simulation process itself) and should be disclosed as such in Methods,
    not silently treated as comparable.
  - Noise: CUNQA's noise model is currently wired up for the no-communication
    scheme only (confirmed by reading the backend JSON schemas, not just the
    docs). A `--backend` noise config passed here would be accepted by qraise
    but silently not applied to the quantum-communication path as of the
    version read. Do not use this script for the Phase 3 noise sweep until
    that changes upstream; noiseless comparison only.
  - Execution model: qraise/qdrop happen once per run (one vQPU family per
    benchmark point), matching the per-run SLURM-job model the other four
    scripts use, at the cost of the amortization CUNQA is normally used for.

  - Result correlation across nodes (resolved below, but worth recording why
    it needed a workaround): each `Result.counts` is a per-circuit histogram
    aggregated over shots -- confirmed by reading result.py, which builds
    `counts` straight from one circuit's own classical register. Reading the
    C++ executor (src/sim/backends/quantum_comm/qc_executor.cpp) confirms the
    underlying shot loop *is* correlated -- one shared simulator steps every
    node through the same shot before moving to the next -- but that pairing
    is not exposed back through the Python API as a joint bitstring. Two
    independent per-node marginal histograms cannot be recombined into the
    joint distribution the fidelity metric needs, since the nodes' final
    measurement outcomes are entangled, not independent.

    Fix: rather than measuring locally on every node, every non-zero node
    classically relays its measured bits to node 0 with `send`/`recv` -- the
    same primitive `cat_entangler` itself uses for its correction bits, and
    it executes inside the same per-shot lockstep loop, so the pairing is
    exact. Node 0 ends up holding the full global bitstring for every shot in
    one classical register, and `results[0].counts` is directly the joint
    distribution. This costs classical bits only, no extra e-bits, and is
    counted separately from the ladder's own communication cost.
"""

import argparse
import os as _os
import resource
import sys as _sys
import time

import numpy as np

# --- shared helpers -----------------------------------------------------------
_REPO_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", ".."))
if _REPO_ROOT not in _sys.path:
    _sys.path.insert(0, _REPO_ROOT)
from common.dqc_common import distribute_qubits, fidelity_from_counts, write_result_row
# ------------------------------------------------------------------------------

# cunqa is only importable on a CESGA login/compute node with the module
# installed under $HOME (see README). Imported lazily so the rest of the suite
# (tests, plotting) does not require it.
_sys.path.append(_os.getenv("HOME", ""))


def myqft_inverse_local(circuit, qubits, swap_qubits=False):
    """Apply the local (single-node) part of the inverse QFT to `qubits`.

    Mirrors emulators/qiskit_aer/scripts/qft_multinode.py::myqft_inverse.
    Uses `cp`, not `crz` -- the inverse QFT is built from controlled-PHASE
    gates. CunqaCircuit exposes both as distinct methods (same trap as
    Qiskit), so this is stated explicitly rather than left implicit.
    """
    n = len(qubits)
    for j in range(n - 1, -1, -1):
        circuit.h(qubits[j])
        for k in range(j - 1, -1, -1):
            angle = -np.pi / 2 ** (j - k)
            circuit.cp(angle, qubits[k], qubits[j])
    if swap_qubits:
        for i in range(n // 2):
            circuit.swap(qubits[i], qubits[n - 1 - i])


def build_distributed_iqft(node_sizes, theta):
    """Build one CunqaCircuit per node implementing the packed-column
    distributed inverse QFT.

    Returns (circuits, node0_global_clbits) where circuits[i] is the
    CunqaCircuit for node i, and node0_global_clbits gives the classical bit
    index on node 0 holding each global qubit's measurement outcome, in
    global-qubit order (0 = most significant, matching distribute_qubits'
    node ordering). Every node measures its own data qubits locally; every
    non-zero node then classically relays its bits to node 0 (see module
    docstring), so results[0].counts is the joint distribution directly.
    """
    from cunqa.circuit import CunqaCircuit

    num_nodes = len(node_sizes)
    num_qubits = sum(node_sizes)

    # Clbit budget per node:
    #   [0, 1)              -- reused scratch bits for telegate corrections
    #                          during the ladder (2 bits; telegate blocks are
    #                          processed sequentially so reuse is safe)
    #   node 0 only: [2, 2 + num_qubits) -- final joint bitstring
    #   node i>0:    [2, 2 + node_sizes[i]) -- local measurement, then relayed
    TELEGATE_SCRATCH = 2
    circuits = []
    for i, size in enumerate(node_sizes):
        n_clbits = TELEGATE_SCRATCH + (num_qubits if i == 0 else size)
        circuits.append(CunqaCircuit((size, 1), num_clbits=n_clbits, id=f"node_{i}"))

    data_qubits, comm_qubits = [], []
    for c in circuits:
        d, cm = c.get_qubits()
        data_qubits.append(d)
        comm_qubits.append(cm)

    # 1. Local state preparation: fourier_state(theta) on every physical qubit,
    #    numbered globally 0..N-1 in node order (matches distribute_qubits).
    global_index = 0
    for node, size in enumerate(node_sizes):
        for local_q in range(size):
            circuits[node].h(data_qubits[node][local_q])
            circuits[node].rz(theta * 2 * np.pi * 2 ** global_index, data_qubits[node][local_q])
            global_index += 1

    # 2. Packed-column ladder, descending node order (mirrors
    #    recursive_qft_telegate in the Qiskit script).
    from cunqa.qc_protocols import cat_entangler, cat_disentangler

    for i in range(num_nodes - 1, -1, -1):
        control_circuit = circuits[i]
        control_qubits = data_qubits[i]

        # local column within the control node
        myqft_inverse_local(control_circuit, control_qubits, swap_qubits=False)

        target_nodes = list(range(i - 1, -1, -1))
        if not target_nodes:
            continue

        for local_j, control_qubit in enumerate(control_qubits):
            remote_nodes = target_nodes
            block = [control_circuit] + [circuits[t] for t in remote_nodes]
            block_comm = [comm_qubits[i][0]] + [comm_qubits[t][0] for t in remote_nodes]
            clbits = [0] * (1 + len(remote_nodes))

            cat_entangler(block, control_qubit, block_comm, clbits, tag=f"col_{i}_{local_j}")

            for t in remote_nodes:
                for local_k, target_qubit in enumerate(data_qubits[t]):
                    global_target = sum(node_sizes[:t]) + local_k
                    global_control = sum(node_sizes[:i]) + local_j
                    angle = -np.pi / 2 ** (global_control - global_target)
                    circuits[t].cp(angle, comm_qubits[t][0], target_qubit)

            cat_disentangler(
                block, control_qubit,
                [comm_qubits[t][0] for t in remote_nodes],
                list(range(1, len(remote_nodes) + 1)),
                [0] * len(remote_nodes),
            )

    # 3. Local measurement, then classical relay of every non-zero node's
    #    bits into node 0's classical register (see module docstring).
    node0_global_clbits = [None] * num_qubits
    global_index = 0
    for node, size in enumerate(node_sizes):
        local_clbits = list(range(TELEGATE_SCRATCH, TELEGATE_SCRATCH + size))
        circuits[node].measure(data_qubits[node], local_clbits)

        if node == 0:
            for local_c, clbit in enumerate(local_clbits):
                node0_global_clbits[global_index + local_c] = clbit
        else:
            # Relay each bit to a fresh clbit on node 0, appended after the
            # bits already claimed there.
            for local_c, clbit in enumerate(local_clbits):
                dest = TELEGATE_SCRATCH + node_sizes[0] + sum(
                    node_sizes[1:node]) + local_c
                circuits[node].send([clbit], circuits[0])
                circuits[0].recv([dest], circuits[node])
                node0_global_clbits[global_index + local_c] = dest

        global_index += size

    return circuits, node0_global_clbits


def main():
    parser = argparse.ArgumentParser(description="Distributed inverse QFT on CUNQA.")
    parser.add_argument("--num_qubits", type=int, required=True)
    parser.add_argument("--num_nodes", type=int, required=True)
    parser.add_argument("--theta", type=float, required=True)
    parser.add_argument("--shots", type=int, default=100)
    parser.add_argument("--comm_mode", type=str, default="packed",
                        choices=["packed"],
                        help="Communication schedule. Only 'packed' is "
                             "implemented: this script builds the "
                             "packed-column construction explicitly. The flag "
                             "exists so the job scaffolding can pass the axis "
                             "uniformly to every emulator.")
    parser.add_argument("--max_time", type=str, default="00:20:00",
                        help="SLURM wall-clock budget for the vQPU allocation (D-HH:MM:SS)")
    args = parser.parse_args()

    node_sizes = distribute_qubits(args.num_qubits, args.num_nodes)

    from cunqa.qpu import get_QPUs, qraise, qdrop, run
    from cunqa.qjob import gather

    family = qraise(
        args.num_nodes, args.max_time,
        quantum_comm=True, simulator="Aer", co_located=True,
    )
    try:
        qpus = get_QPUs(co_located=True, family=family)
        circuits, node0_global_clbits = build_distributed_iqft(node_sizes, args.theta)

        start = time.time()
        qjobs = run(circuits, qpus, shots=args.shots)
        results = gather(qjobs)
        execution_time = time.time() - start

        # results[0].counts keys are node 0's full classical register as one
        # contiguous bitstring (see Result.counts docstring: a single
        # register is not space-separated). node0_global_clbits gives the
        # position of each global qubit's bit within that string.
        #
        # BIT ORDER IS UNVERIFIED. The other four scripts in this suite each
        # needed their own reversal fix at some point (classical_reverse() in
        # the Qiskit script exists for exactly this reason -- see also the
        # "palindrome targets mask ordering bugs" entry in the project notes).
        # This line assumes CunqaCircuit reports classical bit `c` at string
        # index `c`; do not trust it until validation step 1 (a small,
        # non-palindromic theta, checked by hand) confirms it on-cluster.
        raw_counts = results[0].counts
        phases = {}
        for bitstring, count in raw_counts.items():
            ordered_bits = "".join(bitstring[c] for c in node0_global_clbits)
            val = int(ordered_bits, 2) / 2 ** args.num_qubits
            phases[val] = phases.get(val, 0) + count

    finally:
        qdrop(family)

    peak_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # driver process only; see module docstring

    fidelity = fidelity_from_counts(phases, args.num_qubits, args.theta)
    print(f"Fidelity: {fidelity}")

    csv_file_path = write_result_row(
        emulator="cunqa", num_nodes=args.num_nodes, num_qubits=args.num_qubits,
        theta=args.theta, shots=args.shots, execution_time=execution_time,
        current_kb=peak_kb, peak_kb=peak_kb,
        result=phases, algorithm="qft", fidelity=fidelity,
        comm_mode=args.comm_mode,
    )
    print(f"Wrote {csv_file_path}")
    print(
        "NOTE: current/peak memory above are the driver process only "
        "(resource.getrusage), not the vQPU server -- see script header."
    )


if __name__ == "__main__":
    main()
