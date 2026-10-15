#!/usr/bin/env python3
import logging
import numpy as np
import argparse
import multiprocessing

from typing import List
from collections import Counter

import os

#os.chdir("/mnt/Q_SWAP/gdiaz/D-QFT/collaboration/benchmarking-distributed-qft")

import time
import tracemalloc

# --- shared helpers -----------------------------------------------------------
# Anchor on this file's location so the script runs identically from the repo
# root, from this directory, or from a SLURM working directory.
import os as _os, sys as _sys
_REPO_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", ".."))
if _REPO_ROOT not in _sys.path:
    _sys.path.insert(0, _REPO_ROOT)
from common.dqc_common import (
    distribute_qubits, fidelity_from_counts, write_result_row, package_version,
)
# ------------------------------------------------------------------------------

from squanch import Agent, QStream, H, RZ, CNOT, CPHASE, X, Z


# ---------- Helpers ----------
def global_offsets(network_structure):
    offs, total = {}, 0
    for name, nq in network_structure.items():
        offs[name] = total
        total += nq
    return offs

def fourier_state(local_qubits, theta, start_from=0):
    for i, q in enumerate(local_qubits):
        H(q)
        RZ(q, theta*np.pi*(2**(i+1+start_from)))

def inverse_QFT_local(local):
    n = len(local)
    for i in range(n - 1, 0, -1):
        H(local[i])
        for j in range(i - 1, -1, -1):
            CRz(local[i], local[j], angle=-np.pi/(2**(i-j)))
    H(local[0])

def CRz(control, target, angle):
    RZ(target, angle/2)
    CNOT(control, target)
    RZ(target, -angle/2)
    CNOT(control, target)

# ---------- QFTNode ----------
class QFTNode(Agent):
    def __init__(self, qstream, out, *, name, peers, network_structure, theta,
                 data_qubits, comm_mode="packed", qsystem_lock=None, verbose=False):
        # DO NOT pass `verbose` to super()
        super().__init__(qstream, out, name=name)
        self.comm_mode = comm_mode
        # Gate application mutates the shared density matrix in place
        # (QSystem.apply does `state[...] = U @ state @ U.conj().T`), which is
        # a read-modify-write of the whole array and is NOT atomic. Two agents
        # applying gates concurrently corrupt each other even on disjoint
        # qubits, and SQUANCH does not serialise this. The lock guards blocks
        # of gate applications and is never held across a send/recv.
        self.qsystem_lock = qsystem_lock
        self.theta = theta
        self.network_structure = network_structure
        self.data_qubits = data_qubits
        self._peer_obj = {}
        self._g_offs = global_offsets(network_structure)
        self.peers = peers
        self.verbose = verbose  # just store it here

    # Index layout inside the SINGLE shared qsystem:
    #   [0 .. n_total)        data qubits, in node order
    #   [n_total + 2i, +1]    node i's two communication ancillas
    #
    # All nodes address the SAME qsystem. SQUANCH's quantum channel transmits
    # only a reference -- Qubit.serialize() returns (system_index,
    # qubit_index) and the receiver resolves it against ITS OWN qstream via
    # Qubit.from_stream. With one QStream per node those indices name a
    # different, untouched qubit, so no entanglement can cross a node
    # boundary: measured Bell correlation was 0.500, i.e. chance.
    # See scripts/diagnose_squanch.py.

    @property
    def _n_total(self):
        return sum(self.network_structure.values())

    def _my_local(self, qs):
        base = self._g_offs[self.name]
        return [qs.qubit(base + i) for i in range(self.data_qubits)]

    def _comm_pair(self, qs):
        base = self._n_total + 2 * self.index
        return qs.qubit(base), qs.qubit(base + 1)

    def _locked(self):
        class _Null(object):
            def __enter__(self_):
                return None

            def __exit__(self_, *exc):
                return False
        return self.qsystem_lock if self.qsystem_lock is not None else _Null()

    # Minimal CAT, no tags — enough for scaling study
    def _cat_entangle_source(self, q_ctrl, peer, b0, b1):
        with self._locked():
            H(b0); CNOT(b0, b1)
        self.qsend(self._peer_obj[peer], b1)
        with self._locked():
            CNOT(q_ctrl, b0)
            x = b0.measure()
            if x: X(b0)          # reset to |0> for the next teleportation
        self.csend(self._peer_obj[peer], x)
        z = self.crecv(self._peer_obj[peer])
        with self._locked():
            if z: Z(q_ctrl)
        self.qrecv(self._peer_obj[peer])

    def _cat_entangle_target(self, peer):
        token = self.qrecv(self._peer_obj[peer])
        x = self.crecv(self._peer_obj[peer])
        with self._locked():
            if x: X(token)
        return token

    def _cat_disentangle_target(self, token, peer):
        with self._locked():
            H(token)
            z = token.measure()
            if z:
                # Reset to |0>. The ancilla is reused for the next control
                # qubit; H+CNOT on a dirty |1> prepares |Phi-> instead of
                # |Phi+>, silently corrupting every teleportation after the
                # first. This was masked entirely while the channel carried no
                # entanglement at all.
                X(token)
        self.csend(self._peer_obj[peer], z)
        self.qsend(self._peer_obj[peer], token)

    def controlled_phase_gradient_source(self, local_qubits, peer, qs):
        """Send this node's controls to a lower-index peer.

        packed: one teleportation per control qubit (the whole column of gates
                shares it).
        basic:  one teleportation per cross-node gate. The peer's qubit count
                is read from network_structure rather than negotiated, so the
                extra mode adds no handshake and no deadlock surface.
        """
        self.csend(self._peer_obj[peer], len(local_qubits))
        b0, b1 = self._comm_pair(qs)
        base = self._g_offs[self.name]
        n_target = self.network_structure[peer]
        for k, q in enumerate(local_qubits):
            g_c = base + k
            reps = 1 if self.comm_mode == "packed" else n_target
            for _ in range(reps):
                self.csend(self._peer_obj[peer], g_c)
                self._cat_entangle_source(q, peer, b0, b1)

    def controlled_phase_gradient_target(self, local_qubits, peer):
        num_peer_qubits = self.crecv(self._peer_obj[peer])
        my_base = self._g_offs[self.name]
        for _ in range(num_peer_qubits):
            if self.comm_mode == "packed":
                g_c = self.crecv(self._peer_obj[peer])
                token = self._cat_entangle_target(peer)
                with self._locked():
                    for j, q_t in enumerate(local_qubits):
                        g_t = my_base + j
                        if g_t < g_c:
                            CRz(token, q_t, -np.pi / (2 ** (g_c - g_t)))
                self._cat_disentangle_target(token, peer)
            else:  # basic: one teleportation per gate
                for j, q_t in enumerate(local_qubits):
                    g_c = self.crecv(self._peer_obj[peer])
                    token = self._cat_entangle_target(peer)
                    g_t = my_base + j
                    with self._locked():
                        if g_t < g_c:
                            CRz(token, q_t, -np.pi / (2 ** (g_c - g_t)))
                    self._cat_disentangle_target(token, peer)

    def run_dynamic(self):
        """'dynamic' comm mode: semiclassical inverse QFT, zero e-bits.

        Each node owns a PRIVATE QStream holding only its own data qubits (the
        Fourier input is a product state and nothing quantum crosses a node),
        so the simulated state is 2^(2m) per node instead of the joint
        2^(2(n+2P)). Per shot: receive every bit of every higher node (each
        sender emits in descending global order, so per-channel FIFO order
        matches), then for each own qubit from the highest down apply the
        classically conditioned phases, H, measure, and send the bit to every
        lower node.
        """
        names = list(self.network_structure.keys())
        n_total = self._n_total
        base = self._g_offs[self.name]
        higher = [p for p in names if names.index(p) > self.index]
        lower = [p for p in names if names.index(p) < self.index]
        outputs = []
        for qs in self.qstream:
            local = [qs.qubit(i) for i in range(self.data_qubits)]
            fourier_state(local, self.theta, start_from=base)
            bits = {}
            for peer in higher:
                pb = self._g_offs[peer]
                for g in range(pb + self.network_structure[peer] - 1, pb - 1, -1):
                    bits[g] = self.crecv(self._peer_obj[peer])
            for lj in range(self.data_qubits - 1, -1, -1):
                gj = base + lj
                q = local[lj]
                for gk in range(n_total - 1, gj, -1):
                    if bits[gk]:
                        RZ(q, -np.pi / (2 ** (gk - gj)))
                H(q)
                b = q.measure()
                bits[gj] = b
                for peer in lower:
                    self.csend(self._peer_obj[peer], b)
            outputs.append([bits[base + i] for i in range(self.data_qubits)])
        self.output(outputs)

    def run(self):
        if self.comm_mode == "dynamic":
            return self.run_dynamic()
        outputs = []
        for qs in self.qstream:
            local = self._my_local(qs)
            with self._locked():
                fourier_state(local, self.theta,
                              start_from=self._g_offs[self.name])
            for peer in reversed(self.peers):
                if list(self.network_structure.keys()).index(peer) > self.index:
                    self.controlled_phase_gradient_target(local, peer)
            with self._locked():
                inverse_QFT_local(local)
            for peer in self.peers:
                if list(self.network_structure.keys()).index(peer) < self.index:
                    self.controlled_phase_gradient_source(local, peer, qs)
            with self._locked():
                outputs.append([q.measure() for q in local])
        self.output(outputs)

    @property
    def index(self):
        return list(self.network_structure.keys()).index(self.name)

# ---------- Runner ----------
def expected_ebits(node_qubits, comm_mode):
    """Analytic e-bit cost. SQUANCH agents are separate processes, so a counter
    incremented inside a node would not come back to the parent.

        packed = m * C(M,2)              (one per control qubit, per peer node)
        basic  = n^2 (M-1) / (2M)        (one per cross-node gate)
    """
    sizes = list(node_qubits)
    n = sum(sizes)
    owner, acc = {}, 0
    for node_i, size in enumerate(sizes):
        for j in range(size):
            owner[acc + j] = node_i
        acc += size
    if comm_mode == "dynamic":
        return 0
    if comm_mode == "basic":
        return sum(1 for c in range(n) for t in range(c) if owner[c] != owner[t])
    return sum(len({owner[t] for t in range(c) if owner[t] != owner[c]})
               for c in range(n))


def run_DQFT_squanch(theta, node_qubits, node_names, shots=1, verbose=False,
                     comm_mode="packed"):
    """
    Run multinode D-QFT on Squanch and return per-node shot matrices,
    matching Squidasm's run_DQFT return format.

    Returns:
        List[np.ndarray]: one array per node (in node_names order),
                         each shaped (shots, qubits_on_that_node) with 0/1 ints.
    """
    # Ordered mapping: node -> #data qubits (no comm ancillas)
    network = {name: nq for name, nq in zip(node_names, node_qubits)}
    P = len(node_names)

    # ONE shared QStream. Every data qubit plus two communication ancillas per
    # node live in the SAME qsystem, which is what makes an inter-node Bell
    # pair representable at all: SQUANCH's channel passes (system_index,
    # qubit_index) and the receiver resolves it against its own stream, so with
    # per-node streams the reference names an untouched qubit.
    #
    # Cost: the qsystem is a DENSE density matrix of side 2^(n_total + 2P), so
    # memory and time grow with the total system, not per node. The previous
    # per-node-QStream version was cheap precisely because it was not
    # simulating the joint state.
    if comm_mode == "dynamic":
        # No entanglement crosses a node: one private stream per node, no lock.
        streams = [QStream(nq, shots) for nq in node_qubits]
        qsystem_lock = None
    else:
        system_size = sum(node_qubits) + 2 * P
        streams = [QStream(system_size, shots)] * P
        qsystem_lock = multiprocessing.Lock()
    out = Agent.shared_output()

    # Build agents
    nodes = {
        name: QFTNode(
            streams[i], out,
            name=name,
            peers=[n for n in node_names if n != name],
            network_structure=network,
            comm_mode=comm_mode,
            qsystem_lock=qsystem_lock,
            theta=theta,
            data_qubits=node_qubits[i],
            verbose=verbose,   # OK: consumed by QFTNode, not forwarded
        )
        for i, name in enumerate(node_names)
    }


    # Fully connect quantum + classical channels
    for a in node_names:
        for b in node_names:
            if a == b:
                continue
            nodes[a].qconnect(nodes[b])
            nodes[a].cconnect(nodes[b])
            nodes[a]._peer_obj[b] = nodes[b]

    # Run
    for n in nodes.values(): n.start()
    for n in nodes.values(): n.join()

    # Return per-node outputs as numpy arrays (shots, qubits_on_node)
    per_node_outputs = [np.asarray(out[name], dtype=int) for name in node_names]
    return per_node_outputs

# ---------- Main ----------
if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Run QFT on multiple nodes with Squanch.")
    parser.add_argument('--num_qubits', type=int, required=True, help='Number of qubits')
    parser.add_argument('--num_nodes', type=int, required=True, help='Number of nodes')    
    parser.add_argument('--theta', type=float, required=True, help='Initial angle theta')
    parser.add_argument('--shots', type=int, required=True, help='Number of shots')
    parser.add_argument('--comm_mode', type=str, default='packed',
                        choices=['basic', 'packed', 'dynamic'],
                        help="Communication schedule. 'dynamic' is the "
                             "semiclassical iQFT (zero e-bits, private "
                             "per-node streams). 'packed' shares one "
                             "teleportation across a whole column of gates; "
                             "'basic' uses one per cross-node gate.")

    # Parse the arguments
    parser.add_argument('--noise', type=str, default='none', choices=['none'],
                        help="SQUANCH runs noiseless here; accepted so the "
                             "scaffolding can pass the axis uniformly.")
    args = parser.parse_args()
    
    emulator = "squanch"
    verbose = False
    config_template = '../scripts/config_template.yaml'
    num_qubits = args.num_qubits
    num_nodes = args.num_nodes
    theta = args.theta
    shots = args.shots  

    node_qubits = distribute_qubits(num_qubits, num_nodes)  # List: number of qubits per node
    node_names = [f"Node{n}" for n in range(len(node_qubits))]

    tracemalloc.start(); start_time = time.time()  # Start memory profiling and record the start time
    delete_temp_file = False  # Placeholder to match Squidasm signature
    
    # Run main script
    out = run_DQFT_squanch(
        theta,
        node_qubits=node_qubits,
        node_names=node_names,
        shots=shots,
        comm_mode=args.comm_mode,
    )
    ebits_used = expected_ebits(node_qubits, args.comm_mode)
    
    execution_time = time.time() - start_time  # Calculate execution time
    current, peak = tracemalloc.get_traced_memory()  # Get memory usage
    tracemalloc.stop()  # Stop memory profiling
    
    # Print or save the results
    print(f"Execution Time: {execution_time} seconds")
    print(f"Current Memory Usage: {current / 1024} KB")
    print(f"Peak Memory Usage: {peak / 1024} KB")
    print(f"Result: {out}")
    
    # If run_DQFT_squanch returns per-node outputs, combine like Squidasm version
    # Example: if out is (phase_counts, bitstring_counts) -> adapt accordingly
    # Here assuming out is a list-of-lists like [[bits for shot0], [bits for shot1], ...] per node
    combined = np.hstack(out)  # shape: (shots, total_qubits)
    bitstrings = ["".join(map(str, row)) for row in combined]
    float_values = [int(bits, 2) / (2 ** combined.shape[1]) for bits in bitstrings]
    counts = dict(Counter(float_values))
    sorted_counts = dict(sorted(counts.items()))
    print(f"Phases Counts: {sorted_counts}")
    
    fidelity = fidelity_from_counts(sorted_counts, num_qubits, theta)
    print(f"Fidelity: {fidelity}")

    csv_file_path = write_result_row(
        emulator=emulator, num_nodes=num_nodes, num_qubits=num_qubits,
        theta=theta, shots=shots, execution_time=execution_time,
        current_kb=current / 1024, peak_kb=peak / 1024,
        result=sorted_counts, algorithm="qft", fidelity=fidelity,
        ebits=ebits_used, comm_mode=args.comm_mode, noise=args.noise,
        emulator_version=package_version("squanch"),
    )
    print(f"Wrote {csv_file_path}")
