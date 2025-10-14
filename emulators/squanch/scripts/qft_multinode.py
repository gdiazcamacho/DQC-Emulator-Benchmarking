#!/usr/bin/env python3
import logging
import numpy as np
import argparse
import csv

from typing import List
from collections import Counter

import os

#os.chdir("/mnt/Q_SWAP/gdiaz/D-QFT/collaboration/benchmarking-distributed-qft")

import time
import tracemalloc

from squanch import Agent, QStream, H, RZ, CNOT, CPHASE, X, Z


# ---------- Helpers ----------
def distribute_qubits(num_qubits: int, num_nodes: int):
    base = (num_qubits + num_nodes - 1) // num_nodes  # ceil
    qubits = [base] * num_nodes
    over = sum(qubits) - num_qubits
    for i in range(over):
        qubits[-(i + 1)] -= 1
    return qubits

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
    def __init__(self, qstream, out, *, name, peers, network_structure, theta, data_qubits, verbose=False):
        # DO NOT pass `verbose` to super()
        super().__init__(qstream, out, name=name)
        self.theta = theta
        self.network_structure = network_structure
        self.data_qubits = data_qubits
        self._peer_obj = {}
        self._g_offs = global_offsets(network_structure)
        self.peers = peers
        self.verbose = verbose  # just store it here

    def _my_local(self, qs):
        return [qs.qubit(i) for i in range(self.data_qubits)]

    def _comm_pair(self, qs):
        return qs.qubit(self.data_qubits), qs.qubit(self.data_qubits + 1)

    # Minimal CAT, no tags — enough for scaling study
    def _cat_entangle_source(self, q_ctrl, peer, b0, b1):
        H(b0); CNOT(b0, b1)
        self.qsend(self._peer_obj[peer], b1)
        CNOT(q_ctrl, b0)
        x = b0.measure()
        if x: X(b0)
        self.csend(self._peer_obj[peer], x)
        z = self.crecv(self._peer_obj[peer])
        if z: Z(q_ctrl)
        self.qrecv(self._peer_obj[peer])

    def _cat_entangle_target(self, peer):
        token = self.qrecv(self._peer_obj[peer])
        x = self.crecv(self._peer_obj[peer])
        if x: X(token)
        return token

    def _cat_disentangle_target(self, token, peer):
        H(token)
        z = token.measure()
        self.csend(self._peer_obj[peer], z)
        self.qsend(self._peer_obj[peer], token)

    def controlled_phase_gradient_source(self, local_qubits, peer, qs):
        self.csend(self._peer_obj[peer], len(local_qubits))
        b0, b1 = self._comm_pair(qs)
        base = self._g_offs[self.name]
        for k, q in enumerate(local_qubits):
            g_c = base + k
            self.csend(self._peer_obj[peer], g_c)
            self._cat_entangle_source(q, peer, b0, b1)

    def controlled_phase_gradient_target(self, local_qubits, peer):
        num_peer_qubits = self.crecv(self._peer_obj[peer])
        my_base = self._g_offs[self.name]
        for _ in range(num_peer_qubits):
            g_c = self.crecv(self._peer_obj[peer])
            token = self._cat_entangle_target(peer)
            for j, q_t in enumerate(local_qubits):
                g_t = my_base + j
                if g_t < g_c:
                    exp = g_c - g_t
                    angle = -np.pi / (2 ** exp)
                    CRz(token, q_t, angle)
            self._cat_disentangle_target(token, peer)

    def run(self):
        outputs = []
        for qs in self.qstream:
            local = self._my_local(qs)
            fourier_state(local, self.theta, start_from=self._g_offs[self.name])
            for peer in reversed(self.peers):
                if list(self.network_structure.keys()).index(peer) > self.index:
                    self.controlled_phase_gradient_target(local, peer)
            inverse_QFT_local(local)
            for peer in self.peers:
                if list(self.network_structure.keys()).index(peer) < self.index:
                    self.controlled_phase_gradient_source(local, peer, qs)
            outputs.append([q.measure() for q in local])
        self.output(outputs)

    @property
    def index(self):
        return list(self.network_structure.keys()).index(self.name)

# ---------- Runner ----------
def run_DQFT_squanch(theta, node_qubits, node_names, shots=1, verbose=False):
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

    # Each node gets: data + 2 comm ancillas
    per_node_qubits = [nq + 2 for nq in node_qubits]
    qstreams = [QStream(per_node_qubits[i], shots) for i in range(P)]
    out = Agent.shared_output()

    # Build agents
    nodes = {
        name: QFTNode(
            qstreams[i], out,
            name=name,
            peers=[n for n in node_names if n != name],
            network_structure=network,
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
    
    # Parse the arguments
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
        shots=shots
    )
    
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
    
    # Save to CSV
    csv_file_path = f"../results/{emulator}_{num_nodes}_q{num_qubits}_a{theta}.csv"
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            "Emulator", "Nodes", "Num Qubits", "Theta", "Shots",
            "Execution Time (s)", "Current Memory (KB)", "Peak Memory (KB)", "Result"
        ])
        writer.writerow([
            emulator, num_nodes, num_qubits, theta, shots,
            execution_time, current / 1024, peak / 1024, sorted_counts
        ])

