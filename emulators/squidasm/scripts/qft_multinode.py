import logging
import numpy as np
import argparse

from netqasm.sdk import Qubit
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
    distribute_qubits, fidelity_from_counts, write_result_row,
)
# ------------------------------------------------------------------------------

from squidasm.run.stack.run import run
from squidasm.sim.stack.program import ProgramContext
from squidasm.run.stack.config import StackNetworkConfig
# from squidasm.run.stack.common import AllocError
from AbstractNode import AbstractNode, CP
from copy import deepcopy
import yaml

####################################################

def setup_network_yaml(
        network_structure: dict,
        config_template: str = "config_template.yaml",
        temp_config_file: str = ".config_file.yaml"):

    node_names, node_qubits = list(zip(*network_structure.items()))
    num_nodes = len(node_qubits)

    # Read template config file
    with open(config_template, "r") as stream:
        try:
            config_yaml = (yaml.safe_load(stream)).copy()
        except yaml.YAMLError as exc:
            print(exc)

    # Copy default config from template
    stack0 = config_yaml["stacks"][0].copy()
    link0 = config_yaml["links"][0].copy()
    clink0 = config_yaml["clinks"][0].copy()

    # Delete default config
    del config_yaml["stacks"][0]
    del config_yaml["links"][0]
    del config_yaml["clinks"][0]

    # Add a new entry for each node, link and classical link
    for i, nq in enumerate(node_qubits):
        stack = deepcopy(stack0)  # Use template stack
        stack["name"] = node_names[i]  # Set nodename
        stack['qdevice_cfg']['num_qubits'] = nq + 1  # Add extra qubit for communication
        config_yaml["stacks"].append(stack)  # Add to new config file

        for j  in range(i+1, num_nodes):
            link = deepcopy(link0)  # Use template link
            clink = deepcopy(clink0)  # Use template classical link

            link["stack1"] = node_names[i]  # Set nodename of stack1
            link["stack2"] = node_names[j]  # Set nodename of stack2

            clink["stack1"] = node_names[i]  # Set nodename of stack1
            clink["stack2"] = node_names[j]  # Set nodename of stack2

            config_yaml["links"].append(link)  # Add to new config file
            config_yaml["clinks"].append(clink)  # Add to new config file

    # Save new config file to temporary yaml
    with open(temp_config_file, "w") as stream:
        try:
            yaml.safe_dump(config_yaml, stream, sort_keys=False, indent=4)
        except yaml.YAMLError as exc:
            print(exc)
    return StackNetworkConfig.from_file(temp_config_file)

def node_distance(node1, node2, network_structure):
    return max( 
        sum(
            [nq for i, nq in enumerate(network_structure.values()) 
            if list(network_structure.keys()).index(node1) 
            < i < 
            list(network_structure.keys()).index(node2) ]
        ),
        sum(
            [nq for i, nq in enumerate(network_structure.values()) 
            if list(network_structure.keys()).index(node2) 
            < i < 
            list(network_structure.keys()).index(node1) ]
        )
    )
    
    
class QFTNode(AbstractNode):
    def __init__(
            self,
            *args,
            theta,
            network_structure,
            verbose: bool = False,
            **kwargs
            ):

        super().__init__(*args, **kwargs)
        self.theta = theta
        self.dim_qft = self.max_qubits
        self.network_structure = network_structure
        self.index = list(self.network_structure.keys()).index(self.name)
        self.verbose = verbose

    def _fourier_state(
        self,
        qubits_qft: List[Qubit],
        theta,
        start_from: int = 0,
        verbose: bool = False
    ):
        """ Prepares a Fourier State of a given angle
    
        Parameters:
        - qubits_qft: list of ancilla qubits for inverse QFT
        - theta: angle for the fourier state
        - start_from: which order of Rz must be applied. e.g., if 4 it will begin with U^4
        - verbose: print logs
        """

        # Apply the U rotations, controlled at the QFT qubits and acting on the aux qubit
        for i, qubit in enumerate(qubits_qft):
            qubit.H()
            if self.verbose:
                print(f"{self.name}: (F) H({i})")

            qubit.rot_Z(angle=theta*np.pi*2**(i+1+start_from)) 
            if self.verbose:
                print(f"{self.name}: (F) Rz({theta*np.pi*2**(i+1+start_from)}, {i})")

    def controlled_phase_gradient_source(self, qubits: List[Qubit], peer: str, verbose: bool = False):
        """Send control to peer's qubits using CAT entanglement and apply CP from source side."""
        csocket = self.context.csockets[peer]
        
        # Send how many control qubits I have
        if self.verbose:
            print(f"{self.name}: sending num_qubits = {len(qubits)} to {peer}")
        msg = str(len(qubits))
        csocket.send(msg)
        yield from self.connection.flush()
        if self.verbose:
            print("\n")
        # For each control qubit, entangle and apply remote CP via CAT
        for q in qubits:
            yield from self.cat_entangle_source(q, peer)
            if self.verbose:
                print(f"{self.name}:cat entangle to {peer}")
            yield from self.cat_disentangle_source(q, peer)
            if self.verbose:
                print(f"{self.name}:cat disentangle to {peer}")

    def controlled_phase_gradient_target(self, qubits: List[Qubit], peer: str, verbose: bool = False):
        """Receive control from peer's qubits and apply CP locally."""
        csocket = self.context.csockets[peer]
        if self.verbose:
            print(f"{self.name}: waiting for num_peer_qubits from {peer}")
        msg = yield from csocket.recv()
        if self.verbose:
            print(f"{self.name}: received num_peer_qubits = {msg} from {peer}")
        num_peer_qubits = int(msg)
        num_own_qubits = len(qubits)

        for i in range(num_peer_qubits):
            epr = yield from self.cat_entangle_target(peer)
            for j, q in enumerate(qubits):
                distance = node_distance(self.name, peer, self.network_structure)

                angle = -np.pi / (2 ** (i + num_own_qubits - j + distance))
                CP(epr, q, angle=angle)
                if self.verbose:
                    print(f"{self.name}: (PG) CP({i}, {j}) = -p/{2**(i + num_own_qubits - j + distance)}")
            yield from self.cat_disentangle_target(epr, peer)

    def run(self, context: ProgramContext):
        self.context = context

        my_index = list(self.network_structure.keys()).index(self.name)
        if self.verbose:
            print(f"{self.name}: Hello world! I have {len(self.qubits)} qubits!")
            print(f"{self.name}: My index is {my_index}")
        # Initialise qubits
        [self.init_qubit(i) for i in range(self.max_qubits)]

        # Know which qubit is the one this node starts with
        offset = sum([nq for i, nq in enumerate(self.network_structure.values()) if i < my_index])

        # Prepare state in Fourier state (continuing from Alice's number of qubits)
        self._fourier_state(
            qubits_qft=self.qubits,
            theta=self.theta,
            start_from=offset,
            verbose=self.verbose
        )
        if self.verbose:
            print(f"{self.name}: Prepared Fourier state")

        # Step 1: Receive phase gradients 
        for peer in reversed(self.peers):
            peer_index = list(self.network_structure.keys()).index(peer)
            if peer_index > my_index:
                # I'm a target, I receive control from peer with higher indexes (if any)
                # if self.verbose:
                #     print(f"{self.name}: {my_index} receives from {peer_index}")
                yield from self.controlled_phase_gradient_target(self.qubits, peer, self.verbose)
                if self.verbose:
                    print(f"{self.name}: Received Phase Grandient from {peer}")          
            else:
                pass
        if self.verbose:
            print(f"{self.name}: I have no more peers")

        # Step 2: Apply local iQFT
        self.inverse_QFT(self.qubits, reorder=False,verbose=self.verbose)
        if self.verbose:
            print(f"{self.name}: Applied local iQFT")

        # Step 3: Apply distributed controlled-phase gradient
        # Interact with all earlier qubits from lower-index nodes
        for peer in self.peers:
            peer_index = list(self.network_structure.keys()).index(peer)
            if peer_index < my_index:
                # I'm a source, I receive control qubits from peer with lower indexes (if any)
                # if self.verbose:
                #     print(f"{self.name}: {my_index} sends to {peer_index}") 
                yield from self.controlled_phase_gradient_source(self.qubits, peer, self.verbose)
                if self.verbose:
                    print(f"{self.name}: Sent Phase Grandient to {peer}")   
            else:
                pass    
        if self.verbose:
            print(f"{self.name}: I have no more peers")

        # Measure qubits
        output = []
        for i, q in enumerate(self.qubits):
            if self.verbose:
                print(f"{self.name}: Measure({i})")
            result = q.measure()
            output += [result]

        yield from self.connection.flush()
        if self.verbose:
            print("\n")
        if self.verbose:
            print(f"{self.name}: Finished!","\n")
        
        return [int(o) for o in output]

def run_DQFT(
        theta: float,
        node_qubits=[4, 4],
        node_names=[],
        shots=1,
        config_template="config_template.yaml",
        verbose=False,
        delete_temp_file=True,
    ):

    network_structure = {node: nq for node, nq in zip(node_names, node_qubits)}
    temp_config_file =  f"{time.time()}-config.yaml"    

    # Generate config file and node names
    cfg  = setup_network_yaml(
        network_structure=network_structure,
        config_template=config_template,
        temp_config_file = temp_config_file
        )

    # Create program for each node
    programs = {
        name: QFTNode(
            name=name,
            peers=[n for n in network_structure.keys() if n != name],
            network_structure=network_structure,
            theta=theta,
            qubits=nq,
            verbose=verbose
            )
        for name, nq in network_structure.items()
    }

    # Run the simulation
    out = run(config=cfg, programs=programs, num_times=shots)

    # Delete temporary yaml
    if delete_temp_file:
        if os.path.exists(".config_file.yaml"):
            os.remove(".config_file.yaml")

    return out

    
####################################################
    
if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Run QFT on two nodes node.")
    parser.add_argument('--num_qubits', type=int, required=True, help='Number of qubits')
    parser.add_argument('--num_nodes', type=int, required=True, help='Number of nodes')    
    parser.add_argument('--theta', type=float, required=True, help='Initial angle theta')
    parser.add_argument('--shots', type=int, required=True, help='Number of shots')
    parser.add_argument('--comm_mode', type=str, default='packed',
                        choices=['packed'],
                        help="Communication schedule. Only 'packed' is "
                             "implemented here so far; the flag exists so the "
                             "job scaffolding can pass the axis uniformly.")
    
    # Parse the arguments
    args = parser.parse_args()
    
    emulator = "squidasm"
    verbose = False
    config_template = '../scripts/config_template.yaml'
    num_qubits = args.num_qubits
    num_nodes = args.num_nodes
    theta = args.theta
    shots = args.shots  
    
    node_qubits = distribute_qubits(num_qubits, num_nodes)  # List: number of qubits in each node
    node_names = [f"Node{n}" for n in range(len(node_qubits))]
    # node_names = ["Alice", "Bob", "Charlie", "David"]
    
    verbose = False  # Bool
    delete_temp_file = False  # Bool
    
    tracemalloc.start(); start_time = time.time()  # Start memory profiling and record the start time
    
    # Run main script
    out = run_DQFT(
      theta,
      node_qubits=node_qubits,
      node_names=node_names,
      shots=shots,
      config_template = '../scripts/config_template.yaml',
      verbose=verbose,
      delete_temp_file=delete_temp_file
      )
    
    execution_time = time.time() - start_time  # Calculate execution time
    current, peak = tracemalloc.get_traced_memory()  # Get memory usage
    tracemalloc.stop()  # Stop memory profiling
    
    # Print or save the results
    print(f"Execution Time: {execution_time} seconds")
    print(f"Current Memory Usage: {current / 1024} KB")
    print(f"Peak Memory Usage: {peak / 1024} KB")
    print(f"Result: {out}")
    
    # Stack horizontally to get (shots, total_qubits)
    combined = np.hstack(out)  # shape: (shots, total_qubits)
    # Convert each shot into a bitstring
    bitstrings = ["".join(map(str, row)) for row in combined]
    # Convert to float values (rescaled binary to [0,1))
    num_qubits = combined.shape[1]
    float_values = [int(bits, 2) / (2 ** total_qubits) for bits in bitstrings]
    # Count frequencies
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
        comm_mode=args.comm_mode,
    )
    print(f"Wrote {csv_file_path}")
