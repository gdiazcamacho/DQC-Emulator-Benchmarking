
import numpy as np
from qiskit import QuantumCircuit, transpile, QuantumRegister, ClassicalRegister
from qiskit_aer.noise import NoiseModel, depolarizing_error
from qiskit_aer import AerSimulator
from qiskit.result import marginal_counts
import string
import argparse

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
    distribute_qubits, partition_list, fidelity_from_counts, write_result_row,
)
# ------------------------------------------------------------------------------
################################################################################


def print_results(counts):
    # Extract the most frequent measurement (the best estimate of theta)
    most_frequent_output = max(counts, key=counts.get)
    total = sum([v for v in counts.values()])
    # Convert the binary string to a fraction
    binary_fraction = int(most_frequent_output, 2) / (2 ** len(most_frequent_output))

    # Theta is the binary fraction
    estimated_theta = binary_fraction

    print(f"Measured output: {most_frequent_output}")
    print(f"Estimated theta: {estimated_theta}")
    print(f"With probability: {counts[most_frequent_output]/total}", )

def create_alphabetical_registers(sizes):
    registers = []
    for i, size in enumerate(sizes):
        name = string.ascii_uppercase[i] if i < 26 else \
               string.ascii_uppercase[i//26-1] + string.ascii_uppercase[i%26]
        registers.append(QuantumRegister(size, name=name))
    return registers

def fourier_state(circ, theta, qubits=None):
    if qubits==None:
        qubits=circ.qubits
    for i,q in enumerate(qubits):
        circ.h(q)
        circ.rz(theta*2*np.pi * 2**i, q)  

def myqft_inverse(n, swap_qubits=True):
    """Apply the Inverse Quantum Fourier Transform in big-endian convention."""
    circuit = QuantumCircuit(n)
    # Apply Hadamard and controlled phase gates starting from the MSB
    for j in range(n - 1, -1, -1):
        circuit.h(j)
        for k in range(j - 1, -1, -1):
            circuit.cp(-np.pi / 2 ** (j - k), j, k)
        circuit.barrier()
    # Swap qubits to correct the order
    if swap_qubits:
        for j in range(n // 2):
            circuit.swap(j, n - j - 1)
    return circuit

def cat_entangle(circ, qubit, EPR_qubits, clbit):
    circ.cx(qubit,EPR_qubits[0])
    circ.measure(EPR_qubits[0],clbit)
    with circ.if_test((clbit,1)):
        circ.x(EPR_qubits[1])
        # RESET
        circ.x(EPR_qubits[0])
        
def cat_disentangle(circ, qubit, EPR_qubits, clbit):
    circ.h(EPR_qubits[1])
    circ.measure(EPR_qubits[1],clbit)
    with circ.if_test((clbit,1)):
        circ.z(qubit)
        # RESET
        circ.x(EPR_qubits[1])

def bell_pair(circ, qubits):
    circ.h(qubits[0])
    circ.cx(qubits[0],qubits[1])
    
def classical_reverse(counts):
    reversed_counts = {}
    for k,v in counts.items():
        reversed_counts[k[::-1]]=v
    return reversed_counts

def recursive_qft(n,theta, partitions, swap_qubits=True, min_rotation=0, verbose=False):
    data_qubits = [q for q in range(n)]
    indices =  partition_list(data_qubits, distribute_qubits(n,partitions))[::-1]

    # Create nodes for all qubits
    registers = create_alphabetical_registers([len(node) for node in indices])
    cregister = ClassicalRegister(n, name="m")

    circ = QuantumCircuit(*registers,cregister)

    fourier_nodes = [reg for reg in registers]
    fourier_qubits = [item for sublist in fourier_nodes for item in sublist] 

    fourier_state(circ,theta, qubits = fourier_qubits)


    # Start from the bottom one, to make an ascending ladder
    for i,control_node in enumerate(registers[::-1]):
        if verbose:
            print('Control Node',control_node)
        circ.compose(myqft_inverse(len(control_node), swap_qubits=False), 
                   control_node, inplace=True)

        # NB: do not bind the loop variable to `n` here -- `n` is the qubit count
        # parameter of this function and was previously shadowed by a register.
        target_nodes = [nd for nd in registers[::-1][i+1:]]
        # If target qubits is empty (i.e. this is the last register), don't do the z_block
        if target_nodes:
            for j,target_node in enumerate(target_nodes):
                if verbose:
                    print('Target Node',target_node)
                for control in control_node:
                    if verbose:
                        print('Control',control)
                    # Global indices, which are needed to know the angle of the rotation
                    # Larger distance = Smaller rotation
                    control_index = fourier_qubits.index(control)
                    target_indices = [fourier_qubits.index(q) for q in target_node]

                    angles = -np.pi/2**(control_index-np.array(target_indices, dtype=float))

                    # If all the angles are smaller than the min_rotation, don't do anything 
                    if any(abs(angles) >= min_rotation):

                        for k,target in enumerate(target_node):
                            if verbose:
                                print('Target',target)
                            if abs(angles[k]) >= min_rotation:
                                # Only apply gates larger than the min_rotation
                                # `cp`, not `crz`: the inverse QFT is built from
                                # controlled-PHASE gates. See common/dqc_common.py
                                # and tests/test_primitives.py for the identity
                                # crz(t) = cp(t) . P_control(-t/2), and note that
                                # myqft_inverse() above already uses cp.
                                circ.cp(angles[k], control, target)

                    circ.barrier()
                circ.barrier()

#     if swap_qubits:
#         circ = reverse_qubits(circ)

    for i,q in enumerate(fourier_qubits):
        circ.measure(q,i)
        
    return circ

def recursive_qft_telegate(n,theta, num_nodes,swap_qubits=True, min_rotation=0, verbose=False):
    data_qubits = [q for q in range(n)]
    indices = partition_list(data_qubits, distribute_qubits(n, num_nodes))[::-1]

    # Create nodes for all qubits, with an additional communication qubit per node
    registers = create_alphabetical_registers([len(node)+1 for node in indices])
    cregister = ClassicalRegister(n, name="m")

    circ = QuantumCircuit(*registers,cregister)

    # For classical control we use the 0-th qubit of the classical register
    # It will be overwritten!

    fourier_nodes = [reg[0:-1] for reg in registers]
    fourier_qubits = [item for sublist in fourier_nodes for item in sublist] 

#     comm_qubits = [reg[-1] for reg in registers]

    fourier_state(circ,theta, qubits = fourier_qubits)

    # Start from the bottom one, to make an ascending ladder
    for i,control_node in enumerate(registers[::-1]):
        if verbose:
            print('Control Node',control_node)
        circ.compose(myqft_inverse(len(control_node[0:-1]), swap_qubits=False), 
                   control_node[0:-1], inplace=True)

        # NB: do not bind the loop variable to `n` (see recursive_qft).
        target_nodes = [nd for nd in registers[::-1][i+1:]]
        # If target qubits is empty (i.e. this is the last register), don't do the z_block
        if target_nodes:
            for j,target_node in enumerate(target_nodes):
                if verbose:
                    print('Target Node',target_node)
                for control in control_node[0:-1]:
                    if verbose:
                        print('Control',control)
                    # Global indices, which are needed to know the angle of the rotation
                    # Larger distance = Smaller rotation
                    control_index = fourier_qubits.index(control)
                    target_indices = [fourier_qubits.index(q) for q in target_node[0:-1]]

                    control_EPR = control_node[-1] 
                    target_EPR = target_node[-1]

                    angles = -np.pi/2**(control_index-np.array(target_indices, dtype=float))

                    # If all the angles are smaller than the min_rotation, don't do anything 
                    if any(abs(angles) >= min_rotation):

                        # The whole column of cp gates shares the same control: only one EPR
                        # Adds a register for the EPR
                        # We could add a classical register too, but for now reuse some clbits

                        EPR_qubits = [control_EPR,target_EPR]
                #         circ.add_register(EPR_clbits)
                        # Entangle them to create a Bell pair for the telegate
                        bell_pair(circ,EPR_qubits)

                        cat_entangle(circ, control, EPR_qubits, circ.clbits[0])

                        for k,target in enumerate(target_node[0:-1]):
                            if verbose:
                                print('Target',target)
                            if abs(angles[k]) >= min_rotation:
                                # Only apply gates larger than the min_rotation
                                # `cp`, not `crz` -- matches myqft_inverse() and the
                                # SquidASM implementation (AbstractNode.CP).
                                circ.cp(angles[k], EPR_qubits[1], target)

                        cat_disentangle(circ, control, EPR_qubits, circ.clbits[0])    

                    circ.barrier()
                circ.barrier()

#     if swap_qubits:
#         circ = reverse_qubits(circ)

    for i,q in enumerate(fourier_qubits):
        circ.measure(q,i)
        
    return circ

###############################

def run_DQFT(num_qft_qubits,num_nodes,theta,shots,verbose):

    # Create a simple noisy model
    noise_model = NoiseModel()
    noise_model.add_all_qubit_quantum_error(depolarizing_error(0.01, 1), ['x', 'h'])
    noise_model.add_all_qubit_quantum_error(depolarizing_error(0.02, 2), ['cx'])

    backend = AerSimulator(noise_model = noise_model)

#     all_qubits = [q for q in range(num_qft_qubits)]
#     nodes = partition_list(all_qubits, partitions)[::-1]

    dqc = recursive_qft_telegate(num_qft_qubits, theta, num_nodes, swap_qubits=False)
    dqct = transpile(dqc,backend)
    
    counts_d_all = backend.run(dqct, shots = shots).result().get_counts()

    # We only care about the data qubits!
    counts_d = marginal_counts(counts_d_all, indices=[q for q in range(num_qft_qubits)])

    reverse = True
    if reverse:
        counts_d = classical_reverse(counts_d)
        
    if verbose:
        print_results(counts_d)
    
    phases = {}
    for bitstring,value in counts_d.items():
        n = len(bitstring)
        k = int(bitstring, 2) / (2 ** n)
        phases[k]=value
    
    return phases
    
###############################################
if __name__ == "__main__":
    # Input variables
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Run QFT on a single node.")
    parser.add_argument('--num_qubits', type=int, required=True, help='Number of qubits')
    parser.add_argument('--num_nodes', type=int, required=True, help='Number of nodes')       
    parser.add_argument('--theta', type=float, required=True, help='Initial angle theta')
    parser.add_argument('--shots', type=int, required=True, help='Number of shots')

    # Parse the arguments
    args = parser.parse_args()
    emulator = "qiskit_aer"
    verbose = False
    config_file = 'config-QFT-noiseless.yaml'
    num_qubits = args.num_qubits
    num_nodes = args.num_nodes
    theta = args.theta
    shots = args.shots    
    
    # Start memory profiling
    tracemalloc.start()
    # Record the start time
    start_time = time.time()
    
    phases = run_DQFT(num_qubits, num_nodes, theta, shots, verbose)

    # Record the end time
    end_time = time.time()
    # Calculate execution time
    execution_time = end_time - start_time
    # Get memory usage
    current, peak = tracemalloc.get_traced_memory()
    # Stop memory profiling
    tracemalloc.stop()
    
    # Print or save the results
    print(f"Execution Time: {execution_time} seconds")
    print(f"Current Memory Usage: {current / 1024} KB")
    print(f"Peak Memory Usage: {peak / 1024} KB")
    print(f"Result: {phases}")
    
    # Fidelity against the ideal inverse-QFT distribution, computed here where
    # the raw counts exist rather than reconstructed later from the CSV.
    fidelity = fidelity_from_counts(phases, num_qubits, theta)
    print(f"Fidelity: {fidelity}")

    csv_file_path = write_result_row(
        emulator=emulator, num_nodes=num_nodes, num_qubits=num_qubits,
        theta=theta, shots=shots, execution_time=execution_time,
        current_kb=current / 1024, peak_kb=peak / 1024,
        result=phases, algorithm="qft", fidelity=fidelity,
    )
    print(f"Wrote {csv_file_path}")
    
