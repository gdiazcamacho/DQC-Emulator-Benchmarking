
import numpy as np
from qiskit import QuantumCircuit, transpile, QuantumRegister, ClassicalRegister
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_aer.noise import NoiseModel, depolarizing_error
from qiskit_aer import AerSimulator
from qiskit.result import marginal_counts
import string
import argparse
import csv

import time
import tracemalloc
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

def partition_list(lst, partition_spec):
    """
    Splits a list into partitions based on a partition specification.

    Args:
        lst (list): The input list to split.
        partition_spec (int or list of int): Either the number of equal partitions,
            or a list specifying the size of each partition.

    Returns:
        list of lists: A list containing sublists for each partition.

    Raises:
        ValueError: If the partitioning cannot be performed.
    """
    if isinstance(partition_spec, int):
        # Old behavior: number of equal partitions
        num_partitions = partition_spec
        if len(lst) % num_partitions != 0:
            raise ValueError("List length is not evenly divisible by the number of partitions.")
        partition_size = len(lst) // num_partitions
        return [lst[i * partition_size : (i + 1) * partition_size] for i in range(num_partitions)]

    elif isinstance(partition_spec, list):
        if sum(partition_spec) != len(lst):
            raise ValueError("Sum of partition sizes does not match list length.")

        result = []
        idx = 0
        for size in partition_spec:
            result.append(lst[idx : idx + size])
            idx += size
        return result

    else:
        raise TypeError("partition_spec must be an int or a list of ints.")

def distribute_qubits(num_qubits: int, num_nodes: int):
    """Distribute qubits across nodes, assigning the remainder to the last node."""
    base = (num_qubits + num_nodes - 1) // num_nodes  # ceiling division
    qubits = [base] * num_nodes
    total = sum(qubits)
    over = total - num_qubits

    # Remove 1 from the last 'over' nodes (starting from the end)
    for i in range(over):
        qubits[-(i + 1)] -= 1
        
    if any(q <= 1 for q in qubits):
        print("Warning: One or more nodes have ≤ 1 qubit.")
        
    return qubits
        
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

        target_nodes = [n for n in registers[::-1][i+1:]]
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
                                circ.crz(angles[k], control, target)

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

        target_nodes = [n for n in registers[::-1][i+1:]]
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

                        # The whole column of crz gates share the same control: only one EPR
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
                                circ.crz(angles[k], EPR_qubits[1], target)

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
    
    # Define the CSV file path
    csv_file_path = f"../results/{emulator}_{num_nodes}_q{num_qubits}_a{theta}.csv"
    
    # Write the results to a CSV file
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write the header
        writer.writerow(["Emulator", "Nodes", "Num Qubits", "Theta", "Shots",
                         "Execution Time (s)", "Current Memory (KB)", "Peak Memory (KB)", "Result"])
        # Write the data
        writer.writerow([emulator, num_nodes, num_qubits, theta, shots,
                         execution_time, current / 1024, peak / 1024, phases])
    
