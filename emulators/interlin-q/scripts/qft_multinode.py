import sys
import numpy as np
sys.path.append("../")

from qunetsim.components import Network
from qunetsim.objects import Logger

from interlinq import (ControllerHost, Constants, Clock,
Circuit, Layer, ComputingHost, Operation)

Logger.DISABLED = True

from collections import Counter
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
    distribute_qubits, fidelity_from_counts, write_result_row,
)
# ------------------------------------------------------------------------------
################################################################################

def phase_gate(theta):
    return np.array([[1, 0], [0, np.exp(1j * theta)]])

def fourier_state(q_map, theta, layers, verbose=False):
    """
    Prepares a local Fourier state on the given qubits.
    
    Parameters:
        q_ids (list): List of qubit IDs (must be on same QPU).
        computing_host_id (str): ID of the QPU (e.g. 'qpu0').
        theta (float): Phase parameter for the Fourier state.
        offset (int): qubit ID where the rotation should start.
        layers (list): List of Layer objects to append to.
    """
    # Apply H in all qubits
    ops = []
    for node in q_map:
        for q_id in q_map[node]:
            op = Operation(
                name=Constants.SINGLE,
                qids=[q_id],
                gate=Operation.H,
                computing_host_ids=[node])
            if verbose:
                print(op.gate, op.qids)  
            ops.append(op)
    layers.append(Layer(ops))

    # Apply rotation with theta*2^i where i is the global index of the qubit
    ops = []
    offset = 0 # Account for the previous nodes
    for node in q_map:
        for i, q_id in enumerate(q_map[node]):
            angle = theta * np.pi * (2 ** (i + 1 + offset))
            op = Operation(
                name=Constants.SINGLE,
                qids=[q_id],
                gate=Operation.RZ,
                gate_param=angle,
                computing_host_ids=[node])
            if verbose:
                print(op.gate, op.qids, op.gate_param)  
            ops.append(op)
        offset += len(q_id) # Add node qubits to the offset for the next node
    layers.append(Layer(ops))
    
    return layers


def inverse_quantum_fourier_transform(q_ids, node, layers, verbose=False):
    """
    Performs inverse quantum fourier transform
    """
    if verbose:
        print(f"\nStarting iQFT on qubits {q_ids} of node {node}")
    q_ids.reverse()

    for i in range(len(q_ids)):
        target_qubit_id = q_ids[i]

        for j in range(i):
            control_qubit_id = q_ids[j]

            op = Operation(
                name=Constants.TWO_QUBIT,
                qids=[control_qubit_id, target_qubit_id],
                gate=Operation.CUSTOM_CONTROLLED,
                gate_param=phase_gate(-np.pi / (2 ** (i - j))),
                computing_host_ids=[node])
            if verbose:
                print(op.gate, op.qids, 1/(2 ** (i - j)))  
            layers.append(Layer([op]))

        op = Operation(
            name=Constants.SINGLE,
            qids=[target_qubit_id],
            gate=Operation.H,
            computing_host_ids=[node])
        if verbose:
            print(op.gate, op.qids)  
        layers.append(Layer([op]))
    return layers

def test_circuit(q_map, theta, verbose=False):
    """
    Returns the monolithic circuit for quantum phase estimation
    algorithm
    """
    layers = []
    computing_host_ids = list(q_map.keys())

    # Prepare the qubits on both computing hosts
    ops = []
    for host_id in computing_host_ids:
        op = Operation(
            name=Constants.PREPARE_QUBITS,
            qids=q_map[host_id],
            computing_host_ids=[host_id])
        if verbose:
            print(op.name, op.qids)  
        ops.append(op)

    layers.append(Layer(ops))

    layers = fourier_state(q_map, theta, layers, verbose)
    
    # The recursive iQFT starts from the last node
    node_list = computing_host_ids[::-1]
    
    # List of qubits flattened, to have some global index. In an actual DQC experiment this should be passed to the nodes.
    all_qubits = [[q for q in q_map[node].copy()] for node in computing_host_ids]
    all_qubits = [item for sublist in all_qubits for item in sublist]
    
    # Now, controlled phase gradient block
    for control_node in node_list:
        q_ids = q_map[control_node].copy() # The qubit ids of the last register
        inverse_quantum_fourier_transform(q_ids, control_node, layers, verbose)
        if verbose:
            print(f'\nStarting Phase Gradient from node {control_node}')
        
        target_nodes = [n for n in node_list[node_list.index(control_node)+1:] ]  # All nodes before the control one
        if verbose:
            print(f'Target nodes: {target_nodes}')
        
        if target_nodes: # Are there any or is this the last node? (i.e. no targets)
            
            for t, target_node in enumerate(target_nodes):
                if verbose:
                    print(f'> Target Node: {target_node}')
                for i, control_qubit_id in enumerate(q_map[control_node].copy() ):
                    if verbose:
                        print(f'>> Control qubit: {control_qubit_id}')
                    # Global indices, which are needed to know the angle of the rotation
                    # Larger distance = Smaller rotation
                    control_index = all_qubits.index(control_qubit_id)
                    
    
    
                    for k, target_qubit_id in enumerate(q_map[target_node].copy() ):
                        target_index = all_qubits.index(target_qubit_id)
                        
                        if verbose:
                            print(f'>>> Target qubit: {target_qubit_id}')
                        op = Operation(
                            name=Constants.TWO_QUBIT,
                            qids=[control_qubit_id, target_qubit_id],
                            gate=Operation.CUSTOM_CONTROLLED,
                            gate_param=phase_gate(-np.pi / (2 ** (control_index - target_index))),
                            computing_host_ids=[control_node, target_node])
                        if verbose:
                            print(op.gate, op.qids, 1/(2 ** (control_index - target_index)))  
                        layers.append(Layer([op]))

    
    # Measure the qubits
    ops = []
    for node in q_map:
        for q_id in q_map[node]:
            op = Operation(
                name=Constants.MEASURE,
                qids=[q_id],
                cids=[q_id],
                computing_host_ids=[node])
            if verbose:
                print(op.name, op.qids)  
            ops.append(op)
    layers.append(Layer(ops))
    
    circuit = Circuit(q_map, layers)

    if verbose:
        print("Test circuit created. Listing all layers and operations:")
        for i, layer in enumerate(layers):
            for op in layer.operations:
                print(f"Layer {i}: {op.name} | qids={op.qids} | cids={op._cids} | host={op.computing_host_ids}")
    
    return circuit

def controller_host_protocol(host, q_map, theta, result_container, verbose=False):
    """
    Protocol for the controller host
    """
    if verbose:
        print(f"{host.host_id}: start circuit")
    circuit = test_circuit(q_map, theta, verbose)
    if verbose:
        print(f"{host.host_id}: finish circuit")
    host.generate_and_send_schedules(circuit)
    if verbose:
        print(f"{host.host_id}: sent schedules")
    host.receive_results()
    if verbose:
        print(f"{host.host_id}: receive resultss")
    
    results = host.results
    computing_host_ids = host.computing_host_ids

    if verbose:
        print('Final results: \n')
    
    # Collect all bits into one result dict
    all_bits = {}
    decimal_value = 0
    for computing_host_id in computing_host_ids:
        i = 0
        bits = results[computing_host_id]['bits']
        all_bits.update(bits)
        for bit_id, bit in bits.items():
            if verbose:
                print("{0}: {1}".format(bit_id, bit))
            decimal_value += (2 ** i) * bit
            i += 1
        if bits:
            phase = decimal_value / (2 ** len(bits.keys()))
            if verbose:
                print("\nThe estimated value of the phase is {0}".format(phase))

            # Store result (e.g., '000111' or similar)
    bitstring = ''.join(str(all_bits[q]) for q in sorted(all_bits.keys()))
    result_container.append(bitstring)  # Mutates shared list

def computing_host_protocol(host, verbose=False):
    """
    Protocol for the computing hosts
    """
    host.receive_schedule()
    if verbose:
        print(f"{host.host_id}: received schedule with {len(host._schedule)} layers")
    if verbose:
        for layer_index, operations in host._schedule.items():
            for op in operations:
                print(f"{host.host_id}: Layer {layer_index} -> {op['name']} on qids={op.get('qids', [])}, cids={op.get('cids', [])}")

    host.send_results()
    if verbose:
        print(f"{host.host_id}: sent results")



###############################

def run_DQFT(num_qft_qubits, num_nodes, theta, shots, verbose):

    results_list = []
    
    # Set number of qubits per host
    
    num_computing_hosts = num_nodes
    num_qubits_per_host = num_qft_qubits // num_nodes
    if num_qft_qubits % num_nodes > 0:
        print('Warning! The number of qubits is not divisible in equal parts between the hosts.')
        print('Interlin-q does not appear to support nodes with different sizes.')
        return 0

    else:
  
      for shot in range(shots):
      
          # initialize network... WHY DO WE HAVE TO DO THIS EVERYTIME? HOW TO RESET?
          network = Network.get_instance()
          network.delay = 0
          network.start()
          
          clock = Clock()
          
          controller_host = ControllerHost(
              host_id="host_1",
              clock=clock,
          )
          
          computing_hosts, q_map = controller_host.create_distributed_network(
              num_computing_hosts=num_computing_hosts,
              num_qubits_per_host=num_qubits_per_host)
          
          controller_host.start()
          network.add_hosts(computing_hosts + [controller_host])
          if verbose:
              print(f"Shot {shot}")
      
          # Shared result container (as a list for thread-safe mutation)
          shot_result = []
          
          t1 = controller_host.run_protocol(
                  controller_host_protocol,
                  (q_map, theta, shot_result, verbose))
          
          computing_threads = [h.run_protocol(computing_host_protocol) for h in computing_hosts]
      
          t1.join()
          for t in computing_threads:
              t.join()
              
          # Append result to global list
          if shot_result:
              results_list.append(shot_result[0])  # Only one value per run
          if verbose:
              print(f"This shot result {shot_result}")
      
      network.stop(True)
      
      print(f"Aggregated results {Counter(results_list)}")
      
      phases = {}
      for bitstring,value in Counter(results_list).items():
          n = len(bitstring)
          k = int(bitstring, 2) / (2 ** n)
          phases[k]=value
      print(f'Phases dictionary: {phases}')
      
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
    emulator = "interlin-q"
    verbose = False
    num_qubits = args.num_qubits
    num_nodes = args.num_nodes
    theta = args.theta
    shots = args.shots    

    if num_qubits % num_nodes > 0:
        print('Warning! The number of qubits is not divisible in equal parts between the hosts.')
        print('Interlin-q does not appear to support nodes with different sizes.')
        
    else:
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
      
      fidelity = fidelity_from_counts(phases, num_qubits, theta)
      print(f"Fidelity: {fidelity}")

      csv_file_path = write_result_row(
          emulator=emulator, num_nodes=num_nodes, num_qubits=num_qubits,
          theta=theta, shots=shots, execution_time=execution_time,
          current_kb=current / 1024, peak_kb=peak / 1024,
          result=phases, algorithm="qft", fidelity=fidelity,
      )
      print(f"Wrote {csv_file_path}")
