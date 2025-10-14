import logging
import numpy as np
import argparse
import csv

from netqasm.sdk import Qubit
from typing import List
from collections import Counter

import time
import tracemalloc

from squidasm.run.stack.run import run
from squidasm.sim.stack.program import ProgramContext
from squidasm.run.stack.config import StackNetworkConfig
from squidasm.run.stack.config import SimulationConfig, SimulationMode

from AbstractNode import AbstractNode


class AliceProgram(AbstractNode):

    def __init__(
            self,
            *args,
            theta,
            dim_operator: int = 1,
            verbose: bool = False,
            **kwargs
            ):

        super().__init__(*args, **kwargs)
        self.theta = theta
        self.dim_qft = self.max_qubits 
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
            qubit.rot_Z(angle=theta*np.pi*2**(i+1+start_from)) 
    
            if verbose:
                print(f"{self.name}: Rz({theta*np.pi*2**(i+1+start_from)}, |state>)")

    def run(self, context: ProgramContext):
        self.context = context

        # Initialise qubits
        [self.init_qubit(i) for i in range(self.max_qubits)]

        qubits_qft = self.qubits
        theta = self.theta


        # Prepare Fourier state
        self._fourier_state(
            qubits_qft=qubits_qft,
            theta=theta,
            start_from=0,
            verbose=self.verbose
        )

        # Distributed inverse QFT
        self.inverse_QFT(qubits_qft, verbose=self.verbose)
        """ END QPE """
        # Measure qubits
        output = []
        for i, q in enumerate(qubits_qft):
            if self.verbose:
                print(f"{self.name}: Measure({i})")
            result = q.measure()
            output += [result]
        yield from self.connection.flush()

        return [int(o) for o in output]


def run_QFT_local(num_qft_qubits, partitions, theta, shots, config_file, verbose):
        # cfg = StackNetworkConfig(stacks=[alice], links=[])
        cfg = StackNetworkConfig.from_file(config_file)
        cfg.stacks[0].qdevice_cfg['num_qubits'] = num_qft_qubits + 1

        # Create instances of programs to run
        alice_program = AliceProgram(
            name="Alice",
            peers=[],
            theta=theta,
            qubits=num_qft_qubits,
            verbose=verbose
            )

        # Toggle logging
        alice_program.logger.setLevel(logging.ERROR)      

        # Run the simulation
        out = run(
            config=cfg,
            programs={"Alice": alice_program},
            num_times=shots,
            sim_config=sim_config  # pass the multithreaded mode here
            )
        out = np.array(out[0]) # Only one stack!
        bitstrings = ["".join(map(str, row[::-1])) for row in out] # Classical reverse!

        return dict(Counter([int(bitstring, 2) / (2 ** num_qft_qubits) for bitstring in bitstrings]))
        
###############################################
if __name__ == "__main__":

    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Run QFT on a single node.")
    parser.add_argument('--num_qubits', type=int, required=True, help='Number of qubits')
    parser.add_argument('--theta', type=float, required=True, help='Initial angle theta')
    parser.add_argument('--shots', type=int, required=True, help='Number of shots')
#     parser.add_argument("--mode", default="multithread")  # or "stack"
    sim_config = SimulationConfig(sim_mode=SimulationMode.MULTI_THREADED)


    # Parse the arguments
    args = parser.parse_args()
    
    emulator = "squidasm"
    partitions = 1
    verbose = False
    config_file = '../scripts/config-QFT-noiseless.yaml'
    num_qubits = args.num_qubits
    theta = args.theta
    shots = args.shots    
    

    tracemalloc.start(); start_time = time.time()  # Start memory profiling and record the start time
    phases = run_QFT_local(num_qubits, partitions, theta, shots, config_file, verbose)  # Run main script
    execution_time = time.time() - start_time  # Calculate execution time
    current, peak = tracemalloc.get_traced_memory()  # Get memory usage
    tracemalloc.stop()  # Stop memory profiling
    
    # Print or save the results
    print(f"Execution Time: {execution_time} seconds")
    print(f"Current Memory Usage: {current / 1024} KB")
    print(f"Peak Memory Usage: {peak / 1024} KB")
    print(f"Result: {phases}")
       
    # Define the CSV file path
    csv_file_path = f"../results/{emulator}_{partitions}_q{num_qubits}_a{theta}.csv"
    
    # Write the results to a CSV file
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write the header
        writer.writerow(["Emulator", "Nodes", "Num Qubits", "Theta", "Shots",
                         "Execution Time (s)", "Current Memory (KB)", "Peak Memory (KB)", "Result"])
        # Write the data
        writer.writerow([emulator, partitions, num_qubits, theta, shots,
                         execution_time, current / 1024, peak / 1024, phases])
    
