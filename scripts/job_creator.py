import os
import yaml

def create_jobs(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    emulators = config['emulators']
    num_qubits_range = config['num_qubits_range']
    num_nodes_range = config['num_nodes_range']
    theta_range = config['theta_range']
    shots = config['shots']

    job_count = 0  # Initialize job counter

    for emulator in emulators:
        for num_qubits in num_qubits_range:
            for theta in theta_range:
                for num_nodes in num_nodes_range:
                    job_name = f"{emulator}_{num_nodes}_q{num_qubits}_a{theta}"
                    slurm_script_path = f"emulators/{emulator}/jobs/{job_name}.sh"
                    slurm_template_path = f"emulators/{emulator}/slurm_template_multinode.sh"

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

if __name__ == "__main__":
    create_jobs('configs/sweep_params.yaml')

