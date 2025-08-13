"""
polygraphpy.dftb.dftb_simulation
=================================

This module provides the `DFTBSimulation` class for executing DFTB+ simulations
on a collection of molecular structures. It manages the execution environment,
handles subprocess calls to the DFTB+ executable, and logs the simulation
status. The simulations are run in parallel for efficiency.
"""

import os
import glob
import subprocess
from multiprocessing import Pool
from pathlib import Path
from datetime import datetime
from polygraphpy.core.simulator import Simulator

class DFTBSimulation(Simulator):
    """Manages and executes DFTB+ simulations for a set of molecules.

    This class orchestrates the running of DFTB+ jobs, typically after
    input files have been prepared by `DFTBInputGenerator`. It handles
    finding the DFTB+ executable, setting up the environment, running
    simulations in parallel, and logging outcomes.

    :param xyz_dir: Directory containing the original `.xyz` files.
                    Defaults to 'polygraphpy/data/xyz_files'.
    :type xyz_dir: str, optional
    :param molecules_dir: Base directory where individual molecule job folders
                          (containing `dftb_in.hsd`, output, and logs) are located.
                          Defaults to 'polygraphpy/data/molecules'.
    :type molecules_dir: str, optional
    :param log_file: Path to the log file for recording simulation events and errors.
                     Defaults to 'dftb_pipeline.log'.
    :type log_file: str, optional
    :param processes: Number of parallel processes (CPU cores) to use for running DFTB+ jobs.
                      Defaults to 20.
    :type processes: int, optional
    :param dftbplus_path: Optional: Explicit path to the DFTB+ executable. If not provided,
                          the system's PATH and DFTBPLUS_PATH environment variable will be checked.
    :type dftbplus_path: str, optional
    """
    
    def __init__(self, xyz_dir: str = 'polygraphpy/data/xyz_files', molecules_dir: str = 'polygraphpy/data/molecules',
                 log_file: str = 'dftb_pipeline.log', processes: int = 20,
                 dftbplus_path: str = None):
        """Initializes the DFTBSimulation.
        """
        super().__init__()
        self.xyz_dir = xyz_dir
        self.molecules_dir = molecules_dir
        self.log_file = log_file
        self.processes = processes
        
        # Set OMP_NUM_THREADS environment variable
        os.environ['OMP_NUM_THREADS'] = '1'
        os.environ['DFTBPLUS_PATH'] = dftbplus_path
        
        # Find and set DFTB+ executable path
        self.dftbplus_cmd = self._find_dftbplus(dftbplus_path)
    
    def _find_dftbplus(self, dftbplus_path: str = None) -> list:
        """Locates the DFTB+ executable.

        It first checks the path provided by `dftbplus_path`, then the
        `DFTBPLUS_PATH` environment variable, and finally the system's PATH.

        :param dftbplus_path: An explicit path to the DFTB+ executable.
        :type dftbplus_path: str, optional
        :raises SystemExit: If the dftb+ executable cannot be found.
        :return: A list containing the full path to the DFTB+ executable.
        :rtype: list
        """
        if dftbplus_path and os.path.isfile(dftbplus_path):
            return [dftbplus_path]

        # Check DFTBPLUS_PATH environment variable
        env_path = os.environ.get("DFTBPLUS_PATH", None)
        if env_path and os.path.isfile(os.path.join(env_path, "dftb+")):
            full_path = os.path.join(env_path, "dftb+")
            return [full_path]
        
        # If no valid executable is found, raise an error
        with open(self.log_file, "a") as log:
            log.write("Error: dftb+ not found. Provide a valid path or set DFTBPLUS_PATH environment variable\n")
        raise SystemExit("Error: dftb+ executable not found")
    
    def prepare_input(self, input_data: str) -> None:
        """This method is not implemented in `DFTBSimulation`.

        Input preparation for DFTB+ is handled by the `DFTBInputGenerator` class.

        :param input_data: Placeholder parameter (not used).
        :type input_data: str
        :raises NotImplementedError: Always, as this method is not intended for use here.
        """
        raise NotImplementedError("Use DFTBInputGenerator to prepare inputs.")
    
    def process_xyz(self, xyz_file: str) -> None:
        """Runs a DFTB+ simulation for a single `.xyz` file.

        This method changes to the molecule's job directory, executes the
        `dftb+` command, and logs the output and any errors. It also checks
        if the simulation has already completed successfully.

        :param xyz_file: The full path to the `.xyz` file for which to run the simulation.
        :type xyz_file: str
        :return: None
        :rtype: None
        """
        try:
            base_name = Path(xyz_file).stem
            job_dir = os.path.join(self.molecules_dir, base_name)
            hsd_file = os.path.join(job_dir, "dftb_in.hsd")
            job_log = os.path.join(job_dir, "process.log")

            detailed_out = os.path.join(job_dir, "detailed.out")
            if os.path.exists(detailed_out):
                with open(detailed_out, "r") as f:
                    # Check for successful completion by looking for polarizability data
                    if "Electric polarisability (a.u.)" in f.read():
                        with open(self.log_file, "a") as log:
                            log.write(f"Skipping {xyz_file}: detailed.out with polarizability exists at {datetime.now()}\n")
                        return
                        
            if not os.path.exists(hsd_file):
                with open(self.log_file, "a") as log:
                    log.write(f"Error: Input file {hsd_file} not found for {xyz_file} at {datetime.now()}\n")
                return
            
            # Run DFTB+ as a subprocess
            result = subprocess.run(
                self.dftbplus_cmd + [hsd_file],
                capture_output=True,
                text=True,
                cwd=job_dir,
                timeout=60000
            )

            if result.returncode == 0:
                with open(job_log, "a") as log:
                    log.write(f"Successfully completed DFTB+ for {xyz_file} at {datetime.now()}\n")
            else:
                with open(job_log, "a") as log:
                    log.write(f"Error running DFTB+ for {xyz_file} at {datetime.now()}\n")
                    log.write(f"DFTB+ output for {xyz_file}:\n{result.stderr}\n")
        
        except (subprocess.TimeoutExpired, subprocess.SubprocessError, OSError) as e:
            with open(self.log_file, "a") as log:
                log.write(f"Error: Failed to run DFTB+ for {xyz_file} at {datetime.now()}: {str(e)}\n")
            with open(job_log, "a") as log:
                log.write(f"Error: Failed to run DFTB+ for {xyz_file} at {datetime.now()}: {str(e)}\n")
    
    def run(self) -> None:
        """Executes DFTB+ simulations for all available `.xyz` files in parallel.

        This is the main method to start the DFTB+ calculations. It identifies
        all `.xyz` files in the specified `xyz_dir`, and then uses a
        multiprocessing pool to run `process_xyz` for each file concurrently.

        :return: None
        :rtype: None
        """
        xyz_files = glob.glob(os.path.join(self.xyz_dir, "*.xyz"))
        if not xyz_files:
            with open(self.log_file, "a") as log:
                log.write(f"Error: No .xyz files found in {self.xyz_dir} at {datetime.now()}\n")
            print(f"Error: No .xyz files found in {self.xyz_dir}")
            return
        
        print(f"Found {len(xyz_files)} .xyz files to process")
        with open(self.log_file, "a") as log:
            log.write(f"Found {len(xyz_files)} .xyz files to process at {datetime.now()}\n")
        
        print('Running DFTB+.')
        with Pool(processes=self.processes) as pool:
            pool.map(self.process_xyz, xyz_files)
        
        with open(self.log_file, "a") as log:
            log.write(f"All DFTB+ jobs completed at {datetime.now()}\n")
        print("All DFTB+ jobs completed")
    
    def process_output(self) -> None:
        """This method is not implemented in `DFTBSimulation`.

        Processing of DFTB+ output files (e.g., extracting polarizability)
        is handled by the `PolarizabilityTrace` class.

        :raises NotImplementedError: Always, as this method is not intended for use here.
        """
        raise NotImplementedError("Use PolarizabilityTrace to process outputs.")