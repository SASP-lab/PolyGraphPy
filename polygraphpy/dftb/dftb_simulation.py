import os
import glob
import subprocess
import shutil
from multiprocessing import Pool
from pathlib import Path
from datetime import datetime
from polygraphpy.core.simulator import Simulator

class DFTBSimulation(Simulator):
    """Run DFTB+ simulations for .xyz files."""
    
    def __init__(self, xyz_dir: str = 'polygraphpy/data/xyz_files', molecules_dir: str = 'polygraphpy/data/molecules',
                 log_file: str = 'dftb_pipeline.log', processes: int = 8,
                 dftbplus_path: str = None):
        """Initialize with directories, log file, number of processes, and optional DFTB+ path."""
        super().__init__()
        self.xyz_dir = xyz_dir
        self.molecules_dir = molecules_dir
        self.log_file = log_file
        self.processes = processes
        
        # Set OMP_NUM_THREADS environment variable
        os.environ['OMP_NUM_THREADS'] = '1'
        
        # Find and set DFTB+ executable path
        self.dftbplus_cmd = self._find_dftbplus(dftbplus_path)
        os.environ['DFTBPLUS_PATH'] = os.path.dirname(self.dftbplus_cmd[0])
    
    def _find_dftbplus(self, dftbplus_path: str = None) -> list:
        """Find DFTB+ executable, using provided path or searching system."""
        # Check if user provided a path
        if dftbplus_path and os.path.isfile(dftbplus_path):
            try:
                subprocess.run([dftbplus_path, "--version"], capture_output=True, check=True)
                return [dftbplus_path]
            except (subprocess.CalledProcessError, FileNotFoundError):
                with open(self.log_file, "a") as log:
                    log.write(f"Error: Provided DFTB+ path {dftbplus_path} is invalid at {datetime.now()}\n")
                raise ValueError(f"Invalid DFTB+ executable path: {dftbplus_path}")
        
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
        """Input preparation handled by DFTBInputGenerator."""
        raise NotImplementedError("Use DFTBInputGenerator to prepare inputs.")
    
    def process_xyz(self, xyz_file: str) -> None:
        """Process a single .xyz file by running DFTB+."""
        try:
            base_name = Path(xyz_file).stem
            job_dir = os.path.join(self.molecules_dir, base_name)
            hsd_file = os.path.join(job_dir, "dftb_in.hsd")
            job_log = os.path.join(job_dir, "process.log")

            os.makedirs(job_dir, exist_ok=True) 
            
            if not os.path.exists(hsd_file):
                with open(self.log_file, "a") as log:
                    log.write(f"Error: Input file {hsd_file} not found at {datetime.now()}\n")
                return
            
            result = subprocess.run(
                self.dftbplus_cmd + [hsd_file],
                capture_output=True,
                text=True,
                cwd=job_dir,
                timeout=6000
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
        finally:
            os.chdir(os.path.dirname(os.path.abspath(__file__)) or ".")
    
    def run(self) -> None:
        """Run DFTB+ simulations for all .xyz files in parallel."""
        xyz_files = glob.glob(os.path.join(self.xyz_dir, "*.xyz"))
        if not xyz_files:
            with open(self.log_file, "a") as log:
                log.write(f"Error: No .xyz files found in {self.xyz_dir} at {datetime.now()}\n")
            print(f"Error: No .xyz files found in {self.xyz_dir}")
            return
        
        print(f"Found {len(xyz_files)} .xyz files to process")
        with open(self.log_file, "a") as log:
            log.write(f"Found {len(xyz_files)} .xyz files to process at {datetime.now()}\n")
        
        with Pool(processes=self.processes) as pool:
            pool.map(self.process_xyz, xyz_files)
        
        with open(self.log_file, "a") as log:
            log.write(f"All DFTB+ jobs completed at {datetime.now()}\n")
        print("All DFTB+ jobs completed")
    
    def process_output(self) -> None:
        """Not implemented; output processing handled by PolarizabilityTrace."""
        raise NotImplementedError("Use PolarizabilityTrace to process outputs.")