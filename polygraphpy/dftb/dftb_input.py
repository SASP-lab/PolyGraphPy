import os
from pathlib import Path
from datetime import datetime
from typing import Set, Dict
from polygraphpy.core.simulator import Simulator

class DFTBInputGenerator(Simulator):
    """Generate DFTB+ input files (dftb_in.hsd) for .xyz files."""
    
    def __init__(self, xyz_dir: str = 'polygraphpy/data/xyz_files', molecules_dir: str = 'polygraphpy/data/molecules',
                 sk_dir: str = '3ob-3-1', log_file: str = 'dftb_pipeline.log'):
        """Initialize with directories and log file."""
        super().__init__()
        self.xyz_dir = xyz_dir
        self.molecules_dir = molecules_dir
        self.sk_dir = sk_dir
        self.log_file = log_file
        os.makedirs(molecules_dir, exist_ok=True)
        with open(log_file, 'w') as log:
            log.write(f"Starting DFTB+ input generation at {datetime.now()}\n")
    
    def get_angular_momentum(self, element: str) -> str:
        """Get MaxAngularMomentum for an element."""
        s_elements = {"H", "Li", "Na", "K", "Rb", "Cs", "Fr", "He", "Ne", "Ar", "Kr", "Xe", "Rn"}
        p_elements = {"Be", "Mg", "Ca", "Sr", "Ba", "Ra", "B", "Al", "Ga", "In", "Tl", "C", "Si", "Ge", "Sn", "Pb", "N", "P", "As", "Sb", "Bi", "O", "Se", "Te", "Po"}
        d_elements = {"F", "Cl", "Br", "I", "At", "S", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Ac", "Th", "Pa", "U", "Np", "Pu", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr"}
        
        if element in s_elements:
            return "s"
        elif element in p_elements:
            return "p"
        elif element in d_elements:
            return "d"
        else:
            with open(self.log_file, "a") as log:
                log.write(f"Warning: Unknown element {element}, using default 'p' at {datetime.now()}\n")
            return "p"
    
    def prepare_input(self, xyz_file: str) -> bool:
        """Generate dftb_in.hsd for a single .xyz file."""
        try:
            base_name = Path(xyz_file).stem
            job_dir = os.path.join(self.molecules_dir, base_name)
            hsd_file = os.path.join(job_dir, "dftb_in.hsd")
            job_log = os.path.join(job_dir, "process.log")
            
            if not os.access(xyz_file, os.R_OK):
                with open(self.log_file, "a") as log:
                    log.write(f"Error: Cannot read {xyz_file} at {datetime.now()}\n")
                return False
            
            os.makedirs(job_dir, exist_ok=True)
            with open(job_log, "w"):
                pass
            
            elements: Set[str] = set()
            with open(xyz_file, "r") as f:
                lines = f.readlines()
                natoms = int(lines[0].strip())
                for line in lines[2:2 + natoms]:
                    if line.strip() and len(line.split()) > 0:
                        elements.add(line.split()[0])
            
            if not elements:
                with open(self.log_file, "a") as log:
                    log.write(f"Error: No elements found in {xyz_file} at {datetime.now()}\n")
                with open(job_log, "a") as log:
                    log.write(f"Error: No elements found in {xyz_file} at {datetime.now()}\n")
                return False
            
            hsd_content = f"""Geometry = xyzFormat {{
   <<< '../../../../{self.xyz_dir}/{base_name}.xyz'
}}

Driver = GeometryOptimization {{
   Optimizer = Rational {{}}
   MaxSteps = 10000
   OutputPrefix = '{base_name}'
   Convergence {{ GradElem = 1E-4 }}
}}

Hamiltonian = DFTB {{
   SCC = Yes
   SCCTolerance = 1e-9
   MaxSCCIterations = 1000

   MaxAngularMomentum = {{
"""
            for element in sorted(elements):
                momentum = self.get_angular_momentum(element)
                hsd_content += f'      {element} = "{momentum}"\n'
            hsd_content += f"""   }}
        
   SlaterKosterFiles = Type2FileNames {{
      Prefix = '../../../../{self.sk_dir}/'
      Separator = '-'
      Suffix = '.skf'
      LowerCaseTypeName = No
   }}

   Filling = Fermi {{
    Temperature [K] = 300
   }}
}}

Analysis = {{
  Polarisability = {{
    Static = Yes
    }}
}}

ElectronDynamics = {{
   Steps = 25000
   TimeStep [au] = 0.1
   Perturbation = Kick {{
     PolarizationDirection = all
   }}
   FieldStrength [v/a] = 0.001
}}

ParserOptions {{
   ParserVersion = 14
}}
"""
            with open(hsd_file, "w") as f:
                f.write(hsd_content)
            return True
        
        except Exception as e:
            with open(self.log_file, "a") as log:
                log.write(f"Error generating input for {xyz_file} at {datetime.now()}: {str(e)}\n")
            return False
    
    def run(self) -> None:
        """Not implemented; input generation only."""
        raise NotImplementedError("Use DFTBSimulation to run simulations.")
    
    def process_output(self) -> None:
        """Not implemented; output processing handled elsewhere."""
        raise NotImplementedError("Use PolarizabilityTrace to process outputs.")