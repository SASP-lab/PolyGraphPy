"""
polygraphpy.dftb.dftb_input
============================

This module contains the `DFTBInputGenerator` class, which is responsible
for creating the necessary input files (`dftb_in.hsd`) for DFTB+
simulations. The generator takes an `.xyz` file as input and configures the
DFTB+ parameters, including atomic species, angular momentum basis sets,
Slater-Koster files, and calculation types (e.g., geometry optimization,
polarizability analysis).
"""

import os
from pathlib import Path
from datetime import datetime
from typing import Set
from polygraphpy.core.simulator import Simulator

class DFTBInputGenerator(Simulator):
    """Generates DFTB+ input files (`dftb_in.hsd`) for molecular systems.

    This class extends the `Simulator` base class and provides methods to
    create a `dftb_in.hsd` file for a given `.xyz` molecular structure. It
    automatically determines the maximum angular momentum for each element
    and configures DFTB+ for geometry optimization and static polarizability
    calculations.

    :param xyz_dir: Directory containing the input `.xyz` files.
                    Defaults to 'polygraphpy/data/xyz_files'.
    :type xyz_dir: str, optional
    :param molecules_dir: Directory where individual molecule job directories
                          (containing `dftb_in.hsd` and logs) will be created.
                          Defaults to 'polygraphpy/data/molecules'.
    :type molecules_dir: str, optional
    :param sk_dir: Directory containing the Slater-Koster files (e.g., '3ob-3-1').
                   This path is relative to the `share/dftb+` installation directory.
                   Defaults to '3ob-3-1'.
    :type sk_dir: str, optional
    :param log_file: Path to the log file for recording input generation events and errors.
                     Defaults to 'dftb_pipeline.log'.
    :type log_file: str, optional
    """
    
    def __init__(self, xyz_dir: str = 'polygraphpy/data/xyz_files', molecules_dir: str = 'polygraphpy/data/molecules',
                 sk_dir: str = '3ob-3-1', log_file: str = 'dftb_pipeline.log'):
        """Initializes the DFTBInputGenerator.
        """
        super().__init__()
        self.xyz_dir = xyz_dir
        self.molecules_dir = molecules_dir
        self.sk_dir = sk_dir
        self.log_file = log_file
        os.makedirs(molecules_dir, exist_ok=True)
        with open(log_file, 'w') as log:
            log.write(f"Starting DFTB+ input generation at {datetime.now()}\n")
    
    def get_angular_momentum(self, element: str) -> str:
        """Determines the maximum angular momentum basis for a given element.

        This method maps common elements to their typical highest angular
        momentum orbitals (s, p, or d) used in DFTB+ calculations.

        :param element: The atomic symbol (e.g., "H", "C", "Fe").
        :type element: str
        :return: The string representing the maximum angular momentum ("s", "p", or "d").
        :rtype: str
        """
        s_elements = {"H", "Li", "Na", "K", "Rb", "Cs", "Fr", "He", "Ne", "Ar", "Kr", "Xe", "Rn"}
        p_elements = {"Be", "Mg", "Ca", "Sr", "Ba", "Ra", "B", "Al", "Ga", 
                      "In", "Tl", "C", "Si", "Ge", "Sn", "Pb", "N", "P", 
                      "As", "Sb", "Bi", "O", "Se", "Te", "Po"}
        d_elements = {"F", "Cl", "Br", "I", "At", "S", "Sc", "Ti", "V", 
                      "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Y", "Zr", "Nb", 
                      "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "La", "Ce", "Pr", 
                      "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", 
                      "Yb", "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", 
                      "Hg", "Ac", "Th", "Pa", "U", "Np", "Pu", "Cm", "Bk", "Cf", 
                      "Es", "Fm", "Md", "No", "Lr"}
        
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
        """Generates the `dftb_in.hsd` input file for a single .xyz molecular structure.

        This method reads the `.xyz` file to determine the elements present,
        configures the DFTB+ calculation settings (geometry optimization, SCC,
        polarizability), and writes the HSD file to a dedicated job directory.

        :param xyz_file: Path to the input `.xyz` file for which to generate the DFTB+ input.
        :type xyz_file: str
        :return: True if the input file was successfully generated or already exists, False otherwise.
        :rtype: bool
        """
        try:
            base_name = Path(xyz_file).stem
            job_dir = os.path.join(self.molecules_dir, base_name)
            hsd_file = os.path.join(job_dir, "dftb_in.hsd")
            if os.path.exists(hsd_file):
                with open(self.log_file, "a") as log:
                    log.write(f"Skipping {xyz_file}: Input file already exists at {datetime.now()}\n")
                return True
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
            
            hsd_content = f"""
Geometry = xyzFormat {{
   <<< '../../../../{self.xyz_dir}/{base_name}.xyz'
}}

Driver = GeometryOptimization {{
   Optimizer = Rational {{}}
   MaxSteps = 2000
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
        """This method is not implemented in `DFTBInputGenerator`.

        Input generation is a preparation step, and the actual simulation
        execution is handled by the `DFTBSimulation` class.

        :raises NotImplementedError: Always, as this method is not intended for use here.
        """
        raise NotImplementedError("Use DFTBSimulation to run simulations.")
    
    def process_output(self) -> None:
        """This method is not implemented in `DFTBInputGenerator`.

        Output processing of DFTB+ results is handled by the
        `PolarizabilityTrace` class.

        :raises NotImplementedError: Always, as this method is not intended for use here.
        """
        raise NotImplementedError("Use PolarizabilityTrace to process outputs.")