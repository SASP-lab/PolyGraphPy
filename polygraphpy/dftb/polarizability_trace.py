import os
import numpy as np
import pandas as pd
import multiprocessing
from joblib import Parallel, delayed
from polygraphpy.core.simulator import Simulator
from tqdm import tqdm

class PolarizabilityTrace(Simulator):
    """Compute and save polarizability traces from DFTB+ output."""
    
    def __init__(self, molecules_dir: str = 'polygraphpy/data/molecules'):
        """Initialize with molecules directory."""
        super().__init__()
        self.molecules_dir = molecules_dir
    
    def prepare_input(self, input_data: str) -> None:
        """Input preparation not required."""
        raise NotImplementedError("Input preparation not required for trace computation.")
    
    def read_polarizability(self, file_path: str) -> np.ndarray:
        """Read electric polarizability from detailed.out file."""
        polarizability = []
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()
                for i, line in enumerate(lines):
                    if "Electric polarisability (a.u.)" in line:
                        for j in range(i + 1, i + 4):
                            values = [float(x) for x in lines[j].split() if x]
                            polarizability.append(values)
                        break
            if len(polarizability) == 3:
                return np.array(polarizability)
            else:
                print(f"Warning: Could not parse polarizability in {file_path}")
                return None
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return None
    
    def compute_trace(self, matrix: np.ndarray) -> tuple:
        """Compute trace and diagonal elements of a 3x3 matrix."""
        if matrix is not None and matrix.shape == (3, 3):
            xx = matrix[0, 0]
            yy = matrix[1, 1]
            zz = matrix[2, 2]
            trace = np.trace(matrix) / 3
            return xx, yy, zz, trace
        return None, None, None, None
    
    def write_trace_file(self, subfolder_path: str, xx: float, yy: float, zz: float, trace: float) -> None:
        """Write trace result to trace.txt file."""
        output_file = os.path.join(subfolder_path, "trace.txt")
        try:
            with open(output_file, 'w') as f:
                f.write("File Path, xx, yy, zz, Polarizability Trace (a.u.)\n")
                rel_path = os.path.basename(subfolder_path) + "/detailed.out"
                f.write(f"{rel_path}, {xx:.6f}, {yy:.6f}, {zz:.6f}, {trace:.6f}\n")
        except Exception as e:
            print(f"Error writing to {output_file}: {e}")
    
    def process_subfolder(self, subfolder_path: str, root_folder: str) -> list:
        """Process detailed.out file in a single subfolder."""
        results = []
        for file in os.listdir(subfolder_path):
            if file == "detailed.out":
                file_path = os.path.join(subfolder_path, file)
                polarizability_matrix = self.read_polarizability(file_path)
                if polarizability_matrix is not None:
                    xx, yy, zz, trace = self.compute_trace(polarizability_matrix)
                    if trace is not None:
                        self.write_trace_file(subfolder_path, xx, yy, zz, trace)
                        rel_path = os.path.relpath(file_path, root_folder)
                        # Extract chain_size and id from subfolder name
                        folder_name = os.path.basename(subfolder_path)
                        chain_size = None
                        mol_id = None
                        if folder_name.startswith('homopoly_'):
                            chain_size_match = pd.Series([folder_name]).str.extract(r'homopoly_\d+_chain_(\d+)')
                            chain_size = int(chain_size_match[0][0]) if not chain_size_match.empty else None
                            id_match = pd.Series([folder_name]).str.extract(r'homopoly_(\d+)_chain_\d+')
                            mol_id = int(id_match[0][0]) if not id_match.empty else None
                        elif folder_name.startswith('monomer_'):
                            chain_size = 0
                            id_match = pd.Series([folder_name]).str.extract(r'monomer_(\d+)')
                            mol_id = int(id_match[0][0]) if not id_match.empty else None
                        results.append((rel_path, xx, yy, zz, trace, chain_size, mol_id))
        return results
    
    def run(self, input_csv: str) -> list:
        """Process all subfolders in parallel to compute traces."""
        if not os.path.exists(self.molecules_dir):
            print(f"Error: The folder '{self.molecules_dir}' does not exist.")
            return []
        
        subfolders = [os.path.join(self.molecules_dir, d) for d in os.listdir(self.molecules_dir)
                      if os.path.isdir(os.path.join(self.molecules_dir, d))]
        
        num_cores = multiprocessing.cpu_count()
        results = Parallel(n_jobs=num_cores)(
            delayed(self.process_subfolder)(subfolder, self.molecules_dir) for subfolder in tqdm(subfolders)
        )
        
        trace_results = [item for sublist in results for item in sublist]
        if not trace_results:
            print("No valid results to process.")
            return []

        # Create polarizability dataset
        df_polarizability = pd.DataFrame(trace_results, columns=['file_path', 'xx', 'yy', 'zz', 'static_polarizability', 'chain_size', 'id'])
        # Filter out rows where id or chain_size could not be extracted
        df_polarizability = df_polarizability.dropna(subset=['id', 'chain_size'])
        if df_polarizability.empty:
            print("Error: No valid IDs or chain sizes extracted from file paths.")
            return trace_results
        df_polarizability['id'] = df_polarizability['id'].astype(int)
        df_polarizability['chain_size'] = df_polarizability['chain_size'].astype(int)
        df_polarizability = df_polarizability.merge(pd.read_csv(input_csv), on=['id'], how='left')

        desired_columns = ['id', 'poly', 'name', 'smiles', 'chain_size', 'xx', 'yy', 'zz', 'static_polarizability']
        available_columns = [col for col in desired_columns if col in df_polarizability.columns]
        df_polarizability = df_polarizability[available_columns]

        df_polarizability.to_csv('polygraphpy/data/polarizability_data.csv', index=False)

        return trace_results
    
    def process_output(self, input_csv: str) -> list:
        """Return results from run method."""
        return self.run(input_csv)