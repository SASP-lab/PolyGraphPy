import os
import numpy as np
import pandas as pd
import multiprocessing
from joblib import Parallel, delayed
from polygraphpy.core.simulator import Simulator

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
    
    def compute_trace(self, matrix: np.ndarray) -> float:
        """Compute the trace of a 3x3 matrix."""
        if matrix is not None and matrix.shape == (3, 3):
            return np.trace(matrix) * 0.1481847 / 3
        return None
    
    def write_trace_file(self, subfolder_path: str, trace: float) -> None:
        """Write trace result to trace.txt file."""
        output_file = os.path.join(subfolder_path, "trace.txt")
        try:
            with open(output_file, 'w') as f:
                f.write("File Path, Polarizability Trace (a.u.)\n")
                rel_path = os.path.basename(subfolder_path) + "/detailed.out"
                f.write(f"{rel_path}, {trace:.6f}\n")
            print(f"Results written to {output_file}")
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
                    trace = self.compute_trace(polarizability_matrix)
                    if trace is not None:
                        self.write_trace_file(subfolder_path, trace)
                        rel_path = os.path.relpath(file_path, root_folder)
                        results.append((rel_path, trace))
        return results
    
    def run(self, input_csv) -> list:
        """Process all subfolders in parallel to compute traces."""
        if not os.path.exists(self.molecules_dir):
            print(f"Error: The folder '{self.molecules_dir}' does not exist.")
            return []
        
        subfolders = [os.path.join(self.molecules_dir, d) for d in os.listdir(self.molecules_dir)
                      if os.path.isdir(os.path.join(self.molecules_dir, d))]
        
        num_cores = multiprocessing.cpu_count()
        results = Parallel(n_jobs=num_cores)(
            delayed(self.process_subfolder)(subfolder, self.molecules_dir) for subfolder in subfolders
        )
        
        trace_results = [item for sublist in results for item in sublist]

        # Create static polarizability dataset
        df_polarizability = pd.DataFrame(trace_results)
        df_polarizability.columns = ['id', 'static_polarizability']
        df_polarizability['id'] = df_polarizability['id'].str.extract(r'monomer_(\d+)/')[0].astype(int)
        df_polarizability = df_polarizability.merge(pd.read_csv(input_csv), on=['id'])

        df_polarizability.to_csv('polygraphpy/data/polarizability_data.csv', index=False)

        return trace_results
    
    def process_output(self) -> list:
        """Return results from run method."""
        return self.run()