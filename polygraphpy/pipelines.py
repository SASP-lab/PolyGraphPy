import os
import importlib.resources as resources
from polygraphpy.dftb.smiles_to_xyz import MonomerXyzGenerator, PolymerXyzGenerator
from polygraphpy.dftb.dftb_input import DFTBInputGenerator
from polygraphpy.dftb.dftb_simulation import DFTBSimulation
from polygraphpy.dftb.polarizability_trace import PolarizabilityTrace

def run_dftb_pipeline(input_csv: str = None, is_polymer: bool = False, 
                      dftbplus_path: str = None, use_example_data: bool = False, polymer_chain_size: int = 2):
    """Run the full DFTB+ pipeline."""
    # Step 1: Generate .xyz files
    if use_example_data:
        with resources.path("polygraphpy.data", "reduced_dataset.csv") as csv_path:
            input_csv = str(csv_path)
    
    if input_csv is None:
        raise ValueError("input_csv must be provided unless use_example_data is True")
    
    if is_polymer:
        xyz_generator = PolymerXyzGenerator(input_csv, polymer_chain_size=polymer_chain_size)
    else:
        xyz_generator = MonomerXyzGenerator(input_csv)
    xyz_results = xyz_generator.generate()
    print(f"XYZ generation complete: {sum('Saved' in r for r in xyz_results)} files created")
    
    # Step 2: Generate DFTB+ input files
    input_generator = DFTBInputGenerator()
    xyz_files = [os.path.join('polygraphpy/data/xyz_files', f) for f in os.listdir('polygraphpy/data/xyz_files') if f.endswith('.xyz')]
    input_results = [input_generator.prepare_input(xyz_file) for xyz_file in xyz_files]
    print(f"Input generation complete: {sum(input_results)} inputs created")
    
    # Step 3: Run DFTB+ simulations
    simulation = DFTBSimulation(dftbplus_path=dftbplus_path)
    simulation.run()
    
    # Step 4: Compute polarizability traces
    trace_processor = PolarizabilityTrace()
    trace_results = trace_processor.run(input_csv)
    print(f"Trace computation complete: {len(trace_results)} traces computed")
    return trace_results