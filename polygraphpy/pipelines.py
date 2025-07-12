import os
import importlib.resources as resources
from polygraphpy.dftb.smiles_to_xyz import MonomerXyzGenerator, PolymerXyzGenerator
from polygraphpy.dftb.dftb_input import DFTBInputGenerator
from polygraphpy.dftb.dftb_simulation import DFTBSimulation
from polygraphpy.dftb.polarizability_trace import PolarizabilityTrace
from polygraphpy.gnn.pre_processing import PreProcess
from polygraphpy.gnn.train import Train
from polygraphpy.gnn.prediction import Prediction

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

def run_gnn_pipeline(input_csv: str = 'polygraph/data/polarizability_data.csv', batch_size: int = 8, learning_rate: float = 1e-3, number_conv_channels: int = 69, 
                     number_fc_channels: int = 69, prediction_target: str = None, polymer_type: str = 'monomer', epochs: int = 200,
                     train_input_data_path: str = 'polygraphpy/data/training_input_data/', gnn_output_path: str = 'polygraphpy/data/gnn_output/',
                     validation_data_path: str ='polygraphpy/data/validation_data/'):
    
    os.makedirs(train_input_data_path, exist_ok=True)
    os.makedirs(gnn_output_path, exist_ok=True)
    os.makedirs(validation_data_path, exist_ok=True)

    # Step 1: Pre processing data
    pre_process_engine = PreProcess(input_csv=input_csv, train_input_data_path=train_input_data_path, polymer_type=polymer_type, target=prediction_target)
    data = pre_process_engine.run()

    # # Step 2: Train GNN model for prediction
    # train_engine = Train(number_conv_channels, number_fc_channels, data, learning_rate, batch_size, epochs, train_input_data_path, gnn_output_path,
    #                      validation_data_path)
    # train_engine.run()

    # # Step 3: Plot validation result and save dataframes
    # prediction_engine = Prediction(validation_data_path, gnn_output_path)
    # prediction_engine.run()
